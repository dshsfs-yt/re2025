# model_validation.py — touchseq(JSON per sentence) ➜ 평가(KE-T5)
# 목적:
#  - 모델 출력(자모 시퀀스; 예: '왉'을 ㅇ ㅗ ㅏ ㄹ ㄱ처럼 기본 자모로 타이핑한 스트림)을
#    한글 음절로 자동 재조합한 뒤 GT 문장과 비교
#  - 지표: EM@1, EM@K, 평균 CER(문자), 평균 WER(단어)
#
# 사용 예:
#   python model_validation.py --json_dir saved_logs_nokk \
#       --model_dir ckpt/ke-t5-small-RnE2025_jamosplit --n 200 --topk 5 --beams 8

import argparse, json, random, unicodedata as ud, time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from krautomata import automata
# ==========================
# 0) 기본 상수
# ==========================
JSON_DIR_DEFAULT  = "saved_logs_nokk"
MODEL_DIR_DEFAULT = "ckpt/ke-t5-small-RnE2025_jamosplit"
MAX_SRC_LEN = 512
MAX_TGT_LEN = 256
SAMPLE_SIZE_DEFAULT = 200
SHOW_COUNT_DEFAULT  = 5
TOPK_DEFAULT        = 5
BATCH_INFER_DEFAULT = 32
SEED_DEFAULT        = 42

SPACE_LABEL = "[SPACE]"
MISS_LABEL  = "[MISS]"
BKSP_LABEL  = "[BKSP]"
PROMPT_FORMAT = "touchseq: {seq} -> text"

# ==========================
# Token mapping (must match model_training.py)
# ==========================
JAMO_TOKENS = ["ㅆ", "ㅃ", "ㅈ", "ㅕ", "ㅑ", "ㅖ", "ㅣ", "ㄸ", "ㅗ", "ㅌ", "ㅍ", "ㅒ", "ㅔ", "ㅏ", "ㅊ", "ㅓ", "ㅉ", "ㅛ", "ㅐ", "ㅁ",
               "ㅂ", "ㄲ", "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅅ", "ㅇ", "ㅋ", "ㅎ", "ㅜ", "ㅠ", "ㅡ"]

SPECIAL_TOKENS = ["[SPACE]", "[BKSP]", "[MISS]", "[FLUSH]", "@"]

# Build token mappings
TOKEN_TO_EXTRA_ID = {}
all_custom_tokens = JAMO_TOKENS + SPECIAL_TOKENS

for idx, token in enumerate(all_custom_tokens):
    TOKEN_TO_EXTRA_ID[token] = f"<extra_id_{idx}>"

# Reverse mapping for decoding
EXTRA_ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_EXTRA_ID.items()}

# ==========================
# 1) EOS stopping (batch)
# ==========================
class EosBatchStop(StoppingCriteria):
    def __init__(self, eos_id: Optional[int]):
        self.eos_id = eos_id
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.eos_id is None:
            return False
        hit = (input_ids == self.eos_id).any(dim=1)
        return bool(hit.all().item())

# ==========================
# 2) device / seed
# ==========================
def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================
# 3) 데이터 적재 & 프롬프트 빌드
# ==========================
def _iter_json_files(json_dir: Path):
    for p in sorted(json_dir.glob("*.json")):
        yield p

def _load_one_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_dataset(json_dir: Path) -> List[Dict[str, Any]]:
    items_raw: List[Dict[str, Any]] = []
    for fp in _iter_json_files(json_dir):
        obj = _load_one_json(fp)
        tgt = (obj.get("target_sentence") or "").strip()
        blocks = obj.get("logs") or []
        if not tgt or not blocks:
            continue
        block0 = blocks[0] or {}
        presses = block0.get("logs") or []
        items_raw.append({"tgt": tgt, "presses": presses, "file": str(fp.name)})
    if not items_raw:
        raise ValueError(f"유효한 샘플이 없습니다: {json_dir}")
    return items_raw

def load_norm_params(model_dir: Path):
    norm_path = model_dir / "normalization.json"
    if norm_path.exists():
        with open(norm_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return {
            "norm_source": obj.get("norm_source", "x_norm"),
            "x_min": float(obj.get("x_min", 0.0)),
            "x_max": float(obj.get("x_max", 1.0)),
            "y_min": float(obj.get("y_min", 0.0)),
            "y_max": float(obj.get("y_max", 1.0)),
        }
    return None

def infer_norm_from_data(items: List[Dict[str, Any]]):
    has_norm = True
    raw_x, raw_y = [], []
    for it in items:
        for e in it["presses"]:
            if ("x_norm" not in e) or ("y_norm" not in e):
                has_norm = False
            if "x" in e and "y" in e:
                try:
                    raw_x.append(float(e["x"]))
                    raw_y.append(float(e["y"]))
                except Exception:
                    pass
    if has_norm:
        return {"norm_source": "x_norm", "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    if not raw_x or not raw_y:
        raise ValueError("정규화 파라미터를 추정할 수 없습니다. (x_norm 없음, raw x/y 부족)")
    return {
        "norm_source": "raw",
        "x_min": float(min(raw_x)), "x_max": float(max(raw_x)),
        "y_min": float(min(raw_y)), "y_max": float(max(raw_y)),
    }

def make_ix_iy_getter(norm: Dict[str, float]):
    src = norm["norm_source"]
    x_min, x_max = norm["x_min"], norm["x_max"]
    y_min, y_max = norm["y_min"], norm["y_max"]
    def _clip01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))
    def _to01(v: float, vmin: float, vmax: float) -> float:
        if vmax <= vmin:
            return 0.5
        return _clip01((float(v) - vmin) / (vmax - vmin))
    def _q99(v01: float) -> int:
        return int(round(_clip01(v01) * 99))
    if src == "x_norm":
        def _get_ix_iy(e: Dict[str, Any]) -> Tuple[int, int]:
            x01 = float(e.get("x_norm", 0.5)); y01 = float(e.get("y_norm", 0.5))
            return _q99(x01), _q99(y01)
    else:
        def _get_ix_iy(e: Dict[str, Any]) -> Tuple[int, int]:
            x = float(e.get("x", 0.0)); y = float(e.get("y", 0.0))
            return _q99(_to01(x, x_min, x_max)), _q99(_to01(y, y_min, y_max))
    return _get_ix_iy

def build_src_from_presses(presses: List[Dict[str, Any]], get_ix_iy) -> str:
    toks: List[str] = []
    for e in presses:
        role = (e.get("role") or "").upper()
        label = e.get("label") or ""
        ok = False; ch = ""
        if role == "CHAR":
            ok = True; ch = label[:1] if label else ""
        elif role == "SPACE":
            ok = True; ch = SPACE_LABEL
        elif role == "MISS":
            ok = True; ch = MISS_LABEL
        elif role == "BKSP":
            ok = True; ch = BKSP_LABEL
        if not ok:
            continue
        ix, iy = get_ix_iy(e)
        toks.append(f"{ix:02d},{iy:02d}@{ch}")
    return PROMPT_FORMAT.format(seq=" ".join(toks))

# ==========================
# 5) 거리/지표
# ==========================
def levenshtein(seq_a, seq_b) -> int:
    la, lb = len(seq_a), len(seq_b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]; dp[0] = i
        ai = seq_a[i-1]
        for j in range(1, lb + 1):
            tmp = dp[j]
            cost = 0 if ai == seq_b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = tmp
    return dp[lb]

def char_cer(ref_text: str, hyp_text: str) -> float:
    return levenshtein(list(ref_text), list(hyp_text)) / max(1, len(ref_text))

def word_wer(ref_text: str, hyp_text: str) -> float:
    """단어(공백 기준) 레벨 편집거리 / ref 단어 수"""
    ref_tokens = [t for t in ref_text.split() if t != ""]
    hyp_tokens = [t for t in hyp_text.split() if t != ""]
    return levenshtein(ref_tokens, hyp_tokens) / max(1, len(ref_tokens))

def reverse_map_extra_ids(text: str, extra_id_map: Dict[str, str]) -> str:
    """
    extra_id를 원래 토큰으로 역매핑.
    예: "<extra_id_22><extra_id_13>" -> "ㄱㅏ"
    """
    result = text
    for extra_id, token in extra_id_map.items():
        result = result.replace(extra_id, token)
    return result

# ==========================
# 6) 생성 + 재조합 + 평가
# ==========================
@torch.no_grad()
def generate_topk_until_eos(model, tokenizer, device, inputs: List[str], topk: int, beams: int, batch_size: int):
    beams = max(1, beams, topk)
    all_top1, all_topk = [], []

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    stop = StoppingCriteriaList([EosBatchStop(eos_id)]) if eos_id is not None else None

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        enc = tokenizer(batch, max_length=MAX_SRC_LEN, truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model.generate(
            **enc,
            num_beams=beams,
            num_return_sequences=topk,
            do_sample=False,
            early_stopping=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            stopping_criteria=stop,
            max_new_tokens=160,
            return_dict_in_generate=True,
        )

        seqs = out.sequences
        decoded = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        for b in range(len(batch)):
            # extra_id를 원래 자모로 역매핑한 후 정규화
            # 예: '<extra_id_22><extra_id_13>' -> 'ㄱㅏ'
            raw = [ud.normalize("NFC", reverse_map_extra_ids(t, EXTRA_ID_TO_TOKEN)) for t in decoded[b*topk:(b+1)*topk]]
            all_topk.append(raw)
            all_top1.append(raw[0] if raw else "")

    return all_top1, all_topk

# ==========================
# 7) main
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", default=JSON_DIR_DEFAULT, type=str)
    ap.add_argument("--model_dir", default=MODEL_DIR_DEFAULT, type=str)
    ap.add_argument("--n", type=int, default=SAMPLE_SIZE_DEFAULT)
    ap.add_argument("--show", type=int, default=SHOW_COUNT_DEFAULT)
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--batch", type=int, default=BATCH_INFER_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = ap.parse_args()

    set_seed(args.seed)
    json_dir = Path(args.json_dir)
    model_dir = Path(args.model_dir)
    device = get_device()

    # 1) 데이터 로드
    raw_items = load_dataset(json_dir)

    # 2) 정규화 파라미터
    norm = load_norm_params(model_dir) or infer_norm_from_data(raw_items)
    get_ix_iy = make_ix_iy_getter(norm)

    # 3) 프롬프트/GT 준비
    data = []
    for it in raw_items:
        src = build_src_from_presses(it["presses"], get_ix_iy)
        gt = (it["tgt"] or "").strip()
        if src.strip() and gt:
            data.append({"src": src, "gt": gt, "file": it["file"]})
    if not data:
        raise ValueError("빈 시퀀스입니다. (MISS/BKSP 포함 여부 확인)")

    if args.n < len(data):
        data = random.sample(data, args.n)

    # 4) 모델/토크나이저
    base = "KETI-AIR/ke-t5-small"
    tok = AutoTokenizer.from_pretrained(model_dir if model_dir.exists() else base)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir if model_dir.exists() else base)
    model.to(device).eval()

    # 5) 생성
    start = time.time()
    inputs = [d["src"] for d in data]
    preds_top1_raw, preds_topk_raw = generate_topk_until_eos(
        model, tok, device, inputs, topk=args.topk, beams=max(args.beams, args.topk), batch_size=args.batch
    )

    # 6) 재조합 + 지표
    em_cnt, emk_cnt = 0, 0
    cer_list, wer_list = [], []
    per_details = []
    for d, p1_raw, pk_raw in zip(data, preds_top1_raw, preds_topk_raw):
        gt_text = ud.normalize("NFC", d["gt"])

        # 자모 스트림 → 음절 재조합
        pred_text = automata(p1_raw)
        cand_texts = [automata(s) for s in pk_raw]

        em  = (pred_text == gt_text)
        emk = (gt_text in cand_texts)
        cer = char_cer(gt_text, pred_text)
        wer = word_wer(gt_text, pred_text)

        em_cnt += int(em)
        emk_cnt += int(emk)
        cer_list.append(cer)
        wer_list.append(wer)

        # 필요시 디버깅용 상세
        per_details.append({
            "file": d["file"],
            "gt": gt_text,
            "pred_text": pred_text,
            "cands_text": cand_texts,
            "em": em,
            "emk": emk,
            "cer": cer,
            "wer": wer,
        })

    n = len(data)
    em_acc   = em_cnt / max(1, n)
    emk_acc  = emk_cnt / max(1, n)
    avg_cer  = 1-(sum(cer_list) / max(1, len(cer_list)))
    avg_wer  = 1-(sum(wer_list) / max(1, len(wer_list)))

    # 7) 결과 출력
    print("==== Evaluation (자모→음절 재조합 후 문장 비교) ====")
    print(f"- Samples evaluated : {n}")
    print(f"- EM@1 (exact text) : {em_acc*100:.2f}%")
    print(f"- EM@{args.topk}     : {emk_acc*100:.2f}%")
    print(f"- Avg CER (char)     : {avg_cer:.4f}")
    print(f"- Avg WER (word)     : {avg_wer:.4f}")
    print()

    print(f"==== Examples (show {min(args.show,n)}) — GT / Pred(recomposed) / Top-{args.topk} recomposed ====")
    show_items = random.sample(per_details, min(args.show, n))
    for i, it in enumerate(show_items, 1):
        print(f"[{i}] file={it['file']}")
        print(f"    GT  : {it['gt']}")
        print(f"    Pred: {it['pred_text']}")
        print(f"    Top-{len(it['cands_text'])} candidates (recomposed):")
        for j, c in enumerate(it['cands_text'], 1):
            mark = " ← GT" if c == it["gt"] else ""
            print(f"     {j:>2}. {c}{mark}")
        print(f"    EM={it['em']} | EM@K={it['emk']} | CER={it['cer']:.4f} | WER={it['wer']:.4f}\n")

    print(f"Elapsed: {time.time()-start:.2f}s")

if __name__ == "__main__":
    main()
