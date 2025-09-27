# tmp_eos_beam.py — EOS까지 생성 + Beam Search 후보 출력 통합판
# 산출: EM, EM@K, 평균 WER/CER + 샘플별 Top-K 후보
import argparse, json, random, unicodedata as ud, time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

# ==========================
# 0) 기본 상수
# ==========================
JSON_DIR_DEFAULT  = "saved_logs"
MODEL_DIR_DEFAULT = "ckpt/ke-t5-small-RnE2025"
MAX_SRC_LEN = 512
MAX_TGT_LEN = 256   # (디코드 상한은 사용하지 않지만 유지)
SAMPLE_SIZE_DEFAULT = 200
SHOW_COUNT_DEFAULT  = 5
TOPK_DEFAULT        = 5
BATCH_INFER_DEFAULT = 32
SEED_DEFAULT        = 425

SPACE_LABEL = "[SPACE]"
MISS_LABEL  = "[MISS]"
BKSP_LABEL  = "[BKSP]"
PROMPT_FORMAT = "touchseq: {seq} -> text"

# ==========================
# EOS 정지 기준: 배치 내 모든 시퀀스가 EOS를 한 번 이상 생성하면 중단
# ==========================
class EosBatchStop(StoppingCriteria):
    def __init__(self, eos_id: int):
        self.eos_id = eos_id
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.eos_id is None:
            return False
        hit = (input_ids == self.eos_id).any(dim=1)
        return bool(hit.all().item())

# ==========================
# 1) 디바이스/시드
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
# 2) JSON 파싱/프롬프트 구성(두 파일과 동일한 규칙)
# ==========================
def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def _to01(v: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.5
    return _clip01((float(v) - vmin) / (vmax - vmin))

def _q99(v01: float) -> int:
    return int(round(_clip01(v01) * 99))

def _canon_char_from_log(role: str, label: str) -> Tuple[bool, str]:
    role = (role or "").upper()
    if role == "CHAR":
        ch = (label or "")
        return True, (ch[:1] if ch else "")
    if role == "SPACE":
        return True, SPACE_LABEL
    if role == "MISS":
        return True, MISS_LABEL
    if role == "BKSP":
        return True, BKSP_LABEL
    return False, ""

def _iter_json_files(json_dir: Path):
    for p in sorted(json_dir.glob("*.json")):
        yield p

def _load_one_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==========================
# 3) 정규화 파라미터 로드/추정
# ==========================
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
        ok, ch = _canon_char_from_log(e.get("role"), e.get("label"))
        if not ok:
            continue
        ix, iy = get_ix_iy(e)
        toks.append(f"{ix:02d},{iy:02d}@{ch}")
    return PROMPT_FORMAT.format(seq=" ".join(toks))

def load_dataset(json_dir: Path) -> List[Dict[str, str]]:
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

# ==========================
# 4) 텍스트 정규화 & 거리
# ==========================
def norm_text(s: str) -> str:
    return ud.normalize("NFC", (s or "").strip())

def levenshtein(seq_a, seq_b) -> int:
    la, lb = len(seq_a), len(seq_b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, lb + 1):
            tmp = dp[j]
            cost = 0 if seq_a[i-1] == seq_b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = tmp
    return dp[lb]

def wer(ref: str, hyp: str) -> float:
    ref = norm_text(ref); hyp = norm_text(hyp)
    rt, ht = ref.split(), hyp.split()
    return levenshtein(rt, ht) / max(1, len(rt))

def cer(ref: str, hyp: str) -> float:
    ref = norm_text(ref); hyp = norm_text(hyp)
    return levenshtein(list(ref), list(hyp)) / max(1, len(ref))

# ==========================
# 5) EOS까지 생성 + Top-K 후보
# ==========================
@torch.no_grad()
def generate_topk_until_eos(model, tokenizer, device, inputs: List[str], topk: int, beams: int, batch_size: int):
    """
    - num_beams ≥ topk
    - EOS가 모든 시퀀스에서 한 번 이상 생성되면 조기 중단
    - 안전 상한: max_new_tokens=1024
    반환: (top1 리스트, topk 후보 리스트)
    """
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
            early_stopping=False,        # 모든 beam이 EOS 도달 시 정지
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            stopping_criteria=stop,     # 배치 차원의 EOS 체크(추가 안전장치)
            max_new_tokens=32,        # 무한생성 방지 상한
            return_dict_in_generate=True,
        )

        seqs = out.sequences
        decoded = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        for b in range(len(batch)):
            cand = [norm_text(t) for t in decoded[b*topk:(b+1)*topk]]
            all_topk.append(cand)
            all_top1.append(cand[0] if cand else "")

    return all_top1, all_topk

# ==========================
# 6) 메인
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", default=JSON_DIR_DEFAULT, type=str)
    ap.add_argument("--model_dir", default=MODEL_DIR_DEFAULT, type=str)
    ap.add_argument("--n", type=int, default=SAMPLE_SIZE_DEFAULT)
    ap.add_argument("--show", type=int, default=SHOW_COUNT_DEFAULT)
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    ap.add_argument("--beams", type=int, default=5, help="num_beams (>= topk 권장)")
    ap.add_argument("--batch", type=int, default=BATCH_INFER_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = ap.parse_args()

    set_seed(args.seed)
    json_dir = Path(args.json_dir)
    model_dir = Path(args.model_dir)
    device = get_device()

    # (1) 데이터 로드
    raw_items = load_dataset(json_dir)

    # (2) 정규화 파라미터
    norm = load_norm_params(model_dir) or infer_norm_from_data(raw_items)
    get_ix_iy = make_ix_iy_getter(norm)

    # (3) 프롬프트 구성
    data = []
    for it in raw_items:
        src = build_src_from_presses(it["presses"], get_ix_iy)
        tgt = norm_text(it["tgt"])
        if src.strip() and tgt:
            data.append({"src": src, "tgt": tgt, "file": it["file"]})
    if not data:
        raise ValueError("빈 시퀀스입니다. 입력 이벤트(MISS/BKSP 포함) 여부를 확인하세요.")

    # (4) 샘플링
    if args.n < len(data):
        data = random.sample(data, args.n)

    # (5) 모델/토크나이저
    base = "KETI-AIR/ke-t5-small"
    tok = AutoTokenizer.from_pretrained(model_dir if model_dir.exists() else base)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir if model_dir.exists() else base)
    model.to(device).eval()

    # (6) 생성
    start = time.time()
    inputs = [d["src"] for d in data]
    preds_top1, preds_topk = generate_topk_until_eos(
        model, tok, device, inputs, topk=args.topk, beams=max(args.beams, args.topk), batch_size=args.batch
    )

    # (7) 지표
    em_cnt, emk_cnt, wers, cers = 0, 0, [], []
    per_details = []
    for d, p1, pk in zip(data, preds_top1, preds_topk):
        gt = d["tgt"]
        em  = (p1 == gt)
        emk = (gt in pk)
        w = wer(gt, p1)
        c = cer(gt, p1)
        em_cnt += int(em); emk_cnt += int(emk)
        wers.append(w); cers.append(c)
        per_details.append({"file": d["file"], "em": em, "emk": emk, "wer": w, "cer": c, "gt": gt, "pred": p1, "cands": pk})

    n = len(data)
    em_acc  = em_cnt / max(1, n)
    emk_acc = emk_cnt / max(1, n)
    avg_wer = sum(wers) / max(1, len(wers))
    avg_cer = sum(cers) / max(1, len(cers))

    # (8) 요약/표시
    print("==== Evaluation (touchseq ➜ text) ====")
    print(f"- Samples evaluated : {n}")
    print(f"- EM@1 accuracy     : {em_acc*100:.2f}%")
    print(f"- EM@{args.topk} accuracy : {emk_acc*100:.2f}%")
    print(f"- Avg WER           : {avg_wer:.4f}")
    print(f"- Avg CER           : {avg_cer:.4f}\n")

    print(f"==== Per-sample details (show {min(args.show, n)}) ====")
    show_items = random.sample(per_details, min(args.show, n))
    for i, it in enumerate(show_items, 1):
        print(f"[{i}] file={it['file']}")
        print(f"    EM={it['em']} | EM@K={it['emk']} | WER={it['wer']:.4f} | CER={it['cer']:.4f}")
        print(f"    GT  : {it['gt']}")
        print(f"    Pred: {it['pred']}")
        print(f"    Top-{len(it['cands'])} candidates:")
        for j, c in enumerate(it["cands"], 1):
            mark = " ← GT" if c == it["gt"] else ""
            print(f"     {j:>2}. {c}{mark}")
        print()
    print(f"Total elapsed time: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()


