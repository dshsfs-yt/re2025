# model_validation.py — touchseq(JSON per sentence) ➜ 평가(KE-T5)
# 목적:
#  - 모델 출력(자모 시퀀스; 예: '왉'을 ㅇ ㅗ ㅏ ㄹ ㄱ처럼 기본 자모로 타이핑한 스트림)을
#    한글 음절로 자동 재조합한 뒤 GT 문장과 비교
#  - 지표: EM@1, EM@K, 평균 CER(문자), 평균 WER(단어)
#
# 사용 예:
#   python model_validation.py --json_dir saved_logs_nokk \
#       --model_dir ckpt/ke-t5-small-RnE2025_jamosplit --n 200 --topk 5 --beams 8

# python model_validation.py --split all --n -1 --save_wrong val_errors.json

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
PROMPT_FORMAT = "touchseq: {seq} coordinate: {xy} -> text"

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
    cors: List[str] = []
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
        toks.append(ch)
        cors.append(f"{ix:02d},{iy:02d}")
    return PROMPT_FORMAT.format(seq=" ".join(toks), xy=" ".join(cors))

# ==========================
# 4) Token mapping functions (from training code)
# ==========================
def map_tokens_to_extra_ids(text: str, token_map: Dict[str, str]) -> str:
    """
    텍스트 내의 커스텀 토큰을 extra_id로 매핑.
    예: "ㄱㅏ[SPACE]ㄴㅏ" -> "<extra_id_22><extra_id_13><extra_id_33><extra_id_23><extra_id_13>"
    """
    result = []
    i = 0
    while i < len(text):
        # 긴 토큰부터 매칭 시도 (예: [SPACE], [BKSP] 등)
        matched = False
        # 최대 7글자 토큰 ([FLUSH] 등)
        for token_len in range(min(7, len(text) - i), 0, -1):
            substr = text[i:i+token_len]
            if substr in token_map:
                result.append(token_map[substr])
                i += token_len
                matched = True
                break

        if not matched:
            # 매핑되지 않은 문자는 그대로 유지 (숫자, 영문, 특수문자 등)
            result.append(text[i])
            i += 1

    return "".join(result)

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

# ==========================
# 6) 생성 + 재조합 + 평가
# ==========================
def remove_special_tokens_from_text(text: str, tokenizer) -> str:
    """
    텍스트에서 특수 토큰들을 제거합니다.
    pad_token, eos_token, unk_token 등을 제거
    """
    # 토크나이저의 특수 토큰들
    tokens_to_remove = []
    if tokenizer.pad_token:
        tokens_to_remove.append(tokenizer.pad_token)
    if tokenizer.eos_token:
        tokens_to_remove.append(tokenizer.eos_token)
    if tokenizer.unk_token:
        tokens_to_remove.append(tokenizer.unk_token)
    if tokenizer.bos_token:
        tokens_to_remove.append(tokenizer.bos_token)
    
    # 텍스트에서 특수 토큰 제거
    result = text
    for token in tokens_to_remove:
        result = result.replace(token, "")
    
    # 연속된 공백을 하나로 축약하고 양끝 공백 제거
    result = " ".join(result.split())
    
    return result.strip()

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
        decoded = tokenizer.batch_decode(seqs, skip_special_tokens=False)
        for b in range(len(batch)):
            batch_decoded = decoded[b*topk:(b+1)*topk]
            
            # extra_id를 원래 자모로 역매핑한 후 정규화
            # 예: '<extra_id_22><extra_id_13>' -> 'ㄱㅏ'
            # 추가로 특수 토큰이 남아있을 경우를 대비한 제거 처리
            raw = []
            for text in batch_decoded:
                # extra_id를 자모로 역매핑
                text_mapped = reverse_map_extra_ids(text, EXTRA_ID_TO_TOKEN)
                # 남은 특수 토큰 제거 (안전장치)
                text_clean = remove_special_tokens_from_text(text_mapped, tokenizer)
                # 유니코드 정규화
                text_normalized = ud.normalize("NFC", text_clean)
                raw.append(text_normalized)
            
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
    ap.add_argument("--n", type=int, default=SAMPLE_SIZE_DEFAULT,
                    help="Number of samples to evaluate. Use -1 for all samples (default: 200)")
    ap.add_argument("--show", type=int, default=SHOW_COUNT_DEFAULT)
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--batch", type=int, default=BATCH_INFER_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "validation"],
                    help="Which data split to evaluate: 'all', 'train', or 'val'/'validation'")
    ap.add_argument("--split_ratio", type=float, default=0.95,
                    help="Train/validation split ratio (default: 0.95 for train)")
    ap.add_argument("--save_wrong", type=str, default=None,
                    help="Save wrong predictions to JSON file (e.g., wrong_preds.json)")
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

    # 3) 프롬프트/GT 준비 (training 코드와 동일한 방식)
    data = []
    for it in raw_items:
        src = build_src_from_presses(it["presses"], get_ix_iy)
        gt = (it["tgt"] or "").strip()
        if src.strip() and gt:
            data.append({"src": src, "gt": gt, "file": it["file"]})
    
    if not data:
        raise ValueError("빈 시퀀스입니다. (MISS/BKSP 포함 여부 확인)")
    
    # 4) Training 코드와 동일한 방식으로 셔플 (src/tgt 쌍을 만든 후 셔플)
    random.Random(SEED_DEFAULT).shuffle(data)
    
    # 5) Train/Validation 분리 (training 코드와 동일)
    total_count = len(data)
    split_point = int(total_count * args.split_ratio) if total_count > 20 else max(1, total_count - 1)
    
    # 분리
    if args.split in ["val", "validation"]:
        data = data[split_point:]  # validation set
        split_name = "VALIDATION"
    elif args.split == "train":
        data = data[:split_point]  # train set
        split_name = "TRAIN"
    else:  # "all"
        split_name = "ALL"
        # data는 그대로 전체 사용
    
    print(f"\n[Data Split] Evaluating on {split_name} set")
    print(f"  Total samples: {total_count}")
    print(f"  Train samples: {split_point} ({args.split_ratio*100:.1f}%)")
    print(f"  Val samples: {total_count - split_point} ({(1-args.split_ratio)*100:.1f}%)")
    print(f"  → Selected {len(data)} samples for {split_name} evaluation")
    
    # 6) 샘플링 (--n 옵션이 지정된 경우)
    original_len = len(data)
    if args.n == -1:
        print(f"  → Using ALL {len(data)} samples from {split_name} set")
    elif args.n < len(data):
        data = random.sample(data, args.n)
        print(f"  → Randomly sampled {args.n} from {original_len} samples in {split_name} set")
    else:
        print(f"  → Using all {len(data)} samples (less than requested {args.n})")

    # 7) 모델/토크나이저
    base = "KETI-AIR/ke-t5-small"
    tok = AutoTokenizer.from_pretrained(model_dir if model_dir.exists() else base)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device).eval()

    # 8) 입력 텍스트를 extra_id로 매핑 (CRITICAL: 이 부분이 누락되어 있었음)
    print(f"\n==== Mapping inputs to extra_ids ====")
    print(f"Original input example: {data[0]['src'][:100]}...")
    
    # 모든 입력에 TOKEN_TO_EXTRA_ID 매핑 적용
    mapped_inputs = []
    for d in data:
        mapped_src = map_tokens_to_extra_ids(d["src"], TOKEN_TO_EXTRA_ID)
        mapped_inputs.append(mapped_src)
    
    print(f"Mapped input example: {mapped_inputs[0][:100]}...")
    print(f"Total {len(TOKEN_TO_EXTRA_ID)} tokens mapped to extra_ids")
    print(f"Special tokens will be removed: pad={tok.pad_token}, eos={tok.eos_token}")

    # 9) 생성 (이제 매핑된 입력 사용)
    start = time.time()
    preds_top1_raw, preds_topk_raw = generate_topk_until_eos(
        model, tok, device, mapped_inputs, topk=args.topk, beams=max(args.beams, args.topk), batch_size=args.batch
    )

    # 10) 재조합 + 지표
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
            # 디버깅용: 원본 자모 시퀀스도 저장
            "pred_raw": p1_raw[:100] + "..." if len(p1_raw) > 100 else p1_raw,
        })

    n = len(data)
    em_acc   = em_cnt / max(1, n)
    emk_acc  = emk_cnt / max(1, n)
    avg_cer  = 1-(sum(cer_list) / max(1, len(cer_list)))
    avg_wer  = 1-(sum(wer_list) / max(1, len(wer_list)))

    # 11) 결과 출력
    print(f"\n==== Evaluation Results ({split_name} Set) ====")
    print(f"- Dataset split  : {split_name}")
    print(f"- Samples evaluated : {n}")
    print(f"- EM@1 (exact text) : {em_acc*100:.2f}%")
    print(f"- EM@{args.topk}     : {emk_acc*100:.2f}%")
    print(f"- Avg CER (char)    : {avg_cer:.4f}")
    print(f"- Avg WER (word)    : {avg_wer:.4f}")
    print()

    print(f"==== Examples from {split_name} set (show {min(args.show,n)}) ====")
    show_items = random.sample(per_details, min(args.show, n))
    for i, it in enumerate(show_items, 1):
        print(f"[{i}] file={it['file']}")
        print(f"    GT       : {it['gt']}")
        print(f"    Pred     : {it['pred_text']}")
        print(f"    Raw jamo : {it['pred_raw']}")  # 디버깅용: 원본 자모 출력
        print(f"    Top-{len(it['cands_text'])} candidates (recomposed):")
        for j, c in enumerate(it['cands_text'], 1):
            mark = " ← GT" if c == it["gt"] else ""
            print(f"     {j:>2}. {c}{mark}")
        print(f"    EM={it['em']} | EM@K={it['emk']} | CER={it['cer']:.4f} | WER={it['wer']:.4f}\n")

    print(f"[{split_name} Set] Total elapsed: {time.time()-start:.2f}s")
    
    # 12) Exact Match 틀린 샘플들 모두 출력
    wrong_predictions = [item for item in per_details if not item["em"]]
    if wrong_predictions:
        print(f"\n==== All Wrong Predictions ({len(wrong_predictions)} samples) ====")
        print(f"Showing all {len(wrong_predictions)} samples where Exact Match failed:\n")
        
        for i, it in enumerate(wrong_predictions, 1):
            print(f"[Wrong {i}/{len(wrong_predictions)}] file={it['file']}")
            print(f"    GT       : {it['gt']}")
            print(f"    Pred     : {it['pred_text']}")
            print(f"    CER={it['cer']:.4f} | WER={it['wer']:.4f} | EM@K={it['emk']}")
            
            # Top-K에 정답이 있는지 표시
            if it['emk']:
                for j, c in enumerate(it['cands_text'], 1):
                    if c == it["gt"]:
                        print(f"    → GT found at rank {j} in top-{len(it['cands_text'])}")
                        break
            print()  # 간격 추가
        
        # 틀린 샘플 통계
        print(f"==== Wrong Predictions Summary ====")
        print(f"- Total wrong: {len(wrong_predictions)}/{len(per_details)} ({len(wrong_predictions)/len(per_details)*100:.1f}%)")
        
        # EM@K에는 포함되는 샘플 수
        wrong_but_in_topk = sum(1 for it in wrong_predictions if it["emk"])
        print(f"- Wrong but in top-{args.topk}: {wrong_but_in_topk}/{len(wrong_predictions)} ({wrong_but_in_topk/max(1,len(wrong_predictions))*100:.1f}%)")
        
        # 평균 CER/WER (틀린 것들만)
        if wrong_predictions:
            avg_wrong_cer = sum(it["cer"] for it in wrong_predictions) / len(wrong_predictions)
            avg_wrong_wer = sum(it["wer"] for it in wrong_predictions) / len(wrong_predictions)
            print(f"- Avg CER (wrong only): {1-avg_wrong_cer:.4f}")
            print(f"- Avg WER (wrong only): {1-avg_wrong_wer:.4f}")
        
        # 틀린 예측들을 파일로 저장 (옵션)
        if args.save_wrong:
            wrong_data = []
            for it in wrong_predictions:
                wrong_data.append({
                    "file": it["file"],
                    "gt": it["gt"],
                    "pred": it["pred_text"],
                    "pred_raw_jamo": it["pred_raw"],
                    "top_k_candidates": it["cands_text"],
                    "em": it["em"],
                    "em_at_k": it["emk"],
                    "cer": float(it["cer"]),
                    "wer": float(it["wer"]),
                })
            
            save_path = Path(args.save_wrong)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "split": split_name,
                    "total_samples": len(per_details),
                    "wrong_count": len(wrong_predictions),
                    "accuracy": float(em_acc),
                    "predictions": wrong_data
                }, f, ensure_ascii=False, indent=2)
            print(f"\n→ Wrong predictions saved to: {save_path.resolve()}")
    else:
        print(f"\n==== Perfect Results! ====")
        print(f"All {len(per_details)} predictions were exactly correct (100% EM@1)")

if __name__ == "__main__":
    main()