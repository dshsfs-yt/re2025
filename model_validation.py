# model_validation.py — tap-logs(JSON per sentence) ➜ 평가(KE-T5)
# 산출: EM 정확도, EM@K 정확도, 평균 WER/CER + 샘플별 상세 지표
import argparse, json, random, unicodedata as ud
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ==========================
# 0) 기본 상수(필요시 여기만 바꾸면 됨)
# ==========================
JSON_DIR_DEFAULT  = "saved_logs"               # 문장 단위 JSON 로그 폴더
MODEL_DIR_DEFAULT = "ckpt/ke-t5-small-RnE2025" # 학습이 저장한 체크포인트 디렉터리
MAX_SRC_LEN = 512
MAX_TGT_LEN = 256

SAMPLE_SIZE_DEFAULT = 200   # 평가에 사용할 샘플 개수(상수)
SHOW_COUNT_DEFAULT  = 5    # 상세 출력할 샘플 개수(상수)
TOPK_DEFAULT        = 5     # EM@K의 K(상수)
BATCH_INFER_DEFAULT = 32    # 추론 배치 크기
SEED_DEFAULT        = 4213    # 랜덤 시드

# 학습 시 사용한 라벨/프롬프트 포맷(학습 스크립트와 동일해야 함)
SPACE_LABEL = "[SPACE]"
MISS_LABEL  = "[MISS]"
BKSP_LABEL  = "[BKSP]"
PROMPT_FORMAT = "touchseq: {seq} -> text"

# ==========================
# 1) 디바이스/시드 유틸
# ==========================
def get_device():
    """CUDA/MPS가 있으면 사용, 없으면 CPU."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int):
    """파이썬/토치 시드 고정."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================
# 2) JSON 파싱/프롬프트 구성
# ==========================
def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def _to01(v: float, vmin: float, vmax: float) -> float:
    """raw 좌표를 0~1로 정규화."""
    if vmax <= vmin:
        return 0.5
    return _clip01((float(v) - vmin) / (vmax - vmin))

def _q99(v01: float) -> int:
    """0~1 값을 00~99 버킷으로 양자화."""
    return int(round(_clip01(v01) * 99))

def _canon_char_from_log(role: str, label: str) -> Tuple[bool, str]:
    """
    로그 이벤트를 학습 입력에 포함할지/무엇으로 넣을지 결정.
    - CHAR : label의 첫 글자 사용
    - SPACE: [SPACE]
    - MISS : [MISS]  (좌표 있음, 글자 미입력)
    - BKSP : [BKSP]
    기타/누락은 제외
    """
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
    """폴더의 *.json 파일을 정렬된 순서로 순회."""
    for p in sorted(json_dir.glob("*.json")):
        yield p

def _load_one_json(path: Path) -> Dict[str, Any]:
    """한 개 JSON 파일 로드."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==========================
# 3) 정규화 파라미터 로드/추정
# ==========================
def load_norm_params(model_dir: Path):
    """
    학습 시 저장된 normalization.json 읽기.
    존재하지 않으면 None 반환(→ 데이터 기반 추정).
    """
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
    """
    normalization.json이 없을 때 데이터에서 정규화 설정을 추정.
    - x_norm/y_norm 존재하면 그대로 사용
    - 없으면 raw x,y의 최소/최대를 이용해 정규화 범위 추정
    """
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
        "x_min": float(min(raw_x)),
        "x_max": float(max(raw_x)),
        "y_min": float(min(raw_y)),
        "y_max": float(max(raw_y)),
    }

def make_ix_iy_getter(norm: Dict[str, float]):
    """이벤트 → (ix, iy) 버킷 변환 함수를 생성."""
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
    """
    터치 이벤트 시퀀스를 학습 프롬프트 문자열로 변환.
    각 이벤트 → "문자@ix,iy" 조합을 공백으로 이어붙인 후
    PROMPT_FORMAT에 삽입.
    """
    toks: List[str] = []
    for e in presses:
        ok, ch = _canon_char_from_log(e.get("role"), e.get("label"))
        if not ok:
            continue
        ix, iy = get_ix_iy(e)
        toks.append(f"{ch}@{ix:02d},{iy:02d}")
    return PROMPT_FORMAT.format(seq=" ".join(toks))

def load_dataset(json_dir: Path) -> List[Dict[str, str]]:
    """
    saved_logs/ 폴더에서 학습과 동일한 구조로 데이터 적재.
    반환: {'src': str, 'tgt': str, 'file': str} 리스트(원본 파일명 포함)
    """
    items_raw: List[Dict[str, Any]] = []
    for fp in _iter_json_files(json_dir):
        obj = _load_one_json(fp)
        tgt = (obj.get("target_sentence") or "").strip()
        blocks = obj.get("logs") or []
        if not tgt or not blocks:
            continue
        # 문장별 JSON에서 첫 세션의 이벤트만 사용(학습 스크립트와 동일)
        block0 = blocks[0] or {}
        presses = block0.get("logs") or []
        items_raw.append({"tgt": tgt, "presses": presses, "file": str(fp.name)})
    if not items_raw:
        raise ValueError(f"유효한 샘플이 없습니다: {json_dir}")
    return items_raw

# ==========================
# 4) 텍스트 정규화 & 거리(Levenshtein)
# ==========================
def norm_text(s: str) -> str:
    """NFC 정규화 + 좌우 공백 제거."""
    return ud.normalize("NFC", (s or "").strip())

def levenshtein(seq_a, seq_b) -> int:
    """삽입/삭제/치환 비용 1의 Levenshtein 거리."""
    la, lb = len(seq_a), len(seq_b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            tmp = dp[j]
            cost = 0 if seq_a[i-1] == seq_b[j-1] else 1
            dp[j] = min(
                dp[j] + 1,        # 삭제
                dp[j-1] + 1,      # 삽입
                prev + cost       # 치환
            )
            prev = tmp
    return dp[lb]

def wer(ref: str, hyp: str) -> float:
    """단어 단위 WER(평균 길이 0 방지)."""
    ref = norm_text(ref); hyp = norm_text(hyp)
    ref_toks = ref.split()
    hyp_toks = hyp.split()
    denom = max(1, len(ref_toks))
    return levenshtein(ref_toks, hyp_toks) / denom

def cer(ref: str, hyp: str) -> float:
    """문자 단위 CER(평균 길이 0 방지)."""
    ref = norm_text(ref); hyp = norm_text(hyp)
    denom = max(1, len(ref))
    return levenshtein(list(ref), list(hyp)) / denom

# ==========================
# 5) 배치 생성(Top-K 후보 포함)
# ==========================
@torch.no_grad()
def generate_topk(model, tokenizer, device, inputs: List[str], topk: int, beams: int, batch_size: int):
    """
    num_beams ≥ topk로 설정하여 상위 K개 후보를 생성.
    반환: (top1 리스트, topk 후보 리스트)
    """
    beams = max(beams, topk, 1)
    all_top1, all_topk = [], []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        enc = tokenizer(
            batch, max_length=MAX_SRC_LEN, truncation=True, padding=True, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_length=MAX_TGT_LEN,
            num_beams=beams,
            num_return_sequences=topk,
            early_stopping=True,
            do_sample=False,
            return_dict_in_generate=True,
        )
        seqs = out.sequences  # (배치 크기 × topk, 길이)
        decoded = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        # 배치별로 topk 묶기
        for b in range(len(batch)):
            cand = [norm_text(t) for t in decoded[b*topk:(b+1)*topk]]
            all_topk.append(cand)
            all_top1.append(cand[0] if cand else "")
    return all_top1, all_topk

# ==========================
# 6) 메인: 로드 → 샘플링 → 생성 → 지표
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", default=JSON_DIR_DEFAULT, type=str, help="문장별 tap-logs JSON 폴더")
    ap.add_argument("--model_dir", default=MODEL_DIR_DEFAULT, type=str, help="학습 체크포인트 폴더")
    ap.add_argument("--n", type=int, default=SAMPLE_SIZE_DEFAULT, help="평가에 사용할 샘플 개수")
    ap.add_argument("--show", type=int, default=SHOW_COUNT_DEFAULT, help="상세 출력할 샘플 개수")
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT, help="EM@K의 K 값")
    ap.add_argument("--batch", type=int, default=BATCH_INFER_DEFAULT, help="추론 배치 크기")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT, help="랜덤 시드")
    args = ap.parse_args()

    set_seed(args.seed)
    json_dir = Path(args.json_dir)
    model_dir = Path(args.model_dir)
    device = get_device()

    # (1) 데이터 로드(문장 단위 JSON → (presses, tgt))
    raw_items = load_dataset(json_dir)

    # (2) 정규화 파라미터 로드(없으면 데이터에서 추정) → (ix,iy) 변환기 생성
    norm = load_norm_params(model_dir) or infer_norm_from_data(raw_items)
    get_ix_iy = make_ix_iy_getter(norm)

    # (3) 학습과 동일한 프롬프트 구성
    data = []
    for it in raw_items:
        src = build_src_from_presses(it["presses"], get_ix_iy)
        tgt = norm_text(it["tgt"])
        if src.strip() and tgt:
            data.append({"src": src, "tgt": tgt, "file": it["file"]})
    if not data:
        raise ValueError("빈 시퀀스입니다. 입력 이벤트(MISS/BKSP 포함) 여부를 확인하세요.")

    # (4) 평가용 샘플 무작위 추출
    if args.n < len(data):
        data = random.sample(data, args.n)

    # (5) 모델/토크나이저 로드(체크포인트 없으면 베이스 모델로)
    tokenizer = AutoTokenizer.from_pretrained(model_dir if model_dir.exists() else "KETI-AIR/ke-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir if model_dir.exists() else "KETI-AIR/ke-t5-small")
    model.to(device).eval()

    # (6) 생성(top-1, top-k 후보)
    inputs = [d["src"] for d in data]
    preds_top1, preds_topk = generate_topk(
        model, tokenizer, device, inputs, topk=args.topk, beams=max(5, args.topk), batch_size=args.batch
    )

    # (7) 지표 집계: EM, EM@K, WER/CER 평균
    em_cnt, emk_cnt = 0, 0
    wers, cers = [], []
    per_details = []
    for d, p1, pk in zip(data, preds_top1, preds_topk):
        gt = d["tgt"]
        em  = (p1 == gt)
        emk = (gt in pk)
        w = wer(gt, p1)
        c = cer(gt, p1)
        em_cnt += int(em)
        emk_cnt += int(emk)
        wers.append(w)
        cers.append(c)
        per_details.append({
            "file": d["file"],
            "em": em, "emk": emk,
            "wer": w, "cer": c,
            "gt": gt, "pred": p1
        })

    n = len(data)
    em_acc  = em_cnt / max(1, n)
    emk_acc = emk_cnt / max(1, n)
    avg_wer = sum(wers) / max(1, len(wers))
    avg_cer = sum(cers) / max(1, len(cers))

    # (8) 요약 출력
    print("==== Evaluation (touchseq ➜ text) ====")
    print(f"- Samples evaluated : {n}")
    print(f"- EM@1 accuracy     : {em_acc*100:.2f}%")
    print(f"- EM@{args.topk} accuracy : {emk_acc*100:.2f}%")
    print(f"- Avg WER           : {avg_wer:.4f}")
    print(f"- Avg CER           : {avg_cer:.4f}")
    print("")
    # (9) 샘플별 상세 출력(무작위로 args.show개)
    print(f"==== Per-sample details (show {min(args.show, n)}) ====")
    show_items = random.sample(per_details, min(args.show, n))
    for i, it in enumerate(show_items, 1):
        print(f"[{i}] file={it['file']}")
        print(f"    EM={it['em']} | EM@K={it['emk']} | WER={it['wer']:.4f} | CER={it['cer']:.4f}")
        print(f"    GT  : {it['gt']}")
        print(f"    Pred: {it['pred']}")
        print()

if __name__ == "__main__":
    main()
