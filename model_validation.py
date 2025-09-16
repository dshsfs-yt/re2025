# eval_now.py  — 디폴트 경로로 즉시 평가 (shift 플래그 반영)
import json, random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = Path("ckpt/ke-t5-small-touch-only(korean)")
CSV_PATH  = "touch_data(korean).csv"  # 열: ref_char, first_frame_touch_x, first_frame_touch_y, prev_shift
N         = 1000
BATCH     = 64
SEED      = 42

def load_checkpoint(model_dir: Path):
    with open(model_dir / "normalization.json", "r", encoding="utf-8") as f:
        norm = json.load(f)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    return mdl, tok, norm, dev

def clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def quantize_xy(x_raw: float, y_raw: float, norm: Dict[str, Any]) -> Tuple[int, int]:
    x_min, x_max = float(norm["x_min"]), float(norm["x_max"])
    y_min, y_max = float(norm["y_min"]), float(norm["y_max"])
    qto = int(norm.get("quantize_to", 100))
    x01 = 0.5 if x_max <= x_min else clip01((x_raw - x_min) / (x_max - x_min))
    y01 = 0.5 if y_max <= y_min else clip01((y_raw - y_min) / (y_max - y_min))
    return int(round(x01 * (qto - 1))), int(round(y01 * (qto - 1)))

def build_prompt(ix: int, iy: int, s: int, fmt: str) -> str:
    """fmt가 shift 자리표시자 {s}를 포함하면 사용, 없으면 (구형 포맷) 좌표만 적용."""
    try:
        return fmt.format(ix=ix, iy=iy, s=s)
    except KeyError:
        return fmt.format(ix=ix, iy=iy)

def canon_label(ch: str) -> str:
    """학습과 동일하게 <SPACE> 정규화 유지."""
    s = str(ch).strip()
    if s in ("<SPACE>", "SPACE", " "):
        return "<SPACE>"
    return s  # 단일 문자/토큰 가정, 각괄호 토큰은 그대로 유지

def canon_pred(pred: str, space_label: str) -> str:
    """모델 출력 정규화: 공백 토큰만 일관 매핑, 나머지는 가능하면 1문자로 축약하되 각괄호 토큰은 유지."""
    p = pred.strip()
    if p in ("<SPACE>", space_label, " "):
        return "<SPACE>"
    # 각괄호 토큰(예: <SPACE>)은 그대로 두고, 일반 텍스트가 여러 글자면 1글자만 비교
    if len(p) > 1 and not p.startswith("<"):
        return p[0]
    return p

@torch.inference_mode()
def batch_predict(model, tok, dev: str, prompts: List[str], bs: int) -> List[str]:
    outs: List[str] = []
    for i in range(0, len(prompts), bs):
        sl = prompts[i:i+bs]
        enc = tok(sl, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        gen = model.generate(**enc, max_new_tokens=4, do_sample=False, num_beams=1)
        outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return outs

def main():
    model, tok, norm, dev = load_checkpoint(MODEL_DIR)
    space_label = norm.get("space_label", "<SPACE>")
    fmt = norm.get("prompt_format", "coords: {ix:02d},{iy:02d} -> char")

    # --- CSV 로드: 최신 4열 스키마 사용 ---
    use_cols = {"ref_char", "first_frame_touch_x", "first_frame_touch_y", "prev_shift"}
    df = pd.read_csv(CSV_PATH, usecols=lambda c: c in use_cols)
    df = df.dropna(subset=["ref_char", "first_frame_touch_x", "first_frame_touch_y"]).copy()

    # prev_shift 정리 (없으면 0)
    if "prev_shift" not in df.columns:
        df["prev_shift"] = 0
    df["prev_shift"] = df["prev_shift"].fillna(0).astype(int).clip(0, 1)

    if len(df) == 0:
        print("[ERROR] 유효한 행이 없습니다."); return

    # 샘플링
    random.seed(SEED)
    n_eval = min(N, len(df))
    df_s = df.sample(n=n_eval, random_state=SEED)

    # 프롬프트 생성
    prompts, gts = [], []
    for _, row in df_s.iterrows():
        gt = canon_label(row["ref_char"])
        ix, iy = quantize_xy(float(row["first_frame_touch_x"]), float(row["first_frame_touch_y"]), norm)
        s = int(row.get("prev_shift", 0))
        prompts.append(build_prompt(ix, iy, s, fmt))
        gts.append(gt)

    # 예측
    preds_raw = batch_predict(model, tok, dev, prompts, BATCH)
    preds = [canon_pred(p, space_label) for p in preds_raw]

    # 정확도
    correct = sum(int(gt == pr) for gt, pr in zip(gts, preds))
    acc = correct / n_eval if n_eval > 0 else 0.0

    print("===== Evaluation =====")
    print(f"Samples     : {n_eval}")
    print(f"Device      : {dev}")
    print(f"Batch size  : {BATCH}")
    print(f"Accuracy    : {correct}/{n_eval} = {acc:.2%}")

if __name__ == "__main__":
    main()
