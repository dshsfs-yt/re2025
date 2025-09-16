# eval_now.py  — 디폴트 경로로 즉시 평가
import json, random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = Path("ckpt/ke-t5-small-touch-only")
CSV_PATH  = "touch_data.csv"
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

def build_prompt(ix: int, iy: int, fmt: str) -> str:
    return fmt.format(ix=ix, iy=iy)

def canon_label(ch: str) -> str:
    ch = str(ch)
    return "<SPACE>" if ch == "SPACE" else ch[:1]

def canon_pred(pred: str, space_label: str) -> str:
    p = pred.strip()
    return "<SPACE>" if p == space_label else (p[:1] if p else "")

@torch.inference_mode()
def batch_predict(model, tok, dev: str, prompts: List[str], bs: int) -> List[str]:
    outs: List[str] = []
    for i in range(0, len(prompts), bs):
        sl = prompts[i:i+bs]
        enc = tok(sl, return_tensors="pt", padding=True, truncation=True)
        enc = {k:v.to(dev) for k,v in enc.items()}
        gen = model.generate(**enc, max_new_tokens=4, do_sample=False, num_beams=1)
        outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return outs

def main():
    model, tok, norm, dev = load_checkpoint(MODEL_DIR)
    space_label = norm.get("space_label", "<SPACE>")

    use_cols = ["ref_char","first_frame_touch_x","first_frame_touch_y","was_deleted"]
    df = pd.read_csv(CSV_PATH, usecols=lambda c: c in use_cols)
    df=df[40000:]
    df = df.dropna(subset=["ref_char","first_frame_touch_x","first_frame_touch_y"]).copy()
    if "was_deleted" in df.columns:
        # 다양한 표현을 False로 처리
        mask = (df["was_deleted"] == False) | (df["was_deleted"] == "False") | (df["was_deleted"] == 0) | (df["was_deleted"].isna())
        df = df[mask].copy()
    if len(df) == 0:
        print("[ERROR] 유효한 행이 없습니다."); return

    random.seed(SEED)
    n_eval = min(N, len(df))
    df_s = df.sample(n=n_eval, random_state=SEED)

    prompts, gts = [], []
    fmt = norm.get("prompt_format", "coords: {ix:02d},{iy:02d} -> char")
    for _, row in df_s.iterrows():
        gt = canon_label(row["ref_char"])
        ix, iy = quantize_xy(float(row["first_frame_touch_x"]), float(row["first_frame_touch_y"]), norm)
        prompts.append(build_prompt(ix, iy, fmt))
        gts.append(gt)

    preds_raw = batch_predict(model, tok, dev, prompts, BATCH)
    preds = [canon_pred(p, space_label) for p in preds_raw]

    correct = sum(int(gt == pr) for gt, pr in zip(gts, preds))
    acc = correct / n_eval
    print("===== Evaluation =====")
    print(f"Samples        : {n_eval}")
    print(f"Device         : {dev}")
    print(f"Batch size     : {BATCH}")
    print(f"Accuracy     : {correct}/{n_eval} = {acc:.2%}")

if __name__ == "__main__":
    main()