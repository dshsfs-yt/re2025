# eval_now.py  — 디폴트 경로로 즉시 평가
# (shift 플래그 반영 + 유니코드 정규화 + 처음 5개 예시에 "디코딩 전" 정보까지 표시)
import json, random, unicodedata as ud
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
    """fmt가 {s}를 포함하면 shift까지 채워서, 아니면 좌표만 채워서 생성."""
    try:
        return fmt.format(ix=ix, iy=iy, s=s)
    except KeyError:
        return fmt.format(ix=ix, iy=iy)

def canon_label(ch: str) -> str:
    """정답 정규화: <SPACE>는 보존, 그 외 텍스트는 NFKC 후 첫 글자."""
    s = str(ch).strip()
    if s in ("<SPACE>", "SPACE", " "):
        return "<SPACE>"
    if s.startswith("<") and s.endswith(">"):
        return s  # 기타 특수 토큰은 그대로
    s = ud.normalize("NFKC", s)
    return s[0] if s else ""

def canon_pred(pred: str, space_label: str) -> str:
    """예측 정규화: 공백 토큰 매핑, 토큰 형태 보존, 일반 텍스트는 NFKC 후 첫 글자."""
    p = pred.strip()
    if p in ("<SPACE>", space_label, " "):
        return "<SPACE>"
    if p.startswith("<") and p.endswith(">"):
        return p
    p = ud.normalize("NFKC", p)
    return p[0] if p else ""

@torch.inference_mode()
def batch_generate(model, tok, dev: str, prompts: List[str], bs: int):
    """
    배치로 generate 수행.
    반환:
      - pred_ids: List[List[int]] : 생성된 토큰 ID 시퀀스(디코딩 전)
      - pred_text: List[str]      : skip_special_tokens=True 로 디코딩한 문자열
      - pred_text_wsp: List[str]  : skip_special_tokens=False 로 디코딩한 문자열(스페셜 토큰 포함)
      - pred_tokens: List[List[str]] : 토큰 문자열 시퀀스(convert_ids_to_tokens)
    """
    all_ids: List[List[int]] = []
    for i in range(0, len(prompts), bs):
        sl = prompts[i:i+bs]
        enc = tok(sl, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        gen_ids = model.generate(**enc, max_new_tokens=4, do_sample=False, num_beams=1)
        # CPU로 옮겨 파이썬 리스트로 저장
        all_ids.extend([seq.cpu().tolist() for seq in gen_ids])

    pred_text      = tok.batch_decode(all_ids, skip_special_tokens=True)
    pred_text_wsp  = tok.batch_decode(all_ids, skip_special_tokens=False)
    pred_tokens    = [tok.convert_ids_to_tokens(ids) for ids in all_ids]
    return all_ids, pred_text, pred_text_wsp, pred_tokens

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
    df_s = df.sample(n=n_eval, random_state=SEED).reset_index(drop=True)

    # 프롬프트/GT 생성 + 예시 보관(원래 정답 포함)
    prompts, gts = [], []
    examples: List[Dict[str, Any]] = []
    for _, row in df_s.iterrows():
        gt_raw = str(row["ref_char"])
        gt = canon_label(gt_raw)
        ix, iy = quantize_xy(float(row["first_frame_touch_x"]), float(row["first_frame_touch_y"]), norm)
        s = int(row.get("prev_shift", 0))
        prompt = build_prompt(ix, iy, s, fmt)
        prompts.append(prompt)
        gts.append(gt)
        examples.append({
            "ix": ix, "iy": iy, "s": s,
            "prompt": prompt,
            "gt_raw": gt_raw,
            "gt": gt
        })

    # 예측 (디코딩 전/후 모두 확보)
    pred_ids, preds_raw, preds_with_special, pred_tokens = batch_generate(model, tok, dev, prompts, BATCH)
    preds = [canon_pred(p, space_label) for p in preds_raw]

    # 정확도
    correct = sum(int(gt == pr) for gt, pr in zip(gts, preds))
    acc = correct / n_eval if n_eval > 0 else 0.0

    print("===== Evaluation =====")
    print(f"Samples     : {n_eval}")
    print(f"Device      : {dev}")
    print(f"Batch size  : {BATCH}")
    print(f"Accuracy    : {correct}/{n_eval} = {acc:.2%}")

    # 처음 5개 입력/출력 결과 출력 (원래 정답 + 디코딩 전 정보)
    print("\n===== First 5 Examples (input -> output) =====")
    k = min(5, n_eval)
    for i in range(k):
        ex = examples[i]
        pr_raw = preds_raw[i].strip()
        pr_raw_wsp = preds_with_special[i].strip()
        pr = preds[i]
        ids = pred_ids[i]
        toks = pred_tokens[i]
        tf= ex["gt"] == pr
        print(
            f"[{i+1}] prompt = {ex['prompt']}"
            f"\n    gt_raw = '{ex['gt_raw']}' -> gt = '{ex['gt']}'"
            f"\n    pred_ids = {ids}"
            f"\n    pred_tokens = {toks}"
            f"\n    pred_raw_wsp = '{pr_raw_wsp}'"
            f"\n    pred_raw = '{pr_raw}' -> pred = '{pr}'"
            f"\n    correct = {tf}"
        )

if __name__ == "__main__":
    main()
