# eval_now.py — RAW 비교 + 사용자 토큰 보존 디코딩
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
    try:
        return fmt.format(ix=ix, iy=iy, s=s)
    except KeyError:
        return fmt.format(ix=ix, iy=iy)

def load_checkpoint(model_dir: Path):
    with open(model_dir / "normalization.json", "r", encoding="utf-8") as f:
        norm = json.load(f)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    return mdl, tok, norm, dev

@torch.inference_mode()
def batch_generate(model, tok, dev: str, prompts: List[str], bs: int):
    all_ids: List[List[int]] = []
    for i in range(0, len(prompts), bs):
        sl = prompts[i:i+bs]
        enc = tok(sl, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        gen_ids = model.generate(**enc, max_new_tokens=4, do_sample=False, num_beams=1)
        all_ids.extend([seq.cpu().tolist() for seq in gen_ids])

    # 참고 출력용(그대로 유지)
    pred_text_wsp = tok.batch_decode(all_ids, skip_special_tokens=False)
    pred_tokens   = [tok.convert_ids_to_tokens(ids) for ids in all_ids]
    return all_ids, pred_text_wsp, pred_tokens

def decode_preserving_user_tokens(pred_tokens: List[List[str]], tok: AutoTokenizer) -> List[str]:
    """HF 기본 스페셜(<pad>, </s>, <unk> 등)만 제거하고, 사용자 정의 토큰([BKSP], [SPACE], [MISS] 및 <...> 변형)은 보존."""
    hf_drop = set(t for t in [
        tok.pad_token, tok.eos_token, tok.unk_token, getattr(tok, "bos_token", None),
        getattr(tok, "sep_token", None), getattr(tok, "cls_token", None),
        getattr(tok, "mask_token", None),
    ] if t)

    # 우리 프로젝트에서 쓰는 사용자 토큰들(대괄호/꺾쇠 둘 다 지원)
    preserve = {"[SPACE]", "[BKSP]", "[MISS]", "<SPACE>", "<BKSP>", "<MISS>"}

    outs: List[str] = []
    for toks in pred_tokens:
        filtered = []
        for t in toks:
            # HF 기본 스페셜은 제거, 단 preserve 집합은 항상 보존
            if t in hf_drop and t not in preserve:
                continue
            filtered.append(t)
        # SentencePiece 공백 처리 등을 포함해 문자열로 복원
        s = tok.convert_tokens_to_string(filtered).strip()
        outs.append(s)
    return outs

def main():
    random.seed(SEED)
    model, tok, norm, dev = load_checkpoint(MODEL_DIR)
    fmt = norm.get("prompt_format", "coords: {ix:02d},{iy:02d} -> char")

    use_cols = {"ref_char", "first_frame_touch_x", "first_frame_touch_y", "prev_shift"}
    df = pd.read_csv(CSV_PATH, usecols=lambda c: c in use_cols)
    # 데이터의 90%만 사용 (기존 코드 유지)
    df = df[int(len(df) * 0.9):]
    df = df.dropna(subset=["ref_char", "first_frame_touch_x", "first_frame_touch_y"]).copy()

    if "prev_shift" not in df.columns:
        df["prev_shift"] = 0
    df["prev_shift"] = df["prev_shift"].fillna(0).astype(int).clip(0, 1)

    if len(df) == 0:
        print("[ERROR] 유효한 행이 없습니다."); return

    n_eval = min(N, len(df))
    df_s = df.sample(n=n_eval, random_state=SEED).reset_index(drop=True)

    prompts: List[str] = []
    gts_raw: List[str] = []
    examples: List[Dict[str, Any]] = []

    for _, row in df_s.iterrows():
        gt_raw = str(row["ref_char"])  # RAW
        ix, iy = quantize_xy(float(row["first_frame_touch_x"]), float(row["first_frame_touch_y"]), norm)
        s = int(row.get("prev_shift", 0))
        prompt = build_prompt(ix, iy, s, fmt)
        prompts.append(prompt)
        gts_raw.append(gt_raw)
        examples.append({"ix": ix, "iy": iy, "s": s, "prompt": prompt, "gt_raw": gt_raw})

    # 생성
    pred_ids, preds_with_special, pred_tokens = batch_generate(model, tok, dev, prompts, BATCH)

    #  사용자 토큰 보존 디코딩 (여기 결과로 정확도/출력 판단)
    preds_raw_keep = decode_preserving_user_tokens(pred_tokens, tok)

    # 정확도(RAW 그대로 비교: [BKSP], [SPACE] 등 문자열 자체 비교)
    correct = sum(int(gt == pr) for gt, pr in zip(gts_raw, preds_raw_keep))
    acc = correct / n_eval if n_eval > 0 else 0.0

    print("===== Evaluation (RAW match w/ user tokens preserved) =====")
    print(f"Samples     : {n_eval}")
    print(f"Device      : {dev}")
    print(f"Batch size  : {BATCH}")
    print(f"Accuracy    : {correct}/{n_eval} = {acc:.2%}")

    print("\n===== First 5 Examples (input -> output) =====")
    k = min(5, n_eval)
    for i in range(k):
        ex = examples[i]
        pr_raw = preds_raw_keep[i]
        pr_raw_wsp = preds_with_special[i]
        ids = pred_ids[i]
        toks = pred_tokens[i]
        tf = (ex["gt_raw"] == pr_raw)
        print(
            f"[{i+1}] prompt = {ex['prompt']}"
            f"\n    gt_raw = {repr(ex['gt_raw'])}"
            f"\n    pred_ids = {ids}"
            f"\n    pred_tokens = {toks}"
            f"\n    pred_raw_wsp = {repr(pr_raw_wsp)}"
            f"\n    pred_raw = {repr(pr_raw)}"
            f"\n    correct = {tf}"
        )

if __name__ == "__main__":
    main()
