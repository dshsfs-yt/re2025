# tap_typing_ke_t5_train_touch_only.py
import os
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
import torch

# ==========================
# 0) 경로 및 기본 설정
# ==========================

CSV_PATH = "touch_data(korean).csv"  # 터치 데이터 CSV 파일 경로 (열: ref_char, first_frame_touch_x, first_frame_touch_y, prev_shift)

SAVE_DIR = Path("ckpt/ke-t5-small-touch-only(korean)")  # 모델/토크나이저/정규화 파라미터 저장 위치
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "KETI-AIR/ke-t5-small"
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

# --------------------------
# GPU/MPS 감지 & 정밀도 결정
# --------------------------
def get_device_and_precision():
    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        bf16_ok = torch.cuda.is_bf16_supported()
        precision = {"bf16": bf16_ok, "fp16": (not bf16_ok)}
    elif use_mps:
        device = torch.device("mps")
        precision = {"bf16": False, "fp16": False}
    else:
        device = torch.device("cpu")
        precision = {"bf16": False, "fp16": False}

    return device, precision, use_cuda, use_mps

device, precision_kwargs, use_cuda, use_mps = get_device_and_precision()
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print(f"[Device] {device} | CUDA: {use_cuda} | MPS: {use_mps}")
if use_cuda:
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        bf16_sup = torch.cuda.is_bf16_supported()
        print(f"[CUDA] device_name={name} | capability={cap} | bf16_supported={bf16_sup}")
    except Exception as e:
        print(f"[CUDA] info fetch error: {e}")

# ==========================
# 1) 터치 CSV 로드 & 전처리
# ==========================
use_cols = {"ref_char", "first_frame_touch_x", "first_frame_touch_y", "prev_shift"}

df = pd.read_csv(
    CSV_PATH,
    usecols=lambda c: c in use_cols  # 필요한 4열만
)

# 데이터의 90%만 사용 (기존 코드 유지)
df = df[: int(len(df) * 0.9)]

# 결측치 제거
df = df.dropna(subset=["ref_char", "first_frame_touch_x", "first_frame_touch_y"]).copy()

# prev_shift 기본값/형 변환
if "prev_shift" not in df.columns:
    df["prev_shift"] = 0
df["prev_shift"] = df["prev_shift"].fillna(0).astype(int).clip(0, 1)

# ref_char 정제: 'SPACE' / ' ' / '<SPACE>'를 통일해 '<SPACE>'로
def map_ref_char(c: str) -> str:
    s = str(c)
    if s == "<SPACE>" or s == " " or s.upper() == "SPACE":
        return "<SPACE>"
    return s

df["ref_char"] = df["ref_char"].map(map_ref_char)

# ==========================
# 2) 좌표 정규화 (CSV 자체 min/max 사용)
# ==========================
x_min = float(df["first_frame_touch_x"].min())
x_max = float(df["first_frame_touch_x"].max())
y_min = float(df["first_frame_touch_y"].min())
y_max = float(df["first_frame_touch_y"].max())

def norm01(v: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.5  # 안전장치
    v = (float(v) - vmin) / (vmax - vmin)
    return max(0.0, min(1.0, v))

df["x"] = df["first_frame_touch_x"].apply(lambda v: norm01(v, x_min, x_max))
df["y"] = df["first_frame_touch_y"].apply(lambda v: norm01(v, y_min, y_max))

# 필요한 열만 유지 (prev_shift 포함)
df = df[["x", "y", "prev_shift", "ref_char"]].reset_index(drop=True)

# ==========================
# 3) HF Datasets 변환 & 스플릿
# ==========================
ds_all = Dataset.from_pandas(df, preserve_index=False).shuffle(seed=RANDOM_SEED)
n = len(ds_all)
split = int(n * 0.95) if n > 20 else max(1, n - 1)  # 작은 데이터 안전장치

raw_ds = DatasetDict({
    "train": ds_all.select(range(split)),
    "validation": ds_all.select(range(split, n)),
})

print(raw_ds)

# ==========================
# 4) 토크나이저/모델 로드
# ==========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ==========================
# 4.5) 토큰 추가
# ==========================

new_tokens = ["ㅆ", "ㅃ", "ㅈ", "ㅕ", "ㅑ", "ㅖ", "ㅣ", "ㄸ", "ㅗ", "ㅌ", "ㅍ", "ㅒ", "ㅔ", "ㅏ", "ㅊ", "ㅓ", "ㅉ", "ㅛ", "ㅐ", "ㅁ", "ㅂ", "ㄲ"]
special_tokens = {"additional_special_tokens": ["[SPACE]","[BKSP]"]}

num_added=tokenizer.add_tokens(new_tokens)
num_added += tokenizer.add_special_tokens(special_tokens)

print(f"[Tokenizer] Added {num_added} tokens.")





try:
    model.to(device)
except Exception:
    pass

# ==========================
# 5) 토크나이즈 함수 (좌표 0..1 → 00..99 버킷 + shift 포함)
# ==========================
def make_tokenize_function_01(max_src_len=40, max_tgt_len=8):
    def _q99(v: float) -> int:
        v = max(0.0, min(float(v), 1.0))
        return int(round(v * 99))

    def tokenize_function(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        xs  = [_q99(v) for v in batch["x"]]
        ys  = [_q99(v) for v in batch["y"]]
        #ss  = [int(bool(s)) for s in batch["prev_shift"]]  # 0/1
        ss=[]
        for s in batch["prev_shift"]:
            if s=="True":
                ss.append(1)
            elif s=="False":
                ss.append(0)
        tgt = [str(v) for v in batch["ref_char"]]          # '<SPACE>' 또는 단일 문자

        # shift 정보를 프롬프트에 포함
        # 예: "coords: 12,83 shift:1 -> char"
        prompts = [f"coords: {ix:02d},{iy:02d} shift:{s} -> char"
                   for ix, iy, s in zip(xs, ys, ss)]

        enc = tokenizer(prompts, max_length=max_src_len, truncation=True)
        lab = tokenizer(text_target=tgt, max_length=max_tgt_len, truncation=True)
        enc["labels"] = lab["input_ids"]
        return enc

    return tokenize_function

tokenize_function = make_tokenize_function_01(max_src_len=40, max_tgt_len=8)

tokenized = raw_ds.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_ds["train"].column_names,
)

# ==========================
# 6) Collator & TrainingArguments & Trainer
# ==========================
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

optim_choice = "adamw_torch_fused" if use_cuda else "adamw_torch"

args = TrainingArguments(
    output_dir=str(SAVE_DIR),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,            # 에폭 증가
    eval_strategy="steps",           # 사용자가 지정한 파라미터명 유지
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_strategy="steps",
    logging_steps=50,
    eval_delay=0,
    report_to="none",
    dataloader_pin_memory=use_cuda,
    optim=optim_choice,
    **precision_kwargs,              # bf16 또는 fp16 자동 적용
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator,
    tokenizer=tokenizer,
)

# ==========================
# 7) 학습
# ==========================
trainer.train()

# ==========================
# 8) 저장: 모델, 토크나이저, 정규화 파라미터
# ==========================
trainer.save_model(str(SAVE_DIR))
tokenizer.save_pretrained(str(SAVE_DIR))

norm_params = {
    "x_min": x_min, "x_max": x_max,
    "y_min": y_min, "y_max": y_max,
    "quantize_to": 100,  # 0..99 버킷
    "prompt_format": "coords: {ix:02d},{iy:02d} shift:{s} -> char",
    "space_label": "<SPACE>",
    "uses_shift_flag": True,
}
with open(SAVE_DIR / "normalization.json", "w", encoding="utf-8") as f:
    json.dump(norm_params, f, ensure_ascii=False, indent=2)

print(f"[OK] Saved to: {SAVE_DIR.resolve()}")
