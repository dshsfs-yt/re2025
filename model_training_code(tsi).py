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

CSV_PATH = "touch_data.csv"  # 터치 데이터 CSV 파일 경로

SAVE_DIR = Path("ckpt/ke-t5-small-touch-only")  # 모델/토크나이저/정규화 파라미터 저장 위치
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
        # cuDNN 튜닝
        torch.backends.cudnn.benchmark = True
        # bf16 지원(암페어 이상) 시 bf16 우선, 아니면 fp16
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
# 선택적: FP32 matmul 가속 (CUDA에서만 의미 있음)
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
# 1) 터치 CSV 로드 & 전처리 (키보드 메타 불필요)
# ==========================
use_cols = [
    "ref_char",
    "first_frame_touch_x",
    "first_frame_touch_y",
    "was_deleted",  # 없으면 무시
]
df = pd.read_csv(
    CSV_PATH,
    usecols=lambda c: c in use_cols or c in ["ref_char", "first_frame_touch_x", "first_frame_touch_y"]
)

df=df[:40000] 

# 결측치 제거
df = df.dropna(subset=["ref_char", "first_frame_touch_x", "first_frame_touch_y"]).copy()

# 삭제된 터치 제외 (열이 있을 때만)
if "was_deleted" in df.columns:
    df = df[df["was_deleted"] == False].copy()

# ref_char 정제: 'SPACE' -> '<SPACE>' (원하면 ' ' 로 바꿔도 됨)
def map_ref_char(c: str) -> str:
    c = str(c)
    return "<SPACE>" if c == "SPACE" else c

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

# 필요한 열만 유지
df = df[["x", "y", "ref_char"]].reset_index(drop=True)

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

# (참고) Trainer가 내부에서 자동으로 모델을 디바이스로 옮기지만,
# 명시적으로 옮겨도 무방.
try:
    model.to(device)
except Exception:
    pass

# ==========================
# 5) 토크나이즈 함수 (좌표 0..1 → 00..99 버킷)
# ==========================
def make_tokenize_function_01(max_src_len=32, max_tgt_len=8):
    def _q99(v: float) -> int:
        v = max(0.0, min(float(v), 1.0))
        return int(round(v * 99))

    def tokenize_function(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        xs  = [ _q99(v) for v in batch["x"] ]
        ys  = [ _q99(v) for v in batch["y"] ]
        tgt = [ str(v) for v in batch["ref_char"] ]  # '<SPACE>' 또는 단일 문자

        prompts = [ f"coords: {ix:02d},{iy:02d} -> char" for ix,iy in zip(xs,ys) ]

        enc = tokenizer(prompts, max_length=max_src_len, truncation=True)
        lab = tokenizer(text_target=tgt, max_length=max_tgt_len, truncation=True)
        enc["labels"] = lab["input_ids"]
        return enc

    return tokenize_function

tokenize_function = make_tokenize_function_01(max_src_len=32, max_tgt_len=8)

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

# 혼합정밀/핀메모리/옵티마이저 자동 설정
optim_choice = "adamw_torch_fused" if use_cuda else "adamw_torch"

args = TrainingArguments(
    output_dir=str(SAVE_DIR),              # 체크포인트도 여기에
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="steps",           # ← 올바른 파라미터명
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_strategy="steps",
    logging_steps=50,
    eval_delay=0,
    report_to="none",                      # wandb 등 안 쓸 경우
    dataloader_pin_memory=use_cuda,        # CUDA 아닐 땐 경고 방지
    optim=optim_choice,
    **precision_kwargs,                    # bf16 또는 fp16 자동 적용
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
# (1) 모델/토크나이저 저장
trainer.save_model(str(SAVE_DIR))           # 모델 가중치 + config
tokenizer.save_pretrained(str(SAVE_DIR))    # 토크나이저

# (2) 정규화 파라미터 저장 (추론 시 재사용)
norm_params = {
    "x_min": x_min, "x_max": x_max,
    "y_min": y_min, "y_max": y_max,
    "quantize_to": 100,  # 0..99 버킷
    "prompt_format": "coords: {ix:02d},{iy:02d} -> char",
    "space_label": "<SPACE>",
}
with open(SAVE_DIR / "normalization.json", "w", encoding="utf-8") as f:
    json.dump(norm_params, f, ensure_ascii=False, indent=2)

print(f"[OK] Saved to: {SAVE_DIR.resolve()}")
