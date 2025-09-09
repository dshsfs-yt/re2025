# tap_typing_ke_t5_train_pairs2sent.py
# 입력: CSV with columns = ["pairs", "target"]
#   - pairs: JSON string like
#       [{"ch":"ㄱ","x":1108.0,"y":312.3}, {"ch":"ㅏ","x":1042.0,"y":298.1}, ...]
#   - target: 정답 문장 (string)
# 출력: ckpt/ke-t5-small-pairs2sent/ 에 모델/토크나이저/정규화 파라미터 저장

import json
from pathlib import Path
from typing import Any, Dict, List

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
CSV_PATH = "tap_sequences_dummy_100.csv"  
SAVE_DIR = Path("ckpt/ke-t5-small-pairs2sent")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "KETI-AIR/ke-t5-small"
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

# ==========================
# 1) 디바이스/정밀도
# ==========================
def get_device_and_precision():
    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        precision = {"bf16": torch.cuda.is_bf16_supported(), "fp16": False}
    elif use_mps:
        device = torch.device("mps")
        precision = {"bf16": False, "fp16": False}
    else:
        device = torch.device("cpu")
        precision = {"bf16": False, "fp16": False}
    return device, precision, use_cuda

device, precision_kwargs, use_cuda = get_device_and_precision()
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
print(f"[Device] {device} | CUDA: {use_cuda}")

# ==========================
# 2) 데이터 로드
#   기대 스키마:
#     - pairs: JSON string -> list[ { "ch": str, "x": float, "y": float } ]
#     - target: str (정답 문장)
# ==========================
raw_df = pd.read_csv(CSV_PATH, usecols=["pairs", "target"])
print(f"[Data] Loaded {len(raw_df)} rows from {CSV_PATH}")
raw_df=raw_df[:(int(len(raw_df)*0.9))] 

raw_df = raw_df.dropna(subset=["pairs", "target"]).reset_index(drop=True)

def _parse_pairs(js: str) -> List[Dict[str, Any]]:
    try:
        arr = json.loads(js)
        # 최소 필드 보정
        out = []
        for it in arr:
            ch = str(it.get("ch", ""))
            x = float(it.get("x", 0.0))
            y = float(it.get("y", 0.0))
            out.append({"ch": ch, "x": x, "y": y})
        return out
    except Exception:
        return []

pairs_list: List[List[Dict[str, Any]]] = [ _parse_pairs(s) for s in raw_df["pairs"].tolist() ]
targets: List[str] = raw_df["target"].tolist()

# 비어있는 샘플 제거
filtered_pairs, filtered_targets = [], []
for p, t in zip(pairs_list, targets):
    if isinstance(p, list) and len(p) > 0 and isinstance(t, str) and len(t) > 0:
        filtered_pairs.append(p)
        filtered_targets.append(t)

if len(filtered_pairs) == 0:
    raise ValueError("유효한 샘플이 없습니다. 'pairs'(JSON)와 'target'(str) 컬럼을 확인하세요.")

# ==========================
# 3) 좌표 전역 정규화 파라미터 계산
# ==========================
xs = [pt["x"] for seq in filtered_pairs for pt in seq]
ys = [pt["y"] for seq in filtered_pairs for pt in seq]
x_min, x_max = float(min(xs)), float(max(xs))
y_min, y_max = float(min(ys)), float(max(ys))

def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def _to01(v: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.5
    return _clip01((float(v) - vmin) / (vmax - vmin))

def _q99(v01: float) -> int:
    return int(round(_clip01(v01) * 99))

SPACE_LABEL = "<SPACE>"  # 공백 표기 일관화(원하면 '▁' 등으로 교체 가능)

def _canon_char(ch: str) -> str:
    # 입력 글자 표준화 (예: ' ' 또는 'SPACE' 등)
    ch = str(ch)
    if ch == " " or ch.upper() == "SPACE":
        return SPACE_LABEL
    return ch[:1] if len(ch) > 0 else ""

# ==========================
# 4) 프롬프트 구성
#    touchseq: <ch>@<ix,iy> ...  -> text
# ==========================
PROMPT_FORMAT = "touchseq: {seq} -> text"

def build_src_text(seq: List[Dict[str, Any]]) -> str:
    toks: List[str] = []
    for pt in seq:
        ch = _canon_char(pt["ch"])
        ix = _q99(_to01(pt["x"], x_min, x_max))
        iy = _q99(_to01(pt["y"], y_min, y_max))
        toks.append(f"{ch}@{ix:02d},{iy:02d}")
    return PROMPT_FORMAT.format(seq=" ".join(toks))

src_texts: List[str] = [ build_src_text(seq) for seq in filtered_pairs ]
tgt_texts: List[str] = filtered_targets

# ==========================
# 5) HF Datasets 변환 & 스플릿
# ==========================
df_for_ds = pd.DataFrame({"src": src_texts, "tgt": tgt_texts})
all_ds = Dataset.from_pandas(df_for_ds, preserve_index=False).shuffle(seed=RANDOM_SEED)

n = len(all_ds)
split = int(n * 0.95) if n > 20 else max(1, n - 1)
raw_ds = DatasetDict({
    "train": all_ds.select(range(split)),
    "validation": all_ds.select(range(split, n)),
})
print(raw_ds)

# ==========================
# 6) 토크나이저/모델 및 토크나이즈
# ==========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
try:
    model.to(device)
except Exception:
    pass

MAX_SRC_LEN = 512
MAX_TGT_LEN = 256

def tokenize_function(batch: Dict[str, List[str]]) -> Dict[str, Any]:
    enc = tokenizer(batch["src"], max_length=MAX_SRC_LEN, truncation=True)
    lab = tokenizer(text_target=batch["tgt"], max_length=MAX_TGT_LEN, truncation=True)
    enc["labels"] = lab["input_ids"]
    return enc

tokenized = raw_ds.map(tokenize_function, batched=True, remove_columns=raw_ds["train"].column_names)

# ==========================
# 7) Collator & Trainer
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
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_strategy="steps",
    logging_steps=50,
    eval_delay=0,
    report_to="none",
    dataloader_pin_memory=use_cuda,
    optim=optim_choice,
    **precision_kwargs,
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
# 8) 학습 & 저장
# ==========================
trainer.train()

# (1) 모델/토크나이저 저장
trainer.save_model(str(SAVE_DIR))
tokenizer.save_pretrained(str(SAVE_DIR))

# (2) 정규화 파라미터 저장
norm_params = {
    "x_min": x_min, "x_max": x_max,
    "y_min": y_min, "y_max": y_max,
    "quantize_to": 100,
    "prompt_format": PROMPT_FORMAT,
    "space_label": SPACE_LABEL,
    "input_token_pattern": "{ch}@{ix:02d},{iy:02d}",
}
with open(SAVE_DIR / "normalization.json", "w", encoding="utf-8") as f:
    json.dump(norm_params, f, ensure_ascii=False, indent=2)

print(f"[OK] Saved to: {SAVE_DIR.resolve()}")
