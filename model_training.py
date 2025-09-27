# model_training.py — tap-logs(JSON per sentence) ➜ seq2seq(KE-T5)
import wandb
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import torch
import unicodedata as ud

# ==========================
# 0) 경로 및 기본 설정
# ==========================
JSON_DIR = Path("saved_logs_nokk")  # 문장별 JSON들이 들어있는 폴더
SAVE_DIR = Path("ckpt/ke-t5-small-RnE2025_jamosplit")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "KETI-AIR/ke-t5-small"
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

BATCH_SIZE = 48  # per device
EPOCHS     = 10000
RUN_NAME   = "ke-t5-small-RnE2025_jamosplit"
STEPS      = 5000  # 평가/저장 주기 (스텝 단위)

os.environ["WANDB_API_KEY"] = open("wandb_api_key.txt").read().strip()
os.environ["WANDB_PROJECT"] = "RnE2025"         # 선택: 프로젝트명
os.environ["WANDB_NAME"] = RUN_NAME       # 선택: 런 이름

# ==========================
# 1) 디바이스 & 로거
# ==========================
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(f"[Env] Device: {device} | CUDA: {use_cuda}")

wandb.init(project=os.environ.get("WANDB_PROJECT", "RnE2025"),
            name=os.environ.get("WANDB_NAME", RUN_NAME),
            config={"model": MODEL_NAME, "batch": BATCH_SIZE})

# ==========================
# 2) JSON 폴더 로드 & 파싱
# ==========================
SPACE_LABEL = "[SPACE]"
MISS_LABEL = "[MISS]"
BKSP_LABEL = "[BKSP]"

# -----------------------------
# Hangul syllable → Jamo string
# -----------------------------
CHOSEONG_LIST = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
JUNGSEONG_LIST = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
JONGSEONG_LIST = [
    "", "ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ",
    "ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"
]
_JONG_SPLIT = {
    "ㄳ":"ㄱㅅ", "ㄵ":"ㄴㅈ", "ㄶ":"ㄴㅎ", "ㄺ":"ㄹㄱ", "ㄻ":"ㄹㅁ", "ㄼ":"ㄹㅂ",
    "ㄽ":"ㄹㅅ", "ㄾ":"ㄹㅌ", "ㄿ":"ㄹㅍ", "ㅀ":"ㄹㅎ", "ㅄ":"ㅂㅅ"
}

def _decompose_hangul_char(ch: str) -> str:
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:  # 가(AC00)~힣(D7A3)
        s_index = code - 0xAC00
        cho = s_index // 588
        jung = (s_index % 588) // 28
        jong = s_index % 28
        out = [CHOSEONG_LIST[cho], JUNGSEONG_LIST[jung]]
        j = JONGSEONG_LIST[jong]
        if j:
            out.extend(list(_JONG_SPLIT.get(j, j)))
        return "".join(out)
    return ch  # 비한글은 그대로

def hangul_to_jamo_target(text: str, space_token: str = SPACE_LABEL) -> str:
    text = ud.normalize("NFKC", text)
    out = []
    for ch in text:
        if ch.isspace():
            if not out or out[-1] != space_token:
                out.append(space_token)
        else:
            out.append(_decompose_hangul_char(ch))
    return "".join(out).strip(space_token)

def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def _quantize_xy(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float, q: int = 100) -> Tuple[int, int]:
    xn = (x - x_min) / (x_max - x_min + 1e-9)
    yn = (y - y_min) / (y_max - y_min + 1e-9)
    xn = _clip01(xn)
    yn = _clip01(yn)
    ix = int(round(xn * (q - 1)))
    iy = int(round(yn * (q - 1)))
    return ix, iy

# 입력 프롬프트 포맷(예: "12,34@ㅇ 13,28@ㅏ ...")
PROMPT_FORMAT = "{seq}"

def _normalize_xy(e: Dict[str, Any], x_min: float, x_max: float, y_min: float, y_max: float) -> Tuple[int, int]:
    if "x_norm" in e and "y_norm" in e:
        ix = int(round(_clip01(e["x_norm"]) * 99))
        iy = int(round(_clip01(e["y_norm"]) * 99))
        return ix, iy
    return _quantize_xy(e["x"], e["y"], x_min, x_max, y_min, y_max, 100)

def _get_ix_iy(e: Dict[str, Any]) -> Tuple[int, int]:
    # 키 없음 방지용
    x = float(e.get("x", 0))
    y = float(e.get("y", 0))
    return _normalize_xy(e, x_min, x_max, y_min, y_max)

def _load_json_files(json_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in sorted(json_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # obj 구조: {target_sentence:str, completed_count:int, logs:[{target, logs:[events...]}], ...}
        # 세션들 중 마지막(완료된) 로그를 사용
        for ses in obj.get("logs", []):
            presses = ses.get("logs", [])
            tgt = obj.get("target_sentence", "")
            items.append({"presses": presses, "tgt": tgt})
    return items

all_items = _load_json_files(JSON_DIR)
print(f"[Load] {len(all_items)} sessions")

# 좌표 범위 산정(안전하게 전체 범위)
xs, ys = [], []
for it in all_items:
    for e in it["presses"]:
        if "x" in e and "y" in e:
            xs.append(float(e["x"]))
            ys.append(float(e["y"]))
x_min = min(xs) if xs else 0.0
x_max = max(xs) if xs else 1.0
y_min = min(ys) if ys else 0.0
y_max = max(ys) if ys else 1.0
print(f"[Norm] x:[{x_min:.1f},{x_max:.1f}] y:[{y_min:.1f},{y_max:.1f}]")

def build_src_from_presses(presses: List[Dict[str, Any]]) -> str:
    toks: List[str] = []
    for e in presses:
        role = e.get("role", "CHAR")
        label = e.get("label", "")
        if role == "SPACE":
            ch, ok = SPACE_LABEL, True
        elif role == "MISS":
            ch, ok = MISS_LABEL, True
        elif role == "BKSP":
            ch, ok = BKSP_LABEL, True
        else:
            ch = label or e.get("key", "")
            ok = isinstance(ch, str) and len(ch) > 0
        if not ok:
            continue
        ix, iy = _get_ix_iy(e)
        toks.append(f"{ix:02d},{iy:02d}@{ch}")
    return PROMPT_FORMAT.format(seq=" ".join(toks))

# 2-3) 최종 DF 생성(빈 시퀀스/빈 타깃 제거)
src_texts, tgt_texts = [], []
for it in all_items:
    s = build_src_from_presses(it["presses"])
    t_raw = (it["tgt"] or "").strip()
    if s.strip() and t_raw:
        t_jamo = hangul_to_jamo_target(t_raw, space_token=SPACE_LABEL)
        if t_jamo:
            src_texts.append(s)
            tgt_texts.append(t_jamo)

if len(src_texts) == 0:
    raise ValueError("모든 샘플이 비었습니다. 입력 이벤트가 남는지(MISS/BKSP 포함) 확인하세요.")

print(f"\n[Data] Loaded {len(src_texts)} samples from {JSON_DIR}")

# =====================
# 3) Dataset & Model
# =====================
raw_df = pd.DataFrame({"src": src_texts, "tgt": tgt_texts})
raw_ds = DatasetDict({
    "train": Dataset.from_pandas(raw_df.sample(frac=0.9, random_state=RANDOM_SEED).reset_index(drop=True)),
    "eval":  Dataset.from_pandas(raw_df.drop(raw_df.sample(frac=0.9, random_state=RANDOM_SEED).index).reset_index(drop=True)),
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
except Exception:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ==========================
# 4.5) 토큰 추가
# ==========================
jamo_full = sorted(set(
    CHOSEONG_LIST + JUNGSEONG_LIST +
    list("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")
))
specials = [SPACE_LABEL, BKSP_LABEL, MISS_LABEL]
existing = set(tokenizer.get_vocab().keys())
new_tokens = [t for t in (jamo_full + specials) if t not in existing]
num_added = tokenizer.add_tokens(new_tokens)
num_added += tokenizer.add_special_tokens({"additional_special_tokens":["@"]})

print(f"\n[Tokenizer] Added {num_added} tokens.")

# 모델 임베딩 길이 조절
old, new = model.get_input_embeddings().weight.size(0), len(tokenizer)
if old != new:
    print(f"\n[fix] resize_token_embeddings: {old} -> {new}")
    model.resize_token_embeddings(new)

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

tokenized = raw_ds.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_ds["train"].column_names,
)

# ==========================
# 5) Collator
# ==========================
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

# ==========================
# 6) Metrics (생성 텍스트 exact-match)
# ==========================
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # -100 패딩 제거 후 디코딩
    labels = [[t for t in seq if t != -100] for seq in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # EM
    em = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip()) / max(1, len(decoded_labels))
    return {"exact_match": em}

# ==========================
# 7) Arguments
# ==========================
args = Seq2SeqTrainingArguments(
    output_dir=str(SAVE_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="steps",
    eval_steps=STEPS,
    save_strategy="steps",
    save_steps=STEPS,
    logging_strategy="steps",
    logging_steps=50,
    report_to=["wandb"],
    predict_with_generate=True,
    generation_max_length=MAX_TGT_LEN,
    generation_num_beams=4,
    load_best_model_at_end=True,
    metric_for_best_model="exact_match",
    greater_is_better=True,
    fp16=use_cuda,  # CUDA일 때만 fp16 사용
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["eval"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# ==========================
# 8) Train & Save
# ==========================
trainer.train()
trainer.save_model(str(SAVE_DIR))
tokenizer.save_pretrained(str(SAVE_DIR))

# 정규화/프롬프트 메타 저장
norm_params = {
    "x_min": x_min, "x_max": x_max,
    "y_min": y_min, "y_max": y_max,
    "quantize_to": 100,
    "prompt_format": PROMPT_FORMAT,
    "space_label": SPACE_LABEL,
    "miss_label": MISS_LABEL,
    "bksp_label": BKSP_LABEL,
    "input_token_pattern": "{ix:02d},{iy:02d}@{ch}",
    "roles_used": ["CHAR", "SPACE", "MISS", "BKSP"],
    "roles_ignored": [],
}
with open(SAVE_DIR / "normalization.json", "w", encoding="utf-8") as f:
    json.dump(norm_params, f, ensure_ascii=False, indent=2)

print(f"[OK] Saved to: {SAVE_DIR.resolve()}")
