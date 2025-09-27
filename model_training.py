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
RUN_NAME  = "ke-t5-small-RnE2025_jamosplit"
STEPS=5000

# -------------------------
# Hangul constants & decomposition maps
# -------------------------
CHOSEONG = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
# 기본(단일) 중성만 둔다 — 복합 중성은 아래 맵으로 분해하여 사용
JUNGSEONG = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅛ","ㅜ","ㅠ","ㅡ","ㅣ"]
# 종성 인덱스 표 (원래 표) — 필요에 따라 참조
JONGSEONG = ["", "ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]

HANGUL_BASE = 0xAC00
NUM_JUNG = 21
NUM_JONG = 28

# 복합 중성 -> 기본 자모 분해 맵 (요청하신 대로 ㅘ -> ㅗ + ㅏ 등)
JUNG_DECOMPOSE_MAP = {
    "ㅘ": ["ㅗ","ㅏ"],
    "ㅙ": ["ㅗ","ㅐ"],
    "ㅚ": ["ㅗ","ㅣ"],
    "ㅝ": ["ㅜ","ㅓ"],
    "ㅞ": ["ㅜ","ㅔ"],
    "ㅟ": ["ㅜ","ㅣ"],
    "ㅢ": ["ㅡ","ㅣ"],
    # 단일 중성(예: ㅏ, ㅓ ...)은 그대로 처리 (맵에 없음)
}

# 복합 종성(겹받침)을 각 자음으로 분해하는 맵 — 전 항목 포함
JONG_DECOMPOSE_MAP = {
    "ㄳ": ["ㄱ","ㅅ"],
    "ㄵ": ["ㄴ","ㅈ"],
    "ㄶ": ["ㄴ","ㅎ"],
    "ㄺ": ["ㄹ","ㄱ"],
    "ㄻ": ["ㄹ","ㅁ"],
    "ㄼ": ["ㄹ","ㅂ"],
    "ㄽ": ["ㄹ","ㅅ"],
    "ㄾ": ["ㄹ","ㅌ"],
    "ㄿ": ["ㄹ","ㅍ"],
    "ㅀ": ["ㄹ","ㅎ"],
    "ㅄ": ["ㅂ","ㅅ"],
    # 단일 종성(ㄱ, ㄴ, ...)은 맵에 없음 -> 그대로 사용
}

def is_hangul_syllable(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)
    return HANGUL_BASE <= o <= 0xD7A3

def decompose_hangul_char_to_key_jamos(ch: str) -> list:
    """
    한 음절을 '키 입력에 가까운' 자모 시퀀스로 분해.
    예: '값' -> ['ㄱ','ㅏ','ㄱ','ㅅ']  (ㄳ -> ㄱ,ㅅ)
         '과' -> ['ㄱ','ㅗ','ㅏ']     (ㅘ -> ㅗ,ㅏ)
    """
    sindex = ord(ch) - HANGUL_BASE
    l_index = sindex // (NUM_JUNG * NUM_JONG)
    v_index = (sindex % (NUM_JUNG * NUM_JONG)) // NUM_JONG
    t_index = sindex % NUM_JONG

    parts = []
    # 초성
    lead = CHOSEONG[l_index]
    parts.append(lead)

    # 중성: 복합이면 분해, 아니면 단일
    # NOTE: JUNGSEONG (인덱스 표)에는 복합 모음(예: ㅘ 등)도 포함되어 있기 때문에
    #       해당 인덱스로 얻은 값이 분해 맵에 있으면 분해한다.
    # To get the jung string for v_index, map to original 21-list of JUNGSEONG_INDEXED:
    # The classic 21-list including composite items:
    JUNG_FULL = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
    jung = JUNG_FULL[v_index]
    if jung in JUNG_DECOMPOSE_MAP:
        parts.extend(JUNG_DECOMPOSE_MAP[jung])
    else:
        parts.append(jung)

    # 종성: 있으면 단일 또는 복합 -> 복합이면 분해해서 각 자음 추가
    if t_index != 0:
        jong = JONGSEONG[t_index]
        if jong in JONG_DECOMPOSE_MAP:
            parts.extend(JONG_DECOMPOSE_MAP[jong])
        else:
            parts.append(jong)

    return parts

def decompose_to_jamos_heavy(text: str, space_label="[SPACE]") -> str:
    """
    문장 전체를 분해(복합중성/복합종성 분해 포함).
    공백은 space_label 토큰으로 대체.
    """
    out = []
    for ch in (text or ""):
        if ch == " ":
            out.append(space_label)
            continue
        if is_hangul_syllable(ch):
            out.extend(decompose_hangul_char_to_key_jamos(ch))
            continue
        # 숫자/영문/특수는 그대로 추가 (필요 시 규칙 추가)
        out.append(ch)
    return "".join(out)



os.environ["WANDB_API_KEY"] = open("wandb_api_key.txt").read().strip()
os.environ["WANDB_PROJECT"] = "RnE2025"         # 선택: 프로젝트명
os.environ["WANDB_NAME"] = RUN_NAME       # 선택: 런 이름

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
# 2) JSON 폴더 로드 & 파싱
# ==========================
SPACE_LABEL = "[SPACE]"
MISS_LABEL = "[MISS]"
BKSP_LABEL = "[BKSP]"

def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def _to01(v: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.5
    return _clip01((float(v) - vmin) / (vmax - vmin))

def _q99(v01: float) -> int:
    return int(round(_clip01(v01) * 99))

def _canon_char_from_log(role: str, label: str) -> Tuple[bool, str]:
    """학습 입력에 포함할지/무엇으로 넣을지 결정.
    - CHAR: label의 첫 글자
    - SPACE: [SPACE]
    - MISS: [MISS]  (좌표가 있지만 글자가 안 찍힌 터치)
    - BKSP: [BKSP]  (백스페이스 키 입력 자체를 신호로 사용)
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
    # 기타(필요하면 확장)
    return False, ""

def _iter_json_files(json_dir: Path):
    for p in sorted(json_dir.glob("*.json")):
        yield p

def _load_one_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 2-1) 1차 로드: 정규화 방식 판단 및 raw 좌표 수집(백업용)
all_items: List[Dict[str, Any]] = []
raw_xs, raw_ys = [], []
has_norm_xy = True  # x_norm / y_norm 가정

files = list(_iter_json_files(JSON_DIR))
if len(files) == 0:
    raise FileNotFoundError(f"No JSON files in: {JSON_DIR.resolve()}")

for fp in files:
    obj = _load_one_json(fp)
    tgt = obj.get("target_sentence") or ""
    logs_blocks = obj.get("logs") or []
    if not tgt or not logs_blocks:
        continue
    block0 = logs_blocks[0] or {}
    presses = block0.get("logs") or []

    for e in presses:
        role = (e.get("role") or "").upper()
        # 네 가지 역할 모두 좌표 확인 대상
        if ("x_norm" not in e) or ("y_norm" not in e):
            has_norm_xy = False
        if "x" in e and "y" in e:
            try:
                raw_xs.append(float(e["x"]))
                raw_ys.append(float(e["y"]))
            except Exception:
                pass

    all_items.append({"tgt": tgt, "presses": presses})

if not all_items:
    raise ValueError("유효한 샘플이 없습니다. JSON 구조와 필드를 확인하세요.")

if (not has_norm_xy) and (not raw_xs or not raw_ys):
    raise ValueError("x_norm/y_norm이 없고 raw x,y도 충분치 않습니다. 입력 JSON을 확인하세요.")

# 2-2) 정규화 파라미터 계산(backup: raw)
if has_norm_xy:
    norm_source = "x_norm"
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0
else:
    norm_source = "raw"
    x_min, x_max = float(min(raw_xs)), float(max(raw_xs))
    y_min, y_max = float(min(raw_ys)), float(max(raw_ys))

def _get_ix_iy(e: Dict[str, Any]) -> Tuple[int, int]:
    if norm_source == "x_norm":
        x01 = float(e.get("x_norm", 0.5))
        y01 = float(e.get("y_norm", 0.5))
        return _q99(x01), _q99(y01)
    else:
        x = float(e.get("x", 0.0))
        y = float(e.get("y", 0.0))
        return _q99(_to01(x, x_min, x_max)), _q99(_to01(y, y_min, y_max))

PROMPT_FORMAT = "touchseq: {seq} -> text"

def build_src_from_presses(presses: List[Dict[str, Any]]) -> str:
    toks: List[str] = []
    for e in presses:
        ok, ch = _canon_char_from_log(e.get("role"), e.get("label"))
        if not ok:
            continue
        ix, iy = _get_ix_iy(e)
        toks.append(f"{ix:02d},{iy:02d}@{ch}")
    return PROMPT_FORMAT.format(seq=" ".join(toks))

# 2-3) 최종 DF 생성(빈 시퀀스/빈 타깃 제거)
src_texts, tgt_texts = [], []
for it in all_items:
    s = build_src_from_presses(it["presses"])
    raw_t = (it["tgt"] or "").strip()
    if s.strip() and raw_t:
        # 타깃을 자소 단위로 분해해서 사용
        t = decompose_to_jamos_heavy(raw_t)
        src_texts.append(s)
        tgt_texts.append(t)

if len(src_texts) == 0:
    raise ValueError("모든 샘플이 비었습니다. 입력 이벤트가 남는지(MISS/BKSP 포함) 확인하세요.")

print(f"\n[Data] Loaded {len(src_texts)} samples from {JSON_DIR}")

# ==========================
# 3) HF Datasets 변환 & 스플릿
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
# 4) 토크나이저/모델 및 토크나이즈
# ==========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# ==========================
# 4.5) 토큰 추가
# ==========================


new_tokens = ["ㅆ", "ㅃ", "ㅈ", "ㅕ", "ㅑ", "ㅖ", "ㅣ", "ㄸ", "ㅗ", "ㅌ", "ㅍ", "ㅒ", "ㅔ", "ㅏ", "ㅊ", "ㅓ", "ㅉ", "ㅛ", "ㅐ", "ㅁ", "ㅂ", "ㄲ","ㄱ","ㄴ","ㄷ","ㄹ","ㅅ","ㅇ","ㅋ","ㅎ","ㅜ","ㅠ","ㅡ","[SPACE]","[BKSP]","[MISS]"]

num_added=tokenizer.add_tokens(new_tokens)
num_added=tokenizer.add_special_tokens({'additional_special_tokens': ["@"]})
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
def _replace_ignore(label_ids, ignore_id=-100, pad_id=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    return [[(tok if tok != ignore_id else pad_id) for tok in seq] for seq in label_ids]

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = _replace_ignore(labels, ignore_id=-100, pad_id=tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    exact = sum(p == l for p, l in zip(decoded_preds, decoded_labels)) / max(1, len(decoded_preds))
    return {"exact_match": exact}

# ==========================
# 7) TrainingArguments & Trainer
# ==========================
optim_choice = "adamw_torch_fused" if use_cuda else "adamw_torch"


'''
일단 임의로 인자 제거함
load_best_model_at_end=True,
    metric_for_best_model="exact_match",
    greater_is_better=True,
'''

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
    logging_steps=100,
    report_to="wandb",
    run_name=RUN_NAME,
    dataloader_pin_memory=use_cuda,
    optim=optim_choice,
    predict_with_generate=True,
    generation_max_length=MAX_TGT_LEN,
    generation_num_beams=4,
    save_total_limit=1,
    **precision_kwargs,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# (1) 모델/토크나이저 저장
trainer.save_model(str(SAVE_DIR))
tokenizer.save_pretrained(str(SAVE_DIR))

# (2) 정규화/토큰 정보 저장
norm_params = {
    "norm_source": norm_source,      # "x_norm" 또는 "raw"
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
