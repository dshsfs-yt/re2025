# model_training.py — tap-logs(JSON per sentence) ➜ seq2seq(KE-T5)
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

#from trainer import MonitoringSeq2SeqTrainer

# ==========================
# 0) 경로 및 기본 설정
# ==========================
JSON_DIR = Path("saved_logs_nokk")  # 문장별 JSON들이 들어있는 폴더
SAVE_DIR = Path("ckpt/ke-t5-small-RnE2025_jamosplit")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "KETI-AIR/ke-t5-small"
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

BATCH_SIZE = 46  # per device
EPOCHS = 10000
RUN_NAME = "ke-t5-small-RnE2025_jamosplit"
STEPS = 500

# -------------------------
# Hangul constants & decomposition maps
# -------------------------
CHOSEONG = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
            "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
# 기본(단일) 중성만 둔다 — 복합 중성은 아래 맵으로 분해하여 사용
JUNGSEONG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ",
            "ㅕ", "ㅖ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"]
# 종성 인덱스 표 (원래 표) — 필요에 따라 참조
JONGSEONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ",
            "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

HANGUL_BASE = 0xAC00
NUM_JUNG = 21
NUM_JONG = 28

# 복합 중성 -> 기본 자모 분해 맵 (요청하신 대로 ㅘ -> ㅗ + ㅏ 등)
JUNG_DECOMPOSE_MAP = {
    "ㅘ": ["ㅗ", "ㅏ"],
    "ㅙ": ["ㅗ", "ㅐ"],
    "ㅚ": ["ㅗ", "ㅣ"],
    "ㅝ": ["ㅜ", "ㅓ"],
    "ㅞ": ["ㅜ", "ㅔ"],
    "ㅟ": ["ㅜ", "ㅣ"],
    "ㅢ": ["ㅡ", "ㅣ"],
    # 단일 중성(예: ㅏ, ㅓ ...)은 그대로 처리 (맵에 없음)
}

# 복합 종성(겹받침)을 각 자음으로 분해하는 맵 — 전 항목 포함
JONG_DECOMPOSE_MAP = {
    "ㄳ": ["ㄱ", "ㅅ"],
    "ㄵ": ["ㄴ", "ㅈ"],
    "ㄶ": ["ㄴ", "ㅎ"],
    "ㄺ": ["ㄹ", "ㄱ"],
    "ㄻ": ["ㄹ", "ㅁ"],
    "ㄼ": ["ㄹ", "ㅂ"],
    "ㄽ": ["ㄹ", "ㅅ"],
    "ㄾ": ["ㄹ", "ㅌ"],
    "ㄿ": ["ㄹ", "ㅍ"],
    "ㅀ": ["ㄹ", "ㅎ"],
    "ㅄ": ["ㅂ", "ㅅ"],
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
    JUNG_FULL = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
                "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
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


def map_tokens_to_extra_ids(text: str, token_map: Dict[str, str]) -> str:
    """
    텍스트 내의 커스텀 토큰을 extra_id로 매핑.
    예: "ㄱㅏ[SPACE]ㄴㅏ" -> "<extra_id_22><extra_id_13><extra_id_33><extra_id_23><extra_id_13>"
    """
    result = []
    i = 0
    while i < len(text):
        # 긴 토큰부터 매칭 시도 (예: [SPACE], [BKSP] 등)
        matched = False
        # 최대 7글자 토큰 ([FLUSH] 등)
        for token_len in range(min(7, len(text) - i), 0, -1):
            substr = text[i:i+token_len]
            if substr in token_map:
                result.append(token_map[substr])
                i += token_len
                matched = True
                break

        if not matched:
            # 매핑되지 않은 문자는 그대로 유지 (숫자, 영문, 특수문자 등)
            result.append(text[i])
            i += 1

    return "".join(result)


os.environ["WANDB_API_KEY"] = open("wandb_api_key.txt").read().strip()
os.environ["WANDB_PROJECT"] = "RnE2025"         # 선택: 프로젝트명
os.environ["WANDB_NAME"] = RUN_NAME       # 선택: 런 이름

# ==========================
# 1) 디바이스/정밀도
# ==========================


def get_device_and_precision():
    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(
        torch.backends, "mps") and torch.backends.mps.is_available()
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
    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0
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


PROMPT_FORMAT = "touchseq: {seq} coordinate: {xy} -> text"


def build_src_from_presses(presses: List[Dict[str, Any]]) -> str:
    toks: List[str] = []
    cors: List[str] = []
    for e in presses:
        ok, ch = _canon_char_from_log(e.get("role"), e.get("label"))
        if not ok:
            continue
        ix, iy = _get_ix_iy(e)
        # toks.append(f"{ix:02d},{iy:02d},{ch}")
        toks.append(ch)
        cors.append(f"{ix:02d},{iy:02d}")
    return PROMPT_FORMAT.format(seq=" ".join(toks), xy=" ".join(cors))


# 2-3) 최종 DF 생성(빈 시퀀스/빈 타깃 제거)
# 주의: TOKEN_TO_EXTRA_ID는 아직 정의되지 않았으므로, 데이터 로드 시점에는 원본 텍스트로 저장
# 매핑은 토크나이저 로드 후에 적용
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
# 주의: 매핑 전 텍스트로 데이터셋을 먼저 생성하고, 나중에 매핑을 적용함
all_ds = Dataset.from_pandas(
    pd.DataFrame({"src": src_texts, "tgt": tgt_texts}), preserve_index=False).shuffle(seed=RANDOM_SEED)

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
# 4.5) 토큰 매핑 (extra_id 재사용)
# ==========================

# 기존 ke-t5-small의 extra_id (100개)를 재사용하여 새로운 토큰으로 매핑
# 이렇게 하면 모델 구조 변경 없이 사전학습된 임베딩을 활용할 수 있음

JAMO_TOKENS = ["ㅆ", "ㅃ", "ㅈ", "ㅕ", "ㅑ", "ㅖ", "ㅣ", "ㄸ", "ㅗ", "ㅌ", "ㅍ", "ㅒ", "ㅔ", "ㅏ", "ㅊ", "ㅓ", "ㅉ", "ㅛ", "ㅐ", "ㅁ",
            "ㅂ", "ㄲ", "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅅ", "ㅇ", "ㅋ", "ㅎ", "ㅜ", "ㅠ", "ㅡ"]

# SPECIAL_TOKENS = ["[SPACE]", "[BKSP]", "[MISS]", "[FLUSH]"]
SPECIAL_TOKENS = ["[SPACE]", "[BKSP]", "[MISS]", "[FLUSH]", "@"]

# 토큰 -> extra_id 매핑 생성
TOKEN_TO_EXTRA_ID = {}
all_custom_tokens = JAMO_TOKENS + SPECIAL_TOKENS

for idx, token in enumerate(all_custom_tokens):
    TOKEN_TO_EXTRA_ID[token] = f"<extra_id_{idx}>"

# 역방향 매핑도 생성 (디코딩 시 사용)
EXTRA_ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_EXTRA_ID.items()}


# 제한된 token만 생성
VALID_TOKEN_ID = [tokenizer.pad_token_id, tokenizer.eos_token_id]

for token in EXTRA_ID_TO_TOKEN.keys():
    token_id = tokenizer.convert_tokens_to_ids(token)
    VALID_TOKEN_ID.append(token_id)

# def restrict_decode_vocab(batch_idx, prefix_beam):
#     return VALID_TOKEN_ID



print(
    f"\n[Tokenizer] Mapped {len(TOKEN_TO_EXTRA_ID)} custom tokens to extra_ids (0-{len(TOKEN_TO_EXTRA_ID)-1})")
print(f"  - Jamos: {len(JAMO_TOKENS)} tokens")
print(f"  - Special tokens: {len(SPECIAL_TOKENS)} tokens")
print(f"  - Remaining extra_ids: {100 - len(TOKEN_TO_EXTRA_ID)}")

# 이제 매핑 정의가 완료되었으므로, src/tgt 텍스트에 매핑 적용
print(f"\n[Mapping] Applying token mapping to {len(src_texts)} samples...")
src_texts_mapped = [map_tokens_to_extra_ids(
    s, TOKEN_TO_EXTRA_ID) for s in src_texts]
tgt_texts_mapped = [map_tokens_to_extra_ids(
    t, TOKEN_TO_EXTRA_ID) for t in tgt_texts]

# 매핑 적용 전후 비교 (첫 번째 샘플 예시)
if len(src_texts) > 0:
    print(f"\n[Example] First sample mapping:")
    print(f"  Before (src): {src_texts[0][:100]}...")
    print(f"  After (src):  {src_texts_mapped[0][:100]}...")
    print(f"  Before (tgt): {tgt_texts[0][:50]}...")
    print(f"  After (tgt):  {tgt_texts_mapped[0][:100]}...")

# 매핑된 텍스트로 데이터셋 재생성
print(f"\n[Dataset] Recreating dataset with mapped tokens...")
all_ds_mapped = Dataset.from_pandas(
    pd.DataFrame({"src": src_texts_mapped, "tgt": tgt_texts_mapped}),
    preserve_index=False
).shuffle(seed=RANDOM_SEED)

n_mapped = len(all_ds_mapped)
split_mapped = int(n_mapped * 0.95) if n_mapped > 20 else max(1, n_mapped - 1)
raw_ds = DatasetDict({
    "train": all_ds_mapped.select(range(split_mapped)),
    "validation": all_ds_mapped.select(range(split_mapped, n_mapped)),
})
print(f"  Updated raw_ds with mapped tokens: {raw_ds}")


try:
    model.to(device)
except Exception:
    pass

MAX_SRC_LEN = 512
MAX_TGT_LEN = 256


def tokenize_function(batch: Dict[str, List[str]]) -> Dict[str, Any]:
    enc = tokenizer(batch["src"], max_length=MAX_SRC_LEN, truncation=True)
    lab = tokenizer(text_target=batch["tgt"],
                    max_length=MAX_TGT_LEN, truncation=True)
    enc["labels"] = lab["input_ids"]
    return enc


def reverse_map_extra_ids(text: str, extra_id_map: Dict[str, str]) -> str:
    """
    extra_id를 원래 토큰으로 역매핑.
    예: "<extra_id_22><extra_id_13>" -> "ㄱㅏ"
    """
    result = text
    for extra_id, token in extra_id_map.items():
        result = result.replace(extra_id, token)
    return result


tokenized = raw_ds.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_ds["train"].column_names,
)

# ==========================
# 토큰화된 데이터 샘플 출력
# ==========================
print("\n" + "=" * 80)
print("토큰화된 데이터 샘플 확인")
print("=" * 80)

# 학습 데이터에서 첫 3개 샘플 출력
num_samples_to_show = min(3, len(tokenized["train"]))

# 매핑 전 원본 데이터도 가져오기
for i in range(num_samples_to_show):
    sample = tokenized["train"][i]

    # 매핑 전 원본 텍스트 (raw_ds에서 가져오기)
    original_src = raw_ds["train"][i]["src"]
    original_tgt = raw_ds["train"][i]["tgt"]

    # 토큰화된 데이터
    input_ids = sample["input_ids"]
    labels = sample["labels"]

    # 디코딩 (special tokens 포함)
    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
    decoded_label = tokenizer.decode(
        [tok if tok != -100 else tokenizer.pad_token_id for tok in labels],
        skip_special_tokens=False
    )
    
    # extra_id를 원래 토큰으로 역매핑
    decoded_input_remapped = reverse_map_extra_ids(
        decoded_input, EXTRA_ID_TO_TOKEN)
    decoded_label_remapped = reverse_map_extra_ids(
        decoded_label, EXTRA_ID_TO_TOKEN)

    print(f"\n[샘플 {i+1}]")
    print(f"  Input IDs 길이: {len(input_ids)}")
    print(f"  Label IDs 길이: {len(labels)}")

    print(f"\n매핑 전 원본 Src:")
    print(
        f"    {original_src[:200]}{'...' if len(original_src) > 200 else ''}")

    print(f"\n매핑 전 원본 Tgt:")
    print(
        f"    {original_tgt[:100]}{'...' if len(original_tgt) > 100 else ''}")

    print(f"\n매핑 후 Input (extra_id):")
    print(
        f"    {decoded_input[:200]}{'...' if len(decoded_input) > 200 else ''}")

    print(f"\n역매핑된 Input (자모 복원):")
    print(
        f"    {decoded_input_remapped[:200]}{'...' if len(decoded_input_remapped) > 200 else ''}")

    print(f"\n매핑 후 Label (extra_id):")
    print(
        f"    {decoded_label[:100]}{'...' if len(decoded_label) > 100 else ''}")

    print(f"\n역매핑된 Label (자모 복원):")
    print(
        f"    {decoded_label_remapped[:100]}{'...' if len(decoded_label_remapped) > 100 else ''}")

    print(f"\nToken IDs (처음 30개):")
    print(f"    input_ids: {input_ids[:30]}")
    print(f"    labels: {labels[:30]}")

    # <unk> 토큰이 있는지 확인
    unk_count_input = decoded_input.count('<unk>')
    unk_count_label = decoded_label.count('<unk>')
    if unk_count_input > 0 or unk_count_label > 0:
        print(f"\n<unk> 토큰 발견!")
        print(f"    Input에 <unk> 개수: {unk_count_input}")
        print(f"    Label에 <unk> 개수: {unk_count_label}")

        # 매핑 적용 후 텍스트 확인
        mapped_src = src_texts_mapped[i]
        mapped_tgt = tgt_texts_mapped[i]

        print(f"\n    매핑 적용 후 Src (처음 200자):")
        print(f"      {mapped_src[:200]}")

        print(f"\n    매핑 적용 후 Tgt (처음 100자):")
        print(f"      {mapped_tgt[:100]}")

        # 어떤 문자가 UNK로 변환되었는지 확인
        if unk_count_input > 0:
            print(f"\n    원본 src에서 매핑되지 않은 문자들:")
            unique_chars = set(original_src)
            unmapped_chars = []
            for char in unique_chars:
                if char not in TOKEN_TO_EXTRA_ID:
                    unmapped_chars.append(char)
            if unmapped_chars:
                print(f"      매핑 안된 문자: {unmapped_chars[:20]}")  # 처음 20개만

        if unk_count_label > 0:
            print(f"\n    원본 tgt에서 매핑되지 않은 문자들:")
            unique_chars = set(original_tgt)
            unmapped_chars = []
            for char in unique_chars:
                if char not in TOKEN_TO_EXTRA_ID:
                    unmapped_chars.append(char)
            if unmapped_chars:
                print(f"      매핑 안된 문자: {unmapped_chars[:20]}")

print("\n" + "=" * 80)


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

def remove_special_tokens(label_ids: list):
    special_token_ids = [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]
    return [[int(tok) for tok in seq if int(tok) not in special_token_ids] for seq in label_ids]

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(remove_special_tokens(preds), skip_special_tokens=False)

    labels = _replace_ignore(labels, ignore_id=-100,
                            pad_id=tokenizer.pad_token_id)
    labels = remove_special_tokens(labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

    # extra_id를 원래 토큰으로 역매핑
    decoded_preds = [reverse_map_extra_ids(
        p.strip(), EXTRA_ID_TO_TOKEN) for p in decoded_preds]
    decoded_labels = [reverse_map_extra_ids(
        l.strip(), EXTRA_ID_TO_TOKEN) for l in decoded_labels]

    print("pred:", decoded_preds[:3])
    print("label:", decoded_labels[:3])
    
    exact = sum(p == l for p, l in zip(decoded_preds,
                decoded_labels)) / max(1, len(decoded_preds))
    
    return {"exact_match": exact}


# ==========================
# 7) TrainingArguments & Trainer
# ==========================
optim_choice = "adamw_torch_fused" if use_cuda else "adamw_torch"


args = Seq2SeqTrainingArguments(
    output_dir=str(SAVE_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=STEPS,
    logging_strategy="steps",
    logging_steps=50,
    report_to="wandb",
    run_name=RUN_NAME,
    dataloader_pin_memory=use_cuda,
    optim=optim_choice,
    predict_with_generate=True,
    generation_max_length=MAX_TGT_LEN,
    generation_num_beams=4,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="exact_match",
    do_eval=True, 
    greater_is_better=True,
    learning_rate=5e-4,
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
