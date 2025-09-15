# model_validation.py — saved_logs 기반 빔서치 정확도(@k) 평가
# 요구사항:
#  1) saved_logs 폴더에서 JSON을 무작위로 골라
#  2) 사용자 입력 좌표(CHAR/SPACE/MISS/BKSP)를 시퀀스로 구성하고
#  3) 모델로 빔서치를 수행
#  4) target 문장이 빔 후보 내에 있으면 정답으로 집계하여 정확도 산출
#
# 실행 예:
#   python model_validation.py \
#       --model_dir ckpt/ke-t5-small-coord \
#       --json_dir saved_logs \
#       --num_samples 200 \
#       --beam_width 5 \
#       --max_new_tokens 128 \
#       --show 5
#
# 주의: 학습 시 저장한 normalization.json 내 필드(norm_source, x/y_min/max,
#      prompt_format, input_token_pattern, special_tokens 등)를 그대로 재사용합니다.

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ------------------------------
# 0) 유틸
# ------------------------------
def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def _to01(v: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.5
    return _clip01((float(v) - vmin) / (vmax - vmin))

def _q99(v01: float) -> int:
    return int(round(_clip01(v01) * 99))

def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------
# 1) 체크포인트 로드
# ------------------------------
def load_checkpoint(model_dir: Path):
    norm_path = model_dir / "normalization.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"normalization.json not found in {model_dir.resolve()}")
    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    mdl.to(dev).eval()
    return mdl, tok, norm, dev


# ------------------------------
# 2) JSON 파서 (saved_logs/*.json)
#    기대 구조:
#    {
#      "target_sentence": "...",
#      "logs": [
#         {"logs":[
#             {"role":"CHAR","label":"가","x_norm":0.53,"y_norm":0.72}  # 또는 raw x,y
#             {"role":"SPACE"}, {"role":"MISS", ...}, {"role":"BKSP", ...}, ...
#         ]}
#      ]
#    }
# ------------------------------
ROLES_USE = {"CHAR", "SPACE", "MISS", "BKSP"}

def _canon_char_from_log(role: str, label: str, space_label: str, miss_label: str, bksp_label: str) -> Tuple[bool, str]:
    role = (role or "").upper()
    if role == "CHAR":
        ch = (label or "")
        return True, (ch[:1] if ch else "")
    if role == "SPACE":
        return True, space_label
    if role == "MISS":
        return True, miss_label
    if role == "BKSP":
        return True, bksp_label
    return False, ""

def _iter_json_files(json_dir: Path):
    for p in sorted(json_dir.glob("*.json")):
        yield p

def _load_one_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------
# 3) 입력 시퀀스 구축 (학습과 동일 규칙)
# ------------------------------
def _get_ix_iy(e: Dict[str, Any], norm: Dict[str, Any]) -> Tuple[int, int]:
    norm_source = norm.get("norm_source", "x_norm")
    if norm_source == "x_norm":
        x01 = float(e.get("x_norm", 0.5))
        y01 = float(e.get("y_norm", 0.5))
        return _q99(x01), _q99(y01)
    else:
        x = float(e.get("x", 0.0))
        y = float(e.get("y", 0.0))
        x_min, x_max = float(norm.get("x_min", 0.0)), float(norm.get("x_max", 1.0))
        y_min, y_max = float(norm.get("y_min", 0.0)), float(norm.get("y_max", 1.0))
        return _q99(_to01(x, x_min, x_max)), _q99(_to01(y, y_min, y_max))

def build_src_from_presses(presses: List[Dict[str, Any]], norm: Dict[str, Any]) -> str:
    """학습 시 저장된 prompt_format / input_token_pattern을 그대로 재사용."""
    toks: List[str] = []
    patt = norm.get("input_token_pattern", "{ch}@{ix:02d},{iy:02d}")
    space_label = norm.get("space_label", "<SPACE>")
    miss_label  = norm.get("miss_label", "<MISS>")
    bksp_label  = norm.get("bksp_label", "<BKSP>")
    for e in presses:
        ok, ch = _canon_char_from_log(e.get("role"), e.get("label"), space_label, miss_label, bksp_label)
        if not ok:
            continue
        ix, iy = _get_ix_iy(e, norm)
        toks.append(patt.format(ch=ch, ix=ix, iy=iy))
    prompt_format = norm.get("prompt_format", "touchseq: {seq} -> text")
    return prompt_format.format(seq=" ".join(toks))


# ------------------------------
# 4) 빔서치 평가
# ------------------------------
@torch.inference_mode()
def beam_candidates(model, tok, device: str, srcs: List[str], max_new_tokens: int, beam_width: int) -> List[List[str]]:
    """각 입력마다 빔서치 후보 문자열 리스트를 반환."""
    outs: List[List[str]] = []
    bs = 8  # 적당한 배치
    for i in range(0, len(srcs), bs):
        sl = srcs[i:i+bs]
        enc = tok(sl, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(
            **enc,
            do_sample=False,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            max_new_tokens=max_new_tokens,
        )
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        # num_return_sequences 기준으로 묶어주기
        for j in range(0, len(decoded), beam_width):
            group = [s.strip() for s in decoded[j:j+beam_width]]
            outs.append(group)
    return outs


def evaluate(json_dir: Path, model_dir: Path, num_samples: int, beam_width: int, max_new_tokens: int, seed: int, show: int):
    set_seed(seed)
    model, tok, norm, device = load_checkpoint(model_dir)

    files = list(_iter_json_files(json_dir))
    if len(files) == 0:
        raise FileNotFoundError(f"No JSON files in: {json_dir.resolve()}")

    # 무작위 샘플링
    chosen = files if num_samples <= 0 or num_samples >= len(files) else random.sample(files, num_samples)

    src_list: List[str] = []
    tgt_list: List[str] = []
    keep_idx: List[int] = []

    for idx, fp in enumerate(chosen):
        obj = _load_one_json(fp)
        tgt = (obj.get("target_sentence") or "").strip()
        logs_blocks = obj.get("logs") or []
        if not tgt or not logs_blocks:
            continue
        # 학습과 동일하게 첫 블록만 사용(필요시 여기서 확장 가능)
        presses = (logs_blocks[0] or {}).get("logs") or []
        if not presses:
            continue
        # 좌표 입력 이벤트들로 시퀀스 구성
        src = build_src_from_presses(presses, norm)
        if src.strip():
            src_list.append(src)
            tgt_list.append(tgt)
            keep_idx.append(idx)

    if not src_list:
        raise ValueError("유효한 평가 샘플이 없습니다. JSON 구조/필드를 확인하세요.")

    # 빔 후보 생성
    all_beams = beam_candidates(model, tok, device, src_list, max_new_tokens=max_new_tokens, beam_width=beam_width)

    # 정답 판정: target ∈ beam_set ?
    correct = 0
    hits_at = []  # (hit_rank or None)
    for tgt, beams in zip(tgt_list, all_beams):
        try:
            r = next((k for k, s in enumerate(beams, start=1) if s == tgt), None)
        except Exception:
            r = None
        if r is not None:
            correct += 1
        hits_at.append(r)

    total = len(tgt_list)
    acc = correct / max(1, total)

    print("===== Beam Search Evaluation =====")
    print(f"Model dir     : {model_dir}")
    print(f"JSON dir      : {json_dir}")
    print(f"Device        : {device}")
    print(f"Samples       : {total}")
    print(f"Beam width    : {beam_width}")
    print(f"Max new toks  : {max_new_tokens}")
    print(f"Accuracy@{beam_width}: {correct}/{total} = {acc:.2%}")

    # 샘플 출력
    if show > 0:
        print("\n--- Examples ---")
        for i in range(min(show, total)):
            beams = all_beams[i]
            tgt   = tgt_list[i]
            print(f"[{i:02d}] TARGET: {tgt}")
            for k, s in enumerate(beams, start=1):
                mark = "✅" if s == tgt else "  "
                print(f"   {mark} {k:02d}: {s}")
            print()

    # 랭크 통계(선택)
    hit_ranks = [r for r in hits_at if r is not None]
    if hit_ranks:
        import statistics as st
        print(f"Hit rank (mean/median): {st.mean(hit_ranks):.2f} / {st.median(hit_ranks):.2f}")
    else:
        print("No hits in beams.")


# ------------------------------
# 5) 엔트리 포인트
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="ckpt/ke-t5-small-coord")
    ap.add_argument("--json_dir",  type=str, default="saved_logs")
    ap.add_argument("--num_samples", type=int, default=34, help="0이면 폴더의 모든 JSON 사용")
    ap.add_argument("--beam_width", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=1, help="표시할 예시 수")
    args = ap.parse_args()

    evaluate(
        json_dir=Path(args.json_dir),
        model_dir=Path(args.model_dir),
        num_samples=args.num_samples,
        beam_width=args.beam_width,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        show=args.show,
    )

if __name__ == "__main__":
    main()
