#!/usr/bin/env python3
# build_touch_csv.py
"""
VSCode에서 바로 실행(무인자 시 GUI) + CLI 지원.
입력(ZIP/폴더/JSON) → 4열 touch_csv 생성:
[ref_char, first_frame_touch_x, first_frame_touch_y, prev_shift]

규칙
- ref_char는 이벤트의 label 문자열 (없거나 빈 문자열이면 스킵)
- prev_shift는 "해당 label 자체가 Shift가 필요한 문자"이면 True
  * 라틴 대문자(A-Z)
  * US QWERTY에서 Shift로 입력하는 기호: !@#$%^&*()_+{}|:"<>?~
  * (옵션) 한글 2벌식에서 Shift로 나오는 겹자모: ㅃ,ㅉ,ㄸ,ㄲ,ㅆ, ㅒ,ㅖ
- SPACE/BKSP/MISS도 label이 있으면 그대로 ref_char로 저장하되, prev_shift는 False
- 기타 역할은 제외
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import pandas as pd

CORE_COLS = ["ref_char", "first_frame_touch_x", "first_frame_touch_y", "prev_shift"]
ALLOWED_ROLES = {"CHAR", "SPACE", "BKSP", "MISS"}

SHIFTED_ASCII = set('!@#$%^&*()_+{}|:"<>?~')
SHIFTED_KO_JAMO = set('ㅃㅉㄸㄲㅆㅒㅖ')

def _normalize_xy(e: Dict[str, Any]) -> Tuple[float, float]:
    x = e.get("x_norm")
    y = e.get("y_norm")
    if x is None or y is None:
        xr = float(e.get("x", 0.0))
        yr = float(e.get("y", 0.0))
        w  = float(e.get("w", 1.0))
        h  = float(e.get("h", 1.0))
        x = (xr / w) if w else 0.0
        y = (yr / h) if h else 0.0
    x = max(0.0, min(1.0, float(x)))
    y = max(0.0, min(1.0, float(y)))
    return x, y

def _label_or_none(e: Dict[str, Any]) -> str:
    val = e.get("label")
    if isinstance(val, str):
        val = val.strip()
        if val:
            return val
    return None

def _is_shift_required(label: str) -> bool:
    # 멀티토큰(<SPACE> 등)은 False
    if len(label) != 1:
        # 라틴 대문자 단어 전체가 올 수 있으므로 한 글자씩 검사
        if label.isalpha() and label.upper() == label and label.lower() != label:
            return True
        return False
    ch = label
    # 라틴 대문자
    if 'A' <= ch <= 'Z':
        return True
    # 시프트 기호
    if ch in SHIFTED_ASCII:
        return True
    # 한글 겹자모 (2벌식 Shift)
    if ch in SHIFTED_KO_JAMO:
        return True
    return False

def _json_obj_to_rows(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sessions = obj.get("logs", [])
    for ses in sessions:
        for e in ses.get("logs", []):
            role = e.get("role")
            if role not in ALLOWED_ROLES:
                continue
            ref_char = _label_or_none(e)
            if ref_char is None:
                continue
            x, y = _normalize_xy(e)
            rows.append({
                "ref_char": ref_char,
                "first_frame_touch_x": x,
                "first_frame_touch_y": y,
                "prev_shift": _is_shift_required(ref_char),
            })
    return rows

def _read_json_files_from_zip(zip_path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".json"):
                continue
            with zf.open(info, "r") as f:
                try:
                    yield info.filename, json.loads(f.read().decode("utf-8"))
                except Exception:
                    continue

def _read_json_files_from_dir(dir_path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for p in dir_path.rglob("*.json"):
        try:
            yield str(p), json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

def _read_single_json(json_path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    yield str(json_path), json.loads(json_path.read_text(encoding="utf-8"))

def convert_logs(input_path: str, output_csv: str) -> int:
    inp = Path(input_path)
    if inp.is_file() and inp.suffix.lower() == ".zip":
        it = _read_json_files_from_zip(inp)
    elif inp.is_dir():
        it = _read_json_files_from_dir(inp)
    elif inp.is_file() and inp.suffix.lower() == ".json":
        it = _read_single_json(inp)
    else:
        raise ValueError("입력은 ZIP, 폴더, 또는 JSON 파일이어야 합니다.")

    all_rows: List[Dict[str, Any]] = []
    for _, obj in it:
        all_rows.extend(_json_obj_to_rows(obj))

    if not all_rows:
        raise RuntimeError("유효한 터치 이벤트가 없어 CSV를 생성하지 않았습니다.")

    df = pd.DataFrame(all_rows, columns=CORE_COLS)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return len(df)

def _run_cli() -> bool:
    import sys
    if len(sys.argv) <= 1:
        return False
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="ZIP/폴더/JSON 경로")
    ap.add_argument("-o", "--output", type=str, required=True, help="출력 CSV 경로")
    args = ap.parse_args()

    rows = convert_logs(args.input, args.output)
    print(f"[OK] saved: {Path(args.output).resolve()} | rows={rows}")
    return True

def _run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk(); root.withdraw()
    input_path = filedialog.askopenfilename(
        title="로그 ZIP/JSON 선택 (취소하면 폴더 선택)",
        filetypes=[("Logs", "*.zip *.json"), ("Zip", "*.zip"), ("JSON", "*.json"), ("All", "*.*")]
    )
    if not input_path:
        input_path = filedialog.askdirectory(title="로그 폴더 선택")
        if not input_path:
            messagebox.showwarning("취소됨", "입력 경로가 선택되지 않았습니다."); return

    output_csv = filedialog.asksaveasfilename(
        title="CSV 저장 위치 선택",
        defaultextension=".csv",
        initialfile="touch_data.csv",
        filetypes=[("CSV", "*.csv")]
    )
    if not output_csv:
        messagebox.showwarning("취소됨", "출력 경로가 선택되지 않았습니다."); return

    try:
        rows = convert_logs(input_path, output_csv)
        messagebox.showinfo("완료", f"저장됨:\n{output_csv}\nrows={rows}")
    except Exception as e:
        messagebox.showerror("에러", f"{type(e).__name__}: {e}")

def main():
    did_cli = _run_cli()
    if not did_cli:
        _run_gui()

if __name__ == "__main__":
    main()
