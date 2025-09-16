#!/usr/bin/env python3
# build_touch_csv.py
"""
VSCode에서 바로 실행 가능(무인자 실행 시 GUI로 경로 선택) + CLI 지원.
문장별 터치 로그(JSON)들이 들어있는 ZIP/폴더/단일 JSON을 읽어
학습용 touch_csv(ref_char, first_frame_touch_x, first_frame_touch_y)로 변환.

CLI 예시:
  python build_touch_csv.py "D:/logs/typing_logs.zip" -o "D:/out/touch_data.csv" --token-style angle
  python build_touch_csv.py "D:/logs/json_folder" -o "D:/out/touch_data.csv" --no-meta

VSCode에서 F5로 그냥 실행하면 파일 선택 창이 떠요.
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import pandas as pd

# ---------- 변환 로직 ----------

CORE_COLS = ["ref_char", "first_frame_touch_x", "first_frame_touch_y"]
META_COLS = ["role", "key", "label", "timestamp_ms", "source_file", "session_target", "session_index"]

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

def _role_to_ref_char(e: Dict[str, Any], token_style: str) -> str:
    role = e.get("role")
    if role == "CHAR":
        val = (e.get("key") or e.get("label") or "").strip()
        return val or None
    mapping_plain = {"SPACE": "SPACE", "BKSP": "BKSP", "MISS": "MISS"}
    mapping_angle = {"SPACE": "<SPACE>", "BKSP": "<BKSP>", "MISS": "<MISS>"}
    mapping = mapping_plain if token_style == "plain" else mapping_angle
    if role in mapping:
        return mapping[role]
    return None

def _json_obj_to_rows(obj: Dict[str, Any], source_file: str, token_style: str) -> List[Dict[str, Any]]:
    rows = []
    sessions = obj.get("logs", [])
    for sidx, ses in enumerate(sessions):
        target_sent = ses.get("target", "")
        for e in ses.get("logs", []):
            ref_char = _role_to_ref_char(e, token_style)
            if not ref_char:
                continue
            x, y = _normalize_xy(e)
            row = {
                "ref_char": ref_char,
                "first_frame_touch_x": x,
                "first_frame_touch_y": y,
                "role": e.get("role"),
                "key": e.get("key"),
                "label": e.get("label"),
                "timestamp_ms": e.get("t", None),
                "source_file": source_file,
                "session_target": target_sent,
                "session_index": sidx,
            }
            rows.append(row)
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

def convert_logs(input_path: str, output_csv: str,
                 token_style: str = "plain",
                 include_meta: bool = True) -> int:
    """
    입력 경로(Zip/폴더/단일 JSON)를 읽어 touch_csv 생성.
    return: 저장된 행(row) 수
    """
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
    for src, obj in it:
        all_rows.extend(_json_obj_to_rows(obj, source_file=src, token_style=token_style))

    if not all_rows:
        raise RuntimeError("유효한 터치 이벤트가 없어 CSV를 생성하지 않았습니다.")

    cols = CORE_COLS + (META_COLS if include_meta else [])
    df = pd.DataFrame(all_rows, columns=cols)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return len(df)

# ---------- CLI & VSCode GUI ----------

def _run_cli() -> bool:
    """CLI 인자가 있으면 처리하고 True를, 없으면 False를 반환."""
    import sys
    if len(sys.argv) <= 1:
        return False
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="ZIP/폴더/JSON 경로")
    ap.add_argument("-o", "--output", type=str, required=True, help="출력 CSV 경로")
    ap.add_argument("--token-style", choices=["plain","angle"], default="plain",
                    help="특수키 표기 방식: plain=SPACE/BKSP/MISS, angle=<SPACE>/<BKSP>/<MISS>")
    ap.add_argument("--no-meta", action="store_true", help="메타 컬럼 제외(핵심 3컬럼만 저장)")
    args = ap.parse_args()

    rows = convert_logs(args.input, args.output,
                        token_style=args.token_style,
                        include_meta=not args.no_meta)
    print(f"[OK] saved: {Path(args.output).resolve()} | rows={rows}")
    return True

def _run_gui():
    """VSCode에서 F5로 그냥 실행 시 GUI로 경로 선택."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog

    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo("Touch CSV Builder", "ZIP/JSON 파일을 선택하거나, 취소 후 폴더를 선택하세요.")

    input_path = filedialog.askopenfilename(
        title="로그 ZIP/JSON 선택 (취소하면 폴더 선택으로 진행)",
        filetypes=[("Logs", "*.zip *.json"), ("Zip", "*.zip"), ("JSON", "*.json"), ("All", "*.*")]
    )
    if not input_path:
        input_path = filedialog.askdirectory(title="로그 폴더 선택")
        if not input_path:
            messagebox.showwarning("취소됨", "입력 경로가 선택되지 않았습니다.")
            return

    # 옵션: token style
    token_style = simpledialog.askstring(
        "토큰 표기 방식",
        "특수키 표기 방식 입력 (plain 또는 angle)\n- plain: SPACE/BKSP/MISS\n- angle: <SPACE>/<BKSP>/<MISS>",
        initialvalue="plain"
    ) or "plain"
    token_style = token_style.strip().lower()
    if token_style not in ("plain", "angle"):
        token_style = "plain"

    # 옵션: include meta
    include_meta = messagebox.askyesno("메타 컬럼 포함", "메타 컬럼(role/key/label/.. )을 포함할까요?")

    default_name = "touch_data.csv"
    output_csv = filedialog.asksaveasfilename(
        title="CSV 저장 위치 선택",
        defaultextension=".csv",
        initialfile=default_name,
        filetypes=[("CSV", "*.csv")]
    )
    if not output_csv:
        messagebox.showwarning("취소됨", "출력 경로가 선택되지 않았습니다.")
        return

    try:
        rows = convert_logs(input_path, output_csv,
                            token_style=token_style,
                            include_meta=include_meta)
        messagebox.showinfo("완료", f"저장됨:\n{output_csv}\nrows={rows}")
    except Exception as e:
        messagebox.showerror("에러", f"{type(e).__name__}: {e}")

def main():
    did_cli = _run_cli()
    if not did_cli:
        _run_gui()

if __name__ == "__main__":
    main()
