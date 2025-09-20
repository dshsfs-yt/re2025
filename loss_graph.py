# plot_loss_relpath.py
import json
from pathlib import Path
import matplotlib.pyplot as plt

# ▶ 이 상대경로는 "이 파일이 있는 폴더" 기준입니다.
REL_JSON = Path("ckpt")/"ke-t5-small-touch-only(korean)" /"checkpoint-2676" /"trainer_state.json"

def main():
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / REL_JSON

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    logs = data.get("log_history", [])
    pairs = [
        (rec.get("step", i + 1), float(rec["loss"]))
        for i, rec in enumerate(logs)
        if "loss" in rec
    ]
    if not pairs:
        print("log_history에 'loss'가 없습니다.")
        return

    # step 기준 정렬
    pairs.sort(key=lambda t: t[0])
    xs, ys = zip(*pairs)

    plt.figure()
    plt.plot(xs, ys)  # 색상 지정 X(기본값)
    plt.xlabel("step" if any(("step" in r and "loss" in r) for r in logs) else "index")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
