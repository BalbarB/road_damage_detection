import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("evals/Format/Torchscript.csv")
df = pd.read_csv(csv_path)

df["Base"] = df["Models"].str.extract(r'test_model_([mns])')
df["Version"] = df["Models"].apply(lambda x: "v2" if "v2" in x else "v1")

order = ["m", "s", "n"]
df["Base"] = pd.Categorical(df["Base"], categories=order, ordered=True)
df = df.sort_values("Base")

plt.figure()

for version, g in df.groupby("Version"):
    plt.plot(
        g["Inference time (ms/im)"],
        g["metrics/mAP50-95(B)"],
        marker="o",
        label=version
    )

    for _, r in g.iterrows():
        plt.text(r["Inference time (ms/im)"], r["metrics/mAP50-95(B)"], r["Base"])

plt.xlabel("Inference time")
plt.ylabel("mAP50-95")
plt.title("Accuracy vs Inference time")
plt.legend()
plt.grid(True)
output_path = csv_path.with_suffix(".png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot to: {output_path}")

