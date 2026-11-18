from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Studies that use the same 121-child IEEE DataPort EEG dataset (or the 120-child
# subset used by CTE-Net). Values are as reported in each paper.
studies = [
    ("Hassan 2024\n(frame-level)", 100.00, "leaky"),
    ("Bansal 2025\n(frame-level)", 99.42, "leaky"),
    ("ADHDeepNet 2026\n(claims cross-subject)", 99.17, "claim"),
    ("Spherical 2026\n(window-level)", 98.00, "leaky"),
    ("CTE-Net 2026\n(subject-wise, n=120)", 83.40, "clean"),
    ("CNN 2026\n(subject-wise)", 72.70, "clean"),
    ("SVM 2026\n(subject-wise)", 65.60, "clean"),
]

colors = {"leaky": "#d9534f", "claim": "#f0ad4e", "clean": "#2e8b57"}
labels = [s[0] for s in studies]
values = [s[1] for s in studies]
bar_colors = [colors[s[2]] for s in studies]

fig, ax = plt.subplots(figsize=(11, 6.2))
bars = ax.bar(range(len(studies)), values, color=bar_colors, width=0.68, edgecolor="white", linewidth=1.2)

for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222222")

ax.set_xticks(range(len(studies)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Reported accuracy (%)", fontsize=11)
ax.set_ylim(55, 104)
fig.suptitle("Same 121-child EEG dataset, very different accuracy depending on the split",
             fontsize=13.5, fontweight="bold", y=0.97)
fig.text(0.5, 0.915,
         "Red = epoch/frame-level split; orange = claims subject-wise; green = subject-wise validation",
         ha="center", fontsize=9, color="#444444")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.set_axisbelow(True)

# Annotation showing the leakage gap
ax.set_xlim(-0.5, 7.05)
ax.annotate("", xy=(6.42, 65.6), xytext=(6.42, 100),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.4))
ax.text(6.55, 82, "~34-point\ndrop", fontsize=10, color="#555555", va="center")

fig.text(0.5, 0.01,
         "Different models, same underlying children. The drop tracks the evaluation protocol, not model quality.",
         ha="center", fontsize=9, style="italic", color="#555555")

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
out = Path(__file__).parent / "accuracy_inflation_eeg.png"
plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
