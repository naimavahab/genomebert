import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from math import pi
import numpy as np

# --- Configure global aesthetics ---
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

sns.set_context("paper")
sns.set_style("whitegrid")

# --- Data ---
data = {
    "Model": ["genomicBERT", "GenaLM", "DNABERT2", "NT"],
    "Model_Size_M": [113, 110, 117, 100],
    "Max_Seq_Len": [256, 512, 128, 1000],
    "Tokenizer": ["Unigram", "BPE", "BPE", "k-mer"],
    "Vocab_Size": [4096, 32000, 4096, 4096],
    "Hardware": ["4×A10G", "8–16×A100", "8×2080Ti", "128×A100"],
    "Steps": [53000, 1500000, 500000, None],
}
df = pd.DataFrame(data)

# Approx GPU counts
def estimate_gpu_count(hardware):
    if "128" in hardware: return 128
    if "16" in hardware: return 16
    if "8" in hardware: return 8
    if "4" in hardware: return 4
    return 1
df["GPU_Count"] = df["Hardware"].apply(estimate_gpu_count)

# Colour palette (consistent)
palette = sns.color_palette("colorblind", n_colors=len(df))

# --- 1. Model size comparison ---
plt.figure(figsize=(4.5, 3.2))
sns.barplot(x="Model", y="Model_Size_M", data=df, palette=palette, edgecolor="black")
plt.ylabel("Model Size (Millions)")
plt.title("Model Scale Across Genomic Foundation Models", pad=12)
sns.despine()
plt.tight_layout()
plt.savefig("fig_model_size.png", dpi=300)


# --- 2. Vocabulary size comparison ---
plt.figure(figsize=(4.5, 3.2))
sns.barplot(x="Model", y="Vocab_Size", data=df, palette=palette, edgecolor="black")
plt.yscale("log")
plt.ylabel("Vocab Size (log scale)")
plt.title("Tokenizer Vocabulary Size", pad=12)
sns.despine()
plt.tight_layout()
plt.savefig("fig_vocab_size.png", dpi=300)


# --- 3. Model size vs vocab size ---
plt.figure(figsize=(6, 3.5))
sns.scatterplot(
    data=df,
    x="Model_Size_M",
    y="Vocab_Size",
    hue="Tokenizer",
    style="Tokenizer",
    s=120,
    palette="dark",
)
for i, row in df.iterrows():
    plt.text(row["Model_Size_M"] + 1, row["Vocab_Size"], row["Model"], fontsize=8)
plt.yscale("log")
plt.xlabel("Model Size (Millions)")
plt.ylabel("Vocab Size (log scale)")
plt.title("Tokenisation Trade-off: Model Size vs Vocabulary", pad=10)
sns.despine()
plt.tight_layout()
plt.savefig("fig_model_vocab_relation.png", dpi=300)


# --- 4. Bubble chart: compute efficiency ---
plt.figure(figsize=(5, 3.8))
plt.scatter(df["Model_Size_M"], df["Max_Seq_Len"],
            s=df["GPU_Count"]*20, alpha=0.6,
            c=sns.color_palette("crest", len(df)))
for i, row in df.iterrows():
    plt.text(row["Model_Size_M"], row["Max_Seq_Len"] + 40, row["Model"], ha='center', fontsize=10)
plt.xlabel("Model Size (Millions)")
plt.ylabel("Max Sequence Length")
plt.title("Compute Efficiency: Model Size vs Sequence Length", pad=12)
sns.despine()
plt.tight_layout()
plt.savefig("fig_compute_efficiency.png", dpi=300)

# --- 5. Radar plot (optional multi-feature) ---
features = ["Model_Size_M", "Max_Seq_Len", "Vocab_Size"]
df_norm = df.copy()
df_norm[features] = (df_norm[features] - df_norm[features].min()) / (df_norm[features].max() - df_norm[features].min())
labels = features
num_vars = len(labels)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(5, 5))
for idx, row in df_norm.iterrows():
    values = row[labels].tolist() + row[labels].tolist()[:1]
    plt.polar(angles, values, label=row["Model"], linewidth=2)
plt.xticks(angles[:-1], ["Model Size", "Seq Length", "Vocab Size"], fontsize=11)
plt.title("Normalised Multi-feature Comparison", y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("fig_radar_features.png", dpi=300)

plt.figure(figsize=(7, 4))

sns.scatterplot(
    data=df,
    x="Model_Size_M",
    y="Vocab_Size",
    hue="Tokenizer",
    style="Tokenizer",
    s=150,
    palette="dark",
    edgecolor="k",
    linewidth=1.2
)

# Label the points ABOVE the markers
for i, row in df.iterrows():
    plt.text(
        row["Model_Size_M"],                 # keep centered horizontally
        row["Vocab_Size"] * 1.08,            # move label above the point
        row["Model"],
        fontsize=9,
        weight="bold",
        ha="center",                         # horizontally center align
        va="bottom"                          # vertical alignment just below text baseline
    )

# Axis scaling and labels
plt.yscale("log")
plt.xlabel("Model Size (Millions)", fontsize=14, labelpad=8)
plt.ylabel("Vocab Size (log scale)", fontsize=14, labelpad=8)
plt.title("Tokenisation Trade-off: Model Size vs Vocabulary", fontsize=12, weight="bold", pad=12)

# Tick styling
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Axes styling
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

sns.despine()
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("fig_model_vocab_relation.png", dpi=300, bbox_inches="tight")
