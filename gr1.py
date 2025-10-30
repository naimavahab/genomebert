import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data ---
data = {
    "Model": ["genomicBERT", "GenaLM", "DNABERT2", "NT"],
    "Model_Size_M": [113, 110, 117, 100],
    "Max_Seq_Len": [256, 512, 128, 1000],
    "Tokenizer": ["Unigram", "BPE", "BPE", "k-mer"],
    "Vocab_Size": [4096, 32000, 4096, 4096],
    "Hardware": ["4xA10G", "8-16xA100", "8x2080Ti", "128xA100"],
    "Steps": [53000, 1500000, 500000, None],
}

df = pd.DataFrame(data)


from math import pi
import numpy as np

# Normalise numeric values (0–1 range)
features = ["Model_Size_M", "Max_Seq_Len", "Vocab_Size"]
df_norm = df.copy()
df_norm[features] = (df_norm[features] - df_norm[features].min()) / (df_norm[features].max() - df_norm[features].min())

# Radar setup
labels = features
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(7,7))
for _, row in df_norm.iterrows():
    values = row[labels].tolist()
    values += values[:1]
    plt.polar(angles, values, label=row["Model"])

plt.xticks(angles[:-1], labels)
plt.title("Multi-feature Comparison of Genomic Models", y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.savefig('tets.png')