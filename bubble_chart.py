import matplotlib.pyplot as plt

# Data (Model vs GPU-hours)
models = ["NT", "DNABERT2", "GenaLM", "genomicBERT"]
gpu_hours = [86016, 2688, 10752, 467]

# Bubble sizes scaled
sizes = [h/50 for h in gpu_hours]  # adjust scaling factor if bubbles too big/small
colors = ["#1f77b4", "#ff7f0e",  "#2ca02c", "#9467bd"]
plt.figure(figsize=(10, 9))

# Increase bubble size and visibility
plt.scatter(
    models,
    gpu_hours,
    s=[s * 1.5 for s in sizes],  # enlarge points
    c=colors,
    alpha=0.7,
    edgecolors="k",
    linewidths=1.2
)

# Annotate values above each point
for model, h in zip(models, gpu_hours):
    plt.text(
        model,
        h * 1.05,  # small upward offset
        f"{h:,}",
        ha="center",
        fontsize=14,  # increased text size
        weight="bold"
    )

# Axis and title styling
plt.ylabel("GPU Hours", fontsize=20, labelpad=10)
plt.xlabel("Model", fontsize=20, labelpad=10)
plt.title("GPU Hours vs Model", fontsize=18, weight="bold", pad=15)

plt.xticks(fontsize=18, rotation=15)
plt.yticks(fontsize=18)

plt.yscale("log")  # log scale helps with large differences
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("bubble_chart.png", dpi=300)
