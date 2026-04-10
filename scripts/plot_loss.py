import re
import matplotlib.pyplot as plt

log_path = "train.log"

epochs, losses, coarse_losses, fine_losses = [], [], [], []

pattern = re.compile(
    r"Epoch (\d+) done \| Loss ([\d.]+) \(coarse ([\d.]+), fine ([\d.]+)\)"
)

with open(log_path, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            coarse_losses.append(float(m.group(3)))
            fine_losses.append(float(m.group(4)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, values, title, color in zip(
    axes,
    [losses, coarse_losses, fine_losses],
    ["Total Loss", "Coarse Loss", "Fine Loss"],
    ["steelblue", "darkorange", "seagreen"],
):
    ax.plot(epochs, values, color=color, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)

fig.suptitle("Training Loss vs Epoch", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()
print("Saved to loss_curve.png")
