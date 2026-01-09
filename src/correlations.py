import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load data from CSV
df = pd.read_csv("../Data/points.csv")

# Extract X and Y values
x = df["x"].to_numpy(dtype=float)
y = df["y"].to_numpy(dtype=float)

# Calculate Pearson correlation coefficient
r, p_value = pearsonr(x, y)

print(f"Pearson correlation coefficient (r): {r:.6f}")
print(f"P-value: {p_value:.6g}")
print(f"Number of points: {len(x)}")

# Create scatter plot
plt.figure()
plt.scatter(x, y)

# Best-fit line
m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 200)
plt.plot(x_line, m * x_line + b)

plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Scatter plot with best-fit line (r = {r:.3f})")
plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig("../figures/scatter.png", dpi=200)
plt.show()
