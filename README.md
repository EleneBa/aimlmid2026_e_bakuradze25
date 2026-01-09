# aimlmid2026_e_bakuradze25
# AIML Midterm – Finding the Correlation (e_bakuradze25)

## Task 1
The goal is to find Pearson’s correlation coefficient for the blue points shown in the interactive plot at:

https://max.ge/aiml_midterm/23456_html/e_bakuradze25_23456.html

---

## Data collection
The data was collected manually by hovering over each blue dot and recording its (x, y) coordinates.

File:
- Data/points.csv

---

## Method
Pearson’s correlation coefficient
The value was computed using Python (`scipy.stats.pearsonr`) after loading the data with pandas.

---

## Results
- Number of points: 9
- Pearson correlation coefficient (r): -0.982232
- p-value: 2.41943e-06

This indicates a very strong negative linear relationship.

---

## Visualization
A scatter plot with a best-fit line is provided in:

- figures/scatter.png

---

## Reproducibility
To reproduce:

```bash
pip install numpy scipy matplotlib pandas
cd src
python correlations.py
