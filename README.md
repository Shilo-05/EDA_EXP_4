# Experiment 4: Titanic Survival Analysis using Univariate Analysis

```text
**Name:** Oswald Shilo  
**Reg.No:** 212223040139  
```

## Aim
To perform univariate analysis on the Titanic dataset to understand the distribution and characteristics of individual variables (such as Age, Sex, Pclass, Fare, and Survived) and draw insights about passenger demographics and survival patterns.


## Algorithm / Procedure

1. **Import Libraries**  
   Load Pandas, NumPy, Matplotlib, and Seaborn.

2. **Load the Dataset**  
   Use Seaborn’s built-in dataset loader:
   ```python
   df = sns.load_dataset("titanic")
   ```

3. **Data Inspection**

   * Display first 5 rows using `.head()`
   * Show dataset structure using `.info()`
   * Summary statistics using `.describe()`
   * Identify missing values using `.isnull().sum()`

4. **Univariate Analysis**

   **Categorical Variables:**
   Survived, Sex, Pclass, Embarked
   Use `value_counts()` and percentage calculations.

   **Numerical Variables:**
   Age, Fare
   Use histograms, boxplots, and compute Mean, Median, Skewness.

5. **Interpretation**
   Analyze observed trends such as passenger distribution, survival ratio, and skewness of fare/age.

---

## Program (Python)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
df = sns.load_dataset("titanic")

# 2. Data Inspection
print("--- First 5 Rows ---")
print(df.head())
print("\n--- Dataset Shape ---")
print(df.shape)
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Dataset Info ---")
df.info()
print("\n--- Statistical Description ---")
print(df.describe())

# 3. Categorical Analysis
print("\n--- Gender Distribution ---")
print(df['sex'].value_counts())

print("\n--- Survival Statistics ---")
survived_percentage = df['survived'].value_counts(normalize=True) * 100
dead = round(survived_percentage[0], 3)
alive = round(survived_percentage[1], 3)
print(f"Alive : {alive}%")
print(f"Dead  : {dead}%")

print("\n--- Passenger Class Distribution ---")
print(df['pclass'].value_counts())

print("\n--- Embarkation Point Distribution ---")
print(df['embarked'].value_counts())

print("\n--- Deck Distribution ---")
print(df['deck'].value_counts(sort=True))

# 4. Numerical Analysis: AGE
plt.figure(figsize=(8, 5))
sns.histplot(df['age'].dropna(), bins=30, kde=True, color='black')
plt.title("Distribution of Age")
plt.show()

print("\n--- Age Statistics ---")
print(f"Mean   : {df['age'].mean():.2f}")
print(f"Median : {df['age'].median():.2f}")
print(f"Range  : {df['age'].max() - df['age'].min()}")

# 5. Numerical Analysis: FARE
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['fare'])
plt.grid(True)
plt.title("Boxplot of Fare")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['fare'], bins=30, kde=True, color='blue')
plt.title("Distribution of Fare")
plt.show()

print("\n--- Fare Statistics ---")
Mean = df['fare'].mean()
Median = df['fare'].median()
print(f"Mean   : {Mean:.2f}")
print(f"Median : {Median:.2f}")

if Mean > Median:
    print("Skewness: The Fare distribution is Positively Skewed (Right Skewed).")
else:
    print("Skewness: The Fare distribution is Negatively Skewed (Left Skewed).")
```

---

## Output

**Console Output (Snippet):**

```
--- Survival Statistics ---
Alive : 38.384%
Dead  : 61.616%

--- Age Statistics ---
Mean   : 29.70
Median : 28.00
Range  : 79.58

--- Fare Statistics ---
Mean   : 32.20
Median : 14.45
Skewness: The Fare distribution is Positively Skewed (Right Skewed).
```

**Visualizations Produced:**

## Output (Visualizations)

### 1. Dataset Head & Structure
![Screenshot 1](https://github.com/user-attachments/assets/bc4d0962-12b7-4bc8-a591-704dedae1d8b)

### 2. Age Distribution (Histogram)
![Screenshot 2](https://github.com/user-attachments/assets/fbe5f85f-8230-445e-a4a8-1a9c4ba4e7b1)

### 3. Fare Boxplot
![Screenshot 3](https://github.com/user-attachments/assets/f202057b-05b5-4738-b21c-36996772b1c0)

### 4. Fare Distribution (Histogram)
![Screenshot 4](https://github.com/user-attachments/assets/22a6b866-100a-41be-a199-de49c3c40807)

### 5. Gender, Pclass, Embarked Distribution
![Screenshot 5](https://github.com/user-attachments/assets/bb7665b6-81d6-4d24-a47f-d572dc7c1414)

### 6. Survival Statistics Visual
![Screenshot 6](https://github.com/user-attachments/assets/9193a6a6-4def-42a2-b14c-5a8da7a6eeb4)


## Inference

* **Demographics:**
  Majority of passengers were male and belonged to 3rd class, reflecting socio-economic diversity.

* **Survival Rate:**
  Only **38% survived**, indicating significant imbalance in survival outcomes.

* **Age Distribution:**
  Age shows a near-normal distribution with slight right skew; most passengers were between **20–40 years**.

* **Fare Distribution:**
  Strong positive skewness with several high-value outliers—indicating presence of wealthier passengers in upper classes.

---

## Result

The univariate analysis on the Titanic dataset was completed successfully.
It provides a strong foundational understanding of the dataset’s structure, demographic patterns, and variable distributions—serving as a solid starting point for deeper multivariate or predictive modeling.




