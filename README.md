# Gradient Checking in Logistic Regression

## 📌 Description
This project implements **Gradient Checking using Finite Differences** in **Logistic Regression**.  
It provides two implementations:
- **C** for performance and low-level demonstration.
- **Python** for clarity and flexibility.

The goal is to validate analytical gradients against numerical approximations, ensuring correctness in optimization algorithms.

---

## 📂 Files
- `Gradient.py` → Python implementation (interactive, customizable).  
- `gradient_check.c` → C implementation (fixed dataset size, efficient).  

---

## ⚙️ How to Run

### ▶️ Python
1. Run the script:
   ```bash
   python Gradient.py
   ```
2. The program will ask for:
   - Number of examples `m`
   - Number of features `n`
   - Step size `epsilon`
   - Tolerance
   - Initial parameters `theta`

   Press **Enter** to use defaults.

---

### 💻 C
1. Compile the C code:
   ```bash
   gcc gradient_check.c -o gradient_check -lm
   ```
2. Run:
   ```bash
   ./gradient_check
   ```

---

## 📊 Example Output
Both codes print a comparison of analytical vs numerical gradients:

```
Gradient Comparison Results:
Parameter | Analytical  | Numerical   | Relative Diff
--------- | ----------- | ----------- | --------------
theta0    | 0.023450    | 0.023450    | 1.20e-08
theta1    | -0.112340   | -0.112340   | 3.40e-09
theta2    | 0.004560    | 0.004560    | 8.90e-09

Overall Difference Metric: 1.23e-08
✓ GRADIENT CHECK PASSED! (Difference < 1e-07)
```

---

## 🧮 Key Concepts
- **Analytical Gradient**: Derived from logistic regression cost function.  
- **Numerical Gradient**: Estimated with finite differences.  
- **Gradient Checking**: Confirms implementation correctness.  

---

## 📜 License
This project is licensed under the [MIT License](./LICENSE).
