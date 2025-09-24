# Gradient Checking in Logistic Regression

## 📌 Description
This project implements **Gradient Checking using Finite Differences** in **Logistic Regression**.  
It provides implementations:
- **C** for performance and low-level demonstration.


The goal is to validate analytical gradients against numerical approximations, ensuring correctness in optimization algorithms.

---

## 📂 Files
- `gradient checking logistic regression.c` → C implementation (fixed dataset size, efficient).  

---

### 💻 C
1. Compile the C code:
   ```bash
   gcc gradient checking logistic regression.c -o gradient checking logistic regression -lm
   ```
2. Run:
   ```bash
   ./gradient checking logistic regression
   ```

---
## ✨ Features

* Logistic Regression Cost Function
  Implements sigmoid-based logistic regression with binary labels.

* Gradient Checking

  * Analytical gradient (exact derivative).
  * Numerical gradient (finite difference approximation).
  * Reports relative differences and overall validation.

* Advanced Analysis Tools

  1. Save Results to File – Store gradient comparisons in a report.
  2. Convergence Analysis – Run gradient descent steps and track cost reduction.
  3. Test Different Epsilons – Explore sensitivity of gradient check to step size.
  4. Performance Benchmark – Compare runtime of analytical vs. numerical gradients.
  5. Interactive Mode – Experiment with custom theta values and compute costs/gradients.
     
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
