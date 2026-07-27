# Task 17 – Neural Network Fundamentals: Perceptron, Activation Functions & Backpropagation

**PKCERT AI & Software Development Internship**
**Intern:** Mahin | Reg. No. 230201012 | Batch CS 04B | Islamabad Institute of Technology (BSCS)

## Contents

- `Task17_Neural_Network_Fundamentals.ipynb` — full executed notebook (Parts A–D)
- `banknote.csv` — Banknote Authentication dataset used in Part C
- `outputs/` — saved figures and the pickled trained model
  - `A2_boundary_evolution.png`, `A3_xor_failure.png` — Part A perceptron plots
  - `B2_activations.png` — Part B activation function/derivative plots
  - `C3_loss_curve.png`, `C4_confusion_matrices.png`, `C5_lr_experiment.png` — Part C training/evaluation plots
  - `scratch_mlp_banknote.pkl` — saved weights/biases of the from-scratch MLP + scaler stats

## Summary of work

| Part | Topic | Marks |
|---|---|---|
| A | Perceptron from scratch, Iris (setosa vs. versicolor) linearly-separable demo, XOR failure case, perceptron vs. logistic regression discussion | 20 |
| B | Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax implemented + plotted with derivatives; vanishing gradient and dying ReLU analysis | 20 |
| C | Manual forward/backprop derivation (matrix form) + from-scratch NumPy implementation of a 4→8→1 MLP on the Banknote Authentication dataset; evaluated (accuracy/precision/recall/F1/confusion matrix) against a scikit-learn `MLPClassifier` baseline; learning-rate hyperparameter experiment | 45 |
| D | Model persistence (pickle), pipeline summary, hyperparameter findings, backprop implementation challenges, limitations of a manual MLP vs. production frameworks | 15 |

**Dataset used for Part C:** [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) (4 features: variance, skewness, curtosis, entropy of a wavelet-transformed image; binary target: genuine/forged). Not reused from any earlier internship task.

**Constraint compliance:** Parts A and C's core computations (perceptron learning rule, forward/backward propagation) are implemented using NumPy only — no TensorFlow/PyTorch/Keras autograd. A framework (`scikit-learn`) is used only for the Part C comparative baseline, as permitted.

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter nbconvert --to notebook --execute --inplace Task17_Neural_Network_Fundamentals.ipynb
```

## Git submission

```bash
git add Task17_Neural_Network_Fundamentals.ipynb README.md banknote.csv outputs/
git commit -m "Task 17: Neural network fundamentals — perceptron, activations, backpropagation"
git push
```
