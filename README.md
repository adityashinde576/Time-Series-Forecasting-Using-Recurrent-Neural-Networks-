
# Time-Series Forecasting using Recurrent Neural Networks (PyTorch)

## Project Overview

This project implements a **Time-Series Forecasting model using Recurrent Neural Networks (RNN)** in **PyTorch**.
The model learns **temporal dependencies** from historical numerical data and predicts the **next value** in a sequence using a **many-to-one architecture**.

The project demonstrates how RNNs capture trends in sequential data such as:

* Stock prices
* Temperature readings
* Sales data

---

## Objective

* Model temporal dependencies in numerical data
* Understand univariate time-series representation
* Implement many-to-one sequence modeling
* Apply RNNs to real-world forecasting problems
* Learn the importance of data normalization in RNN training

---

## Problem Statement

Predict the next value in a time series using previous values.

Example:

```
Input:  [price t-3, price t-2, price t-1]
Output: price t
```

---

## Concepts Covered

* Univariate time-series modeling
* Temporal dependency
* Hidden state capturing trend information
* Many-to-one RNN architecture
* Data normalization and denormalization
* Backpropagation Through Time (BPTT)
* Mean Squared Error (MSE) loss

---

## Dataset Description

The project uses a **synthetic stock price dataset** for demonstration.

Example values:

```
[120.5, 121.3, 122.1, 121.8, 122.9, 124.0, 125.2, 126.1, ...]
```

* Each value represents a stock price at a given time step
* Dataset is univariate
* Data is normalized before training to stabilize gradients

---

## Sequence Creation

A sliding window approach is used.

Window size = 3
Example:

```
Input:  [120.5, 121.3, 122.1]
Output: 121.8
```

---

## Model Architecture

* Input Layer: 1 feature per time step
* RNN Layer:

  * Hidden size = 32
  * Batch-first configuration
* Fully Connected Layer:

  * Maps final hidden state to output

Architecture Type:

* Many-to-One (only final hidden state is used)

---

## Hyperparameters

* Window Size: 3
* Input Size: 1
* Hidden Size: 32
* Output Size: 1
* Learning Rate: 0.001
* Epochs: 500
* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam

---

## Training Process

1. Normalize the time-series data
2. Create input-output sequences
3. Train RNN using backpropagation through time
4. Monitor loss convergence
5. Evaluate predictions after denormalization

---

## Results Visualization

The project plots:

* Actual stock prices
* Predicted stock prices

The predicted curve closely follows the actual trend, showing that the RNN successfully learned the temporal pattern.

---

## Example Prediction

Input:

```
[120.5, 121.3, 122.1]
```

Output:

```
Predicted Output: 122.65
```

This value correctly follows the increasing trend in the data.

---

## How to Run the Project

### Step 1: Install Dependencies

```bash
pip install torch numpy matplotlib
```

### Step 2: Run the Notebook

Open the Jupyter Notebook and run all cells sequentially.

The notebook includes:

* Data preparation
* Model definition
* Training loop
* Evaluation
* Visualization
* Next-step prediction

---

## Key Learnings

* RNNs are sensitive to input scale; normalization is critical
* Many-to-one architectures are suitable for forecasting tasks
* Hidden states capture trend and momentum information
* Proper denormalization is required for real-world predictions

---

## Limitations

* Uses a small synthetic dataset
* Simple RNN (not LSTM/GRU)
* Not suitable for long-term dependencies
* No external CSV data loading

---

## Future Improvements

* Replace RNN with LSTM or GRU
* Use real-world stock or temperature CSV datasets
* Add multivariate time-series support
* Implement model saving and loading
* Deploy as a web app using Flask or FastAPI
