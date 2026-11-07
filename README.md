# Prediction of Wine Type using Deep Learning

A small neural network, built with Keras, that predicts whether a wine is **red or white** from 11 physicochemical measurements.

**Repository:** https://github.com/Ajeetku07/Prediction-of-Wine-Type-using-Deep-Learning

---

## Overview

Red and white wines differ in their chemistry (acidity, sulfur dioxide, sugar and so on). This project combines the red and white wine data from the UCI Wine Quality dataset and trains a feed-forward neural network to tell the two types apart.

- **Task:** binary classification (red = 1, white = 0)
- **Input:** 11 physicochemical features per wine
- **Model:** fully connected network in Keras (two hidden layers)
- **Notebook:** `Untitled9.ipynb`

## Dataset

[UCI Machine Learning Repository: Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality) (Cortez et al., 2009). The notebook reads both files directly from the UCI archive.

| Subset | Wines |
|--------|-------|
| Red    | 1,599 |
| White  | 4,898 |
| **Total** | **6,497** |

**Features (11):** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.

**Target:** a `type` column added by the notebook (1 = red, 0 = white). The `quality` score in the original data is not used as a feature.

## Approach

1. Load the red and white wine CSV files with pandas.
2. Explore the data: preview rows, summary statistics, and a check for missing values.
3. Compare the alcohol distribution of red and white wines with histograms.
4. Add the `type` label and combine both datasets into one table.
5. Split the data into training (66%) and test (34%) sets with `train_test_split` (`random_state=45`).
6. Build, compile and train the neural network.
7. Predict wine type for the held-out test set (the model outputs a probability between 0 and 1).

## Model

| Layer | Units | Activation | Parameters |
|-------|-------|------------|------------|
| Dense (input: 11 features) | 12 | ReLU | 144 |
| Dense | 9 | ReLU | 117 |
| Dense (output) | 1 | Sigmoid | 10 |
| **Total** | | | **271** |

- **Loss:** binary cross-entropy
- **Optimizer:** Adam
- **Training:** 3 epochs, batch size 1, on 4,288 training samples

## Results

Training metrics from the notebook:

| Epoch | Loss | Training accuracy |
|-------|------|-------------------|
| 1 | 0.7989 | 88.18% |
| 2 | 0.1893 | 93.70% |
| 3 | 0.1521 | 94.75% |

The notebook generates predictions for the test set (about 2,209 wines), but it does not yet report an accuracy or confusion matrix on that test set. The 94.75% above is **training** accuracy.

## How to run

1. Open `Untitled9.ipynb` in Google Colab or Jupyter.
2. Install the dependencies if needed:
   ```bash
   pip install pandas numpy matplotlib scikit-learn tensorflow
   ```
3. Run all cells from top to bottom. The data downloads automatically from the UCI archive.

> **Note:** the notebook combines the two datasets with `DataFrame.append`, which was removed in pandas 2.0. On a newer pandas version, replace it with
> `wines = pd.concat([red, white], ignore_index=True)`.

## Tech stack

Python, pandas, NumPy, Matplotlib, scikit-learn (`train_test_split`), Keras

## Possible improvements

- Report accuracy and a confusion matrix on the held-out test set.
- Scale the input features (for example with `StandardScaler`) before training.
- Train for more epochs with a larger batch size, and use a validation split.
- Compare against a simple baseline such as logistic regression.

## Author

Ajeet Kumar Singh ([@Ajeetku07](https://github.com/Ajeetku07))
