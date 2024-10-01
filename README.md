# ZipML

<div align="center">

<img src="https://github.com/abdozmantar/zipml/blob/main/public/logo.png?raw=true" alt="ZipML Logo" width="180px"/>

<br/><br/>

[![PyPI](https://img.shields.io/pypi/v/zipml.svg)](https://pypi.org/project/zipml/)
[![Build Status](https://github.com/abdozmantar/zipml/actions/workflows/ci.yml/badge.svg)](https://github.com/abdozmantar/zipml/actions/workflows/ci.yml)
[![Python Versions](https://img.shields.io/pypi/pyversions/zipml.svg)](https://pypi.org/project/zipml/)
[![License](https://img.shields.io/github/license/abdozmantar/zipml.svg)](https://github.com/abdozmantar/zipml/blob/main/LICENSE)

[![Downloads/Week](https://static.pepy.tech/badge/zipml/week)](https://pepy.tech/project/zipml)
[![Downloads](https://static.pepy.tech/badge/zipml)](https://pepy.tech/project/zipml)
[![Stars](https://img.shields.io/github/stars/abdozmantar/zipml?color=yellow&style=flat&label=%E2%AD%90%20Stars)](https://github.com/abdozmantar/zipml/stargazers)

</div>

**ZipML** is a lightweight AutoML library designed for small datasets, offering essential helper functions like train-test splitting, model comparison, and confusion matrix generation.

## Features

- **Automated Model Training**: Automatically train and compare machine learning models on your dataset.
- **Helper Functions**:
  - Train-test split functionality for easy data management.
  - Confusion matrix generation and the ability to save it as a PNG.
  - Custom logging features for better tracking of your model's performance.
- **Model Comparison**: Compare the performance of different models with ease, providing metrics and visual feedback.
- **CLI Support**: Run machine learning tasks directly from the command line.
- **Extensible**: Add your own models and customize workflows as needed.
- **Visualization Tools**: Includes tools for visualizing model performance metrics, helping to understand model behavior better.
- **Hyperparameter Tuning**: Support for hyperparameter tuning to optimize model performance.
- **Data Preprocessing**: Built-in data preprocessing steps to handle missing values, scaling, and encoding.

---

### Package Structure

```bash
zipml/
│
├── data/
│   ├── encoding.py
│   ├── file_operations.py
│   ├── split_data.py
│
├── model/
│   ├── analyze_model_predictions.py
│   ├── calculate_model_results.py
│   ├── measure_prediction_time.py
│
├── utils/
│   ├── calculate_sentence_length_percentile.py
│   ├── read_time_series_data.py
│
├── visualization/
│   ├── combine_and_plot_model_results.py
│   ├── plot_random_image.py
│   ├── plot_time_series.py
│   ├── save_and_plot_confusion_matrix.py
│
└── zipml.py
```

---

### How to Use the `zipml` Package

The `zipml` package provides a variety of utilities for preprocessing data, analyzing models, and visualizing results, all designed to simplify AI and machine learning workflows. Below are the instructions for using some of the key functions.

#### 1. **Model Evaluation**

- **`analyze_model_predictions.py`**: Evaluates model predictions by comparing them with actual values and returns a detailed dataframe of predictions along with the most incorrect predictions.
  ```python
  from zipml.model import analyze_model_predictions
  val_df, most_wrong = analyze_model_predictions(best_model, X_test, y_test)
  ```

## Installation

Install the package via pip:

```bash
pip install zipml
```

Alternatively, clone the repository:

```bash
git clone https://github.com/abdozmantar/zipml.git
cd zipml
pip install .
```

## Usage

### Example Usage with Code

Here's a practical example of how to use ZipML:

```python
import pandas as pd
from zipml.model import analyze_model_predictions
from zipml.model import calculate_model_results
from zipml.visualization import save_and_plot_confusion_matrix
from zipml.data import split_data
from zipml import compare_models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# Sample dataset
data = {
    'feature_1': [0.517, 0.648, 0.105, 0.331, 0.781, 0.026, 0.048],
    'feature_2': [0.202, 0.425, 0.643, 0.721, 0.646, 0.827, 0.303],
    'feature_3': [0.897, 0.579, 0.014, 0.167, 0.015, 0.358, 0.744],
    'feature_4': [0.457, 0.856, 0.376, 0.527, 0.648, 0.534, 0.047],
    'feature_5': [0.046, 0.118, 0.222, 0.001, 0.969, 0.239, 0.203],
    'target': [0, 1, 1, 1, 1, 1, 0]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Splitting data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = split_data(X, y)

# Define models
models = [
    RandomForestClassifier(),
    LogisticRegression(),
    GradientBoostingClassifier()
]

# Compare models and select the best one
best_model, performance = compare_models(models, X_train, X_test, y_train, y_test)
print(f"Best model: {best_model} with performance: {performance}")

# Calculate performance metrics for the best model
best_model_metrics = calculate_model_results(y_test, best_model.predict(X_test))

# Analyze model predictions
val_df, most_wrong = analyze_model_predictions(best_model, X_test, y_test)

# Save and plot confusion matrix
save_and_plot_confusion_matrix(y_test, best_model.predict(X_test), save_path="confusion_matrix.png")
```

### CLI Usage

You can run ZipML from the command line using the following commands:

#### Train a Single Model

```bash
zipml --train train.csv --test test.csv --model randomforest --result results.json
```

- `--train`: Path to the training dataset CSV file.
- `--test`: Path to the testing dataset CSV file.
- `--model`: Name of the model to be trained (e.g., `randomforest`, `logisticregression`, `gradientboosting`).
- `--result`: Path to the JSON file where results will be saved.

#### Compare Multiple Models

```bash
zipml --train train.csv --test test.csv --compare --compare_models randomforest svc knn --result results.json
```

- `--compare`: A flag to indicate multiple model comparison.
- `--compare_models`: A list of models to compare (e.g., `randomforest`, `logisticregression`, `gradientboosting`).
- `--result`: Path to the JSON file where comparison results will be saved.

#### Load a Pre-trained Model and Make Predictions

```bash
zipml --load_model trained_model.pkl --test test.csv --result predictions.json
```

- `--load_model`: Path to the saved model file.
- `--test`: Path to the testing dataset CSV file.
- `--result`: Path to the JSON file where predictions will be saved.

#### Save the Trained Model

To save the trained model after training:

```bash
zipml --train train.csv --test test.csv --model randomforest --save_model trained_model.pkl
```

- `--result`: Path to the file where the trained model will be saved.

### Output

- The output of training and comparison commands will include various performance metrics such as accuracy, precision, recall, and F1 score.
- Results will be saved in JSON format, making them easy to review and analyze.

## Dependencies

- Python 3.6+
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/foo`).
3. Commit your changes (`git commit -am 'Add some foo'`).
4. Push to the branch (`git push origin feature/foo`).
5. Open a pull request.

## Author

**Abdullah OZMANTAR**
GitHub: [@abdozmantar](https://github.com/abdozmantar)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/abdozmantar/zipml/blob/main/LICENSE) file for details.
