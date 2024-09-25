Elbette! Aşağıda, sağladığın kod örneği ve belirttiğin değişiklikler göz önünde bulundurularak güncellenmiş README dosyası yer alıyor:

````markdown
# ZipML

<p align="center">
  <a href="https://badge.fury.io/py/zipml">
    <img src="https://badge.fury.io/py/zipml.svg" alt="PyPI version" />
  </a>
  <a href="https://travis-ci.com/abdozmantar/zipml">
    <img src="https://travis-ci.com/abdozmantar/zipml.svg?branch=main" alt="Build Status" />
  </a>
  <a href="https://github.com/abdozmantar/zipml/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" />
  </a>
  <a href="https://pypi.org/project/zipml/">
    <img src="https://img.shields.io/pypi/pyversions/zipml.svg" alt="Python Versions" />
  </a>
  <a href="https://pypi.org/project/zipml/">
    <img src="https://img.shields.io/pypi/dm/zipml.svg" alt="Downloads" />
  </a>
</p>

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

## Installation

Install the package via pip:

```bash
pip install zipml
```
````

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
from zipml import split_data, compare_models, save_confusion_matrix
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Load data
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Define models
models = [
    RandomForestClassifier(),
    LogisticRegression(),
    GradientBoostingClassifier()
]

# Compare models
best_model, performance = compare_models(models, X_train, X_test, y_train, y_test)
print(f"Best model: {best_model}")

# Save confusion matrix
save_confusion_matrix(y_test, best_model.predict(X_test))
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
