# Iris Flower Classification (Machine Learning)

**Data Science Internship - Machine Learning Project**

**Intern:** Qadir Hussain
**Company:** Code Alpha.

---

## Project Overview

The Iris Flower Classification project is a beginner-friendly Machine Learning application developed in Python using Scikit-learn. It predicts the species of an iris flower (**Setosa, Versicolor, or Virginica**) from four measurements: sepal length, sepal width, petal length, and petal width.

The project loads and cleans the `Iris.csv` dataset using Pandas, splits it into training and testing sets, trains a K-Nearest Neighbors (KNN) classifier, and evaluates it using accuracy and a confusion matrix. After training, users can enter their own flower measurements in the terminal and get an instant species prediction.

This project demonstrates the basics of data preprocessing, model training, model evaluation, and building a simple interactive command line interface.

## Features

* Load and preprocess `Iris.csv` using Pandas
* Remove the `Id` column, missing values, and duplicate rows
* Clean species labels (`Iris-setosa` to `Setosa`)
* Split data into training (80%) and testing (20%) sets
* Train a K-Nearest Neighbors (KNN) classifier
* Calculate model accuracy
* Display a confusion matrix in the terminal
* Show and save the confusion matrix as an image (`confusion_matrix.png`)
* Predict flower species from user-entered measurements
* Validate user input (positive numbers only)
* Predict multiple flowers in one run

## Bonus Features

* Feature scaling with `StandardScaler` for better KNN performance
* Stratified train/test split to keep species balanced
* Confusion matrix plot saved automatically as a PNG
* Command line flags: `--no-plot` and `--no-predict`
* Easy to switch to a Decision Tree classifier

## Technologies Used

* Python 3.10+
* Pandas
* Scikit-learn
* Matplotlib

## Project Structure

```
iris-flower-classification/
│
├── iris_classifier.py
├── Iris.csv
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone Repository

```
git clone https://github.com/qadirhussain222786k-dev/iris-flower-classification.git
```

### 2. Open Project

```
cd iris-flower-classification
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

> The dataset (`Iris.csv`) is already included. You can also download it from
> [Kaggle](https://www.kaggle.com/datasets/saurabh00007/iriscsv).

## Running the Project

```
python iris_classifier.py
```

or

```
py iris_classifier.py
```

Optional flags:

```
python iris_classifier.py --no-plot       # do not open the plot window
python iris_classifier.py --no-predict    # skip the prediction prompt
```

## Dependencies

```
pandas
scikit-learn
matplotlib
```

## Sample Prediction

```
--- Predict a flower species (measurements in cm) ---
Sepal length: 5.1
Sepal width: 3.5
Petal length: 1.4
Petal width: 0.2

>>> Predicted species: Iris Setosa

Predict another? (y/n):
```

## Using a Decision Tree Instead

In the `train_model` function, replace the KNN line with:

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
```

## Concepts Covered

* Python Fundamentals
* Functions
* File Handling (CSV)
* Data Preprocessing with Pandas
* Train/Test Split
* Feature Scaling
* K-Nearest Neighbors (KNN)
* Classification
* Model Evaluation (Accuracy)
* Confusion Matrix
* Data Visualization with Matplotlib
* Input Validation
* Third-Party Packages (Pandas, Scikit-learn, Matplotlib)

## Author

**Name:** Qadir Hussain
**University:** Capital University of Science and Technology (CUST)
**Program:** BS Software Engineering
