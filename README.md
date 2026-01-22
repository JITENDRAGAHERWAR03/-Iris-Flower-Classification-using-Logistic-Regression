# 🌸 Iris Flower Classification using Logistic Regression

This project demonstrates a **machine learning classification model** built using **Logistic Regression** to predict the species of iris flowers based on their physical characteristics.  
It is a beginner-friendly project designed to understand the **end-to-end ML workflow**.

---

## 📌 Project Overview

The Iris dataset is a classic dataset in machine learning containing measurements of:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The goal is to classify flowers into one of the following species:
- Setosa
- Versicolor
- Virginica

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries Used:**
  - Pandas
  - Scikit-learn
  - NumPy

---

## 📂 Dataset

- Source: Built-in **Iris Dataset** from Scikit-learn
- Total Samples: 150
- Features: 4 numerical features
- Target Classes: 3 flower species

---

## ⚙️ Workflow

1. Load the Iris dataset
2. Split the data into training and testing sets
3. Train a Logistic Regression model
4. Evaluate model performance using:
   - Accuracy Score
   - Classification Report
5. Create a custom function for real-time predictions


---

## 🚀 Model Training

- Algorithm Used: **Logistic Regression**
- Train-Test Split: **70% training / 30% testing**
- Model Accuracy: ~**97%**

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

These metrics help in understanding the performance of the model across all classes.

---

## 🔍 Example Predictions

```python
predict_species(5.1, 3.5, 1.4, 0.2)
# Output: setosa

predict_species(6.7, 3.0, 5.2, 2.3)
# Output: virginica
