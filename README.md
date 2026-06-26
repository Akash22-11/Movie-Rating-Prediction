# Movie-Rating-Prediction

# 🎬 Movie Rating Prediction using Machine Learning

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:6A5ACD,50:8A2BE2,100:4169E1&height=220&section=header&text=Movie%20Rating%20Prediction&fontSize=45&fontColor=ffffff&animation=fadeIn"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/Akash22-11/Movie_rating_prediction-?style=flat-square"/>
  <img src="https://img.shields.io/github/stars/Akash22-11/Movie_rating_prediction-?style=flat-square"/>
  <img src="https://img.shields.io/github/forks/Akash22-11/Movie_rating_prediction-?style=flat-square"/>
</p>

---

# 📖 Overview

This project predicts **movie ratings** using various movie attributes from the **TMDB 5000 Movies Dataset**.

The complete Machine Learning pipeline includes:

* 📊 Exploratory Data Analysis
* 🧹 Data Cleaning
* ⚙️ Feature Engineering
* 🤖 Multiple Regression Models
* 📈 Model Evaluation
* 🎯 Feature Importance Analysis

The objective is to estimate a movie's **TMDB rating (`vote_average`)** based on features like **genre, budget, revenue, popularity, runtime, language, director, cast, and release year**.

---

# ✨ Features

* 📊 Exploratory Data Analysis (EDA)
* 🧹 Data Preprocessing
* 🛠 Feature Engineering
* 🤖 Multiple ML Regression Models
* 📈 Performance Comparison
* 🎯 Feature Importance
* 📉 Professional Visualizations

---

# 📂 Project Structure

```text
Movie_rating_prediction-/
│
├── data/
│   └── movies.csv
│
├── plots/
│   ├── movie_eda.png
│   └── movie_evaluation.png
│
├── src/
│   └── movie_rating_prediction.py
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

| Property | Value            |
| -------- | ---------------- |
| Dataset  | TMDB 5000 Movies |
| Movies   | 4,803            |
| Features | 24               |
| Target   | vote_average     |
| Problem  | Regression       |

---

# ⚙️ Machine Learning Workflow

```text
TMDB Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Evaluation
      │
      ▼
Prediction
```

---

# 🛠 Feature Engineering

✔ Genre One-Hot Encoding

✔ Director Average Rating

✔ Cast Count

✔ Release Year Extraction

✔ Decade Extraction

✔ Log Budget

✔ Log Revenue

✔ Log Popularity

✔ Log Vote Count

✔ English Language Indicator

---

# 🤖 Models Used

| Model                |        MAE |       RMSE |   R² Score |
| -------------------- | ---------: | ---------: | ---------: |
| Linear Regression    |     0.3388 |     0.4780 |     0.7717 |
| Ridge Regression     |     0.3389 |     0.4780 |     0.7717 |
| Random Forest        |     0.3168 |     0.4711 |     0.7783 |
| 🌟 Gradient Boosting | **0.3085** | **0.4514** | **0.7964** |

---

# 🏆 Best Model

## 🌟 Gradient Boosting Regressor

| Metric   |      Score |
| -------- | ---------: |
| MAE      | **0.3085** |
| RMSE     | **0.4514** |
| R² Score | **0.7964** |

> **Average prediction error is approximately 0.31 rating points on a 0–10 scale.**

---

# 📈 Visualizations

## 📊 Exploratory Data Analysis

<p align="center">
<img src="./plots/movie_eda.png" width="100%">
</p>

---

## 📉 Model Evaluation

<p align="center">
<img src="./plots/movie_evaluation.png" width="100%">
</p>

---

# 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Akash22-11/Movie_rating_prediction-.git

# Navigate into the project
cd Movie_rating_prediction-

# Install dependencies
pip install -r requirements.txt

# Run the project
python src/movie_rating_prediction.py
```

---

# 💻 Tech Stack

| Category         | Technologies        |
| ---------------- | ------------------- |
| Language         | Python              |
| Data Processing  | Pandas, NumPy       |
| Machine Learning | Scikit-Learn        |
| Visualization    | Matplotlib, Seaborn |

---

# 📌 Results

* ✅ Complete Machine Learning Pipeline
* ✅ Data Cleaning & Feature Engineering
* ✅ Multiple Regression Algorithms
* ✅ Model Comparison
* ✅ Feature Importance Analysis
* ✅ High Prediction Accuracy (R² ≈ 0.80)
* ✅ Professional Visualizations

---

# 📦 Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

or install directly:

```bash
pip install -r requirements.txt
```

---

# 📸 Sample Output

| EDA                                            | Evaluation                                            |
| ---------------------------------------------- | ----------------------------------------------------- |
| <img src="./plots/movie_eda.png" width="450"/> | <img src="./plots/movie_evaluation.png" width="450"/> |

---

# 👨‍💻 Author

### **Akash Sarkar**

🎓 B.Tech Information Technology Student

💼 **CodSoft Data Science Internship — Task 2**

🔗 GitHub: **https://github.com/Akash22-11**

---

<p align="center">

### ⭐ If you found this project helpful, consider giving it a Star!

</p>

<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4169E1,100:8A2BE2&height=120&section=footer"/>
</p>
