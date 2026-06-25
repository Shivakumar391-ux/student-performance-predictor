# 🎓 Student Performance Predictor

**ML Internship Project | VEMU Institute of Technology**

A machine learning web application that predicts a student's average academic score using demographic and socioeconomic features.

---

## 🔗 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-url-here.streamlit.app)

---

## 📌 Problem Statement

Predict a student's **average score** (mean of Math, Reading, Writing) using:
- Gender
- Race / Ethnicity
- Parental Level of Education
- Lunch Type (SES proxy)
- Test Preparation Course Completion

**Task:** Regression  
**Target:** `average_score` (0–100)

---

## 📊 Dataset

- **Source:** [Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Rows:** 1,000
- **Features:** 8 raw → 9 engineered

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| ML Library | scikit-learn, XGBoost |
| Feature Engineering | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| App Framework | Streamlit |
| Deployment | Streamlit Cloud |

---

## 🏗️ Project Structure

```
student-performance-predictor/
│
├── notebooks/
│   ├── Step1_problem_definition.docx
│   ├── Step2_eda_student_performance.ipynb
│   ├── Step3_cleaning_student_performance.ipynb
│   ├── Step4_features_student_performance.ipynb
│   ├── Step5_preprocessing_student_performance.ipynb
│   ├── Step6_training_student_performance.ipynb
│   ├── Step7_evaluation_student_performance.ipynb
│   └── Step8_serialization_student_performance.ipynb
│
├── model_artifacts/
│   ├── preprocessor.pkl
│   ├── feature_selector.pkl
│   ├── ridge_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── model.pkl
│   ├── column_info.json
│   └── metadata.json
│
├── app.py                  ← Streamlit app
├── requirements.txt
└── README.md
```

---

## 🤖 Models Used

| Model | Why Chosen |
|---|---|
| **Ridge Regression** | Regularized linear baseline; handles correlated features |
| **Random Forest** | Captures non-linear patterns; robust to noise |
| **XGBoost** | Gradient boosting; highest accuracy on tabular data |

---

## 📐 Feature Engineering

| Feature | Description |
|---|---|
| `parental_edu_rank` | Ordinal encoding of education level (0–5) |
| `prep_completed` | Binary flag: test prep done (1) or not (0) |
| `ses_index` | SES proxy: standard lunch = 1, free/reduced = 0 |
| `advantage_score` | Composite: prep×2 + ses×2 + edu_rank |
| `gender_code` | Binary: female=1, male=0 |
| `race_group_code` | Label encoded: Group A–E → 0–4 |
| `math_reading_gap` | math_score − reading_score |
| `reading_writing_gap` | reading_score − writing_score |
| `score_std` | Standard deviation across 3 subject scores |

---

## 🎯 Success Metrics

| Metric | Target | Result |
|---|---|---|
| Test MAE | < 5.0 | ✅ |
| Test RMSE | < 7.0 | ✅ |
| Test R² | > 0.85 | ✅ |

---

## 🚀 Run Locally

```bash
git clone https://github.com/Shivakumar391-ux/student-performance-predictor
cd student-performance-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 🏫 About

Built as part of an ML Internship at **VEMU Institute of Technology**  
B.Tech Computer Science Engineering (2023–2027)  
**Author:** Shivakumar | GitHub: [@Shivakumar391-ux](https://github.com/Shivakumar391-ux)
