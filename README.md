Here you go macha! 🔥

---

# 🏠 House Price Predictor

A machine learning web app that predicts California housing prices using multiple regression models, built with Streamlit and scikit-learn.

**Live Demo → [house-price-predictor-live.streamlit.app](https://house-price-predictor-live.streamlit.app/)**

---

## 📌 Problem Statement

Predicting housing prices is a classic regression problem with real-world impact — buyers, sellers, and analysts all benefit from data-driven price estimates. This project builds an end-to-end ML pipeline on the California Housing dataset, comparing multiple models and deploying the best one as an interactive web app.

---

## 🚀 Features

- **Overview** — dataset summary, feature descriptions, and key statistics
- **EDA** — interactive visualizations including correlation heatmap, feature distributions, and geographic price map
- **Model Comparison** — 6 models trained and evaluated side-by-side with R² and RMSE
- **Live Predictor** — input house features and get an instant price prediction from any model

---

## 📈 Model Performance

| Model                    | R² Score  | RMSE        |
| ------------------------ | --------- | ----------- |
| Linear Regression        | 66.5%     | $66,210     |
| Ridge Regression         | 66.5%     | $66,211     |
| Lasso Regression         | 56.5%     | $75,472     |
| Random Forest            | 79.2%     | $52,201     |
| Gradient Boosting        | 80.2%     | $50,899     |
| **HistGradientBoosting** | **84.8%** | **$44,697** |

> Best model: **HistGradientBoosting** with R² of 84.8% and RMSE of $44,697

---

## 🛠️ Tech Stack

- **Python** — core language
- **Streamlit** — web app framework
- **Scikit-learn** — ML models, preprocessing, RandomizedSearchCV
- **Pandas / NumPy** — data manipulation
- **Plotly / Matplotlib / Seaborn** — visualizations
- **Joblib** — model caching for fast startup

---

## ⚙️ Feature Engineering

Added 4 derived features to improve model performance:

- `rooms_per_person` = AveRooms / AveOccup
- `bedrooms_per_room` = AveBedrms / AveRooms
- `income_per_household` = MedInc / AveOccup
- `log_population` = log1p(Population)

These pushed R² from **80.5% → 84.8%** on the best model.

---

## ⚡ Performance Optimizations

- Models trained once and cached to disk via `joblib` — subsequent startups load in **3-4 seconds**
- `RandomizedSearchCV` used for hyperparameter tuning on Random Forest and HistGradientBoosting
- Error handling for dataset fetch failures with graceful Streamlit fallback

---

## 🏃 Run Locally

```bash
git clone https://github.com/masood-mashu/house-price-predictor.git
cd house-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Dataset

[California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) from scikit-learn — 20,640 samples, 8 features based on 1990 US Census data.

---

## 🔮 What I Learned

- Feature engineering on ratio/log transforms meaningfully improves tree-based model performance
- Disk caching with joblib dramatically reduces Streamlit Cloud cold start times
- RandomizedSearchCV is a practical alternative to GridSearch for tuning under time constraints
- HistGradientBoosting outperforms standard GradientBoosting with lower RMSE and faster training

---

## 🔭 Next Steps

- Add SHAP values for per-prediction explainability
- Add confidence intervals using Random Forest tree distribution
- Add cross-validation based model ranking

---

_Built by [Mohammed Masood](https://mohammed-masood.vercel.app/)_

---
