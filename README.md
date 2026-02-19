# 📈 Cryptocurrency Price Predictor

An interactive web application for analyzing and forecasting cryptocurrency prices using Facebook Prophet.

Built with Streamlit for deployment and Plotly for visualization.

---

## 🚀 Features

- Upload custom cryptocurrency CSV datasets
- Automatic data cleaning and preprocessing
- 6-month future price forecasting
- Out-of-sample model evaluation
- Performance metrics:
  - MAE (Mean Absolute Error)
  - R² Score
  - MAPE
- Residual analysis
- Monthly, yearly and daily trend analysis
- Confidence interval visualization

---

## 🛠 Tech Stack

- Python
- Streamlit
- Facebook Prophet
- Pandas
- NumPy
- Scikit-learn
- Plotly

---

## 📊 Machine Learning Approach

- Time series forecasting using Prophet
- Multiplicative seasonality
- Weekly & yearly seasonality
- Optional volume regressor
- Proper train/test split (30-day holdout)
- Out-of-sample evaluation (no data leakage)

---

## 📂 Dataset Format

Your CSV file must include:

- A Date column
- A Price column
- Optional Volume column

Example:

| Date       | Price  | Volume |
|------------|--------|--------|
| 2023-01-01 | 16500  | 123456 |
| 2023-01-02 | 16720  | 143210 |

---

## ▶️ How to Run Locally (Mac / Windows)

1. Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/crypto-price-predictor.git
```

2. Navigate into the folder:

```
cd crypto-price-predictor
```

3. Create virtual environment:

```
python3 -m venv venv
source venv/bin/activate     # Mac
venv\Scripts\activate        # Windows
```

4. Install dependencies:

```
pip install -r requirements.txt
```

5. Run the app:

```
streamlit run app.py
```

---

## 🌍 Deployment

This project can be deployed easily on:

- Streamlit Cloud
- Render
- Railway

---

## 📈 Example Use Cases

- Crypto market analysis
- Educational time-series projects
- Forecasting demonstrations
- FinTech portfolio projects

---

## ⚠️ Disclaimer

Cryptocurrency markets are highly volatile. Predictions are based on historical patterns and should not be used as financial advice.

---

## 👨‍💻 Author

Your Name  
Computer Science Student  
