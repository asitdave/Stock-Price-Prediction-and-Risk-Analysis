# Stock Price Prediction and Risk Analysis

## Project Overview  
This project explores stock market data for leading technology companies – Apple (AAPL), Google (GOOGL), Amazon (AMZN), and Microsoft (MSFT).
Through a combination of statistical analysis, technical indicators, portfolio risk assessment, and predictive modeling, the project aims to:
* Understand stock behavior and volatility
* Assess risk vs. return for individual stocks and portfolios
* Identify market patterns and correlations
* Predict future stock prices using ARIMA and LSTM models



## Project Highlights  

### Objective  
* Perform exploratory and predictive analysis of stock prices
* Compare risk-return trade-offs across assets
* Forecast future stock trends

### Concepts & Techniques Used  
* **Exploratory Data Analysis (EDA)**: summary statistics, price trends, correlation heatmaps
* **Technical Indicators**: moving averages, Bollinger Bands, rolling statistics
* **Portfolio Risk Assessment**: portfolio returns, volatility, covariance/correlation matrices, Value at Risk (VaR), efficient frontier visualization
* **Predictive Modeling**:
   * ARIMA for classical time-series forecasting
   * LSTM networks for deep learning-based sequence modeling

### Tools & Libraries  
- **Python** (Pandas, NumPy, yfinance (Yahoo Finance API))  
- **Matplotlib & Seaborn** (data visualization)  
- **Modeling** statsmodel (ARIMA), tensorflow/keras (LSTM)  


### Interactive Web Application
The accompanying Jupyter Notebook offers a comprehensive walkthrough of stock price exploration, portfolio risk assessment, and forecasting methodologies.
To make the analysis more interactive and user-friendly, I’ve also developed and hosted a Streamlit web application - which you can access [here](https://asitdave-stock-price-prediction-and-risk-analysis-app.streamlit.app). This app allows users to:
* Select a stock of their choice and visualize how the model performs over the last six months
* Forecast future stock prices by specifying the number of days ahead for prediction

This interactive dashboard provides an intuitive way to explore the underlying models and observe their predictive performance in real time.

## Key Results  

1. T**rends & Volatility**: Tech stocks generally move together (high positive correlations). Short-term decoupling observed during specific market events (e.g., Microsoft 2024 outage).
2. **Portfolio Performance**:
   * Equally weighted portfolios smooth volatility compared to single stocks
   * Diversification benefits limited due to strong correlations among tech giants
3. **Predictive Modeling**:
	* ARIMA provides a simple, interpretable baseline for short-term predictions
	* LSTM captures overall trends better but lags in sudden price movements
	* Errors remain within realistic ranges, showing potential for practical forecasting

---