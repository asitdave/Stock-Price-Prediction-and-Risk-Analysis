# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import io
import warnings
warnings.filterwarnings("ignore")

# plt.rcParams.update({
#     'figure.dpi': 300,            # Set the default DPI to 300
#     'figure.facecolor': 'white',   # Set the default figure facecolor to white
#     'axes.grid': False,           # Display grid on axes
#     'grid.color': 'black',        # Set the grid color to black
#     'grid.linestyle': '--',       # Set the grid line style to dashed
#     'grid.linewidth': 0.5,        # Set the grid line width to 0.5
#     'grid.alpha': 0.5,            # Set the grid alpha to 0.5
#     'xtick.top': True,            # Display ticks on the top of the x-axis
#     'xtick.bottom': True,         # Display ticks on the bottom of the x-axis
#     'ytick.left': True,           # Display ticks on the left of the y-axis
#     'ytick.right': True,          # Display ticks on the right of the y-axis
#     'xtick.direction': 'in',       # Set the direction of x-axis ticks to 'in'
#     'ytick.direction': 'in',       # Set the direction of y-axis ticks to 'in'
#     'font.size': 10,              # Set the font size
#     'text.usetex': True,          # Enable LaTeX rendering
#     'font.family': 'serif',       # Font family for text
#     'font.serif': ['Computer Modern Roman'],  # Font name for serif font (Others: 'Times New Roman', 'Georgia', 'Helvetica', 'Palatino')
#     'font.weight': 'bold',      # Font weight
#     'axes.linewidth': 0.25,       # Spine line width
#     'xtick.major.width': 0.25,    # Major tick line width for x-axis
#     'xtick.minor.width': 0.25,    # Minor tick line width for x-axis
#     'ytick.major.width': 0.25,    # Major tick line width for y-axis
#     'ytick.minor.width': 0.25,     # Minor tick line width for y-axis
#     'legend.frameon': False,      # Disable legend frame
# })




st.set_page_config(page_title="Stock Price Forecast", layout="wide")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(ttl=3600)
def fetch_close_series(ticker: str, start_date: str, end_date: str):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df is None or df.empty:
        return None
    df = df[['Close']].dropna()
    return df

def adf_test(series):
    """Return p-value and summary of ADF test."""
    res = adfuller(series.dropna())
    return {'adf_stat': res[0], 'pvalue': res[1], 'crit': res[4]}

def select_order_with_pmdarima(series, d):
    try:
        from pmdarima import auto_arima
        stepwise = auto_arima(series, start_p=0, start_q=0, max_p=5, max_q=5,
                              d=d, seasonal=False, trace=False,
                              error_action='ignore', suppress_warnings=True, stepwise=True)
        return tuple(stepwise.order)
    except Exception:
        return None

def simple_grid_search_order(series, d, max_p=3, max_q=3):
    """Small AIC grid search using statsmodels (fast for small grid)."""
    best_aic = np.inf
    best_order = (0, d, 0)
    for p in range(0, max_p+1):
        for q in range(0, max_q+1):
            try:
                model = ARIMA(series, order=(p, d, q))
                res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue
    return best_order

def business_days_between(start_date, n_bdays):
    start = pd.to_datetime(start_date)
    bdays = pd.bdate_range(start=start + pd.tseries.offsets.BDay(1), periods=n_bdays)
    return bdays

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'MAPE(%)': mape}

def future_business_days(start_date, n_days=60):
    """
    Generate next n_days business days starting after start_date
    """
    start = pd.to_datetime(start_date) + pd.tseries.offsets.BDay(1)
    bdays = pd.bdate_range(start=start, periods=n_days)
    return bdays


# -------------------------
# UI
# -------------------------
st.title("Stock Price Forecasting (walk-forward test + future forecast)")

col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Ticker symbol (yfinance)", value="AAPL")
    horizon_days = st.number_input("Forecast horizon (days)", min_value=1, max_value=15, value=5)
    run_button = st.button("Run ARIMA Forecast")
with col2:
    st.markdown("**Notes**")
    st.markdown("- Last 6 months are used for out-of-sample testing.")
    st.markdown("- Approx. 10 years before that are used for training.")
    st.markdown("- Walk-forward (rolling) 1-step forecasts on the test window.")
    st.markdown("- Final model is refit on full data and used to forecast next N business days using SARIMA.")
    st.markdown("- If `pmdarima` is installed, it will be used to suggest (p,d,q). Otherwise a small grid search is used.")

if not run_button:
    st.stop()

# -------------------------
# Data fetch
# -------------------------
today = pd.Timestamp.today().normalize()
start_required = today - pd.DateOffset(years=11)  # ~11 years to be safe
df = fetch_close_series(ticker.upper(), start_required.strftime("%Y-%m-%d"), (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

if df is None or df.empty:
    st.error("No data returned for ticker. Check ticker symbol and internet connection.")
    st.stop()

last_date = df.index[-1]
test_start = last_date - pd.DateOffset(months=6)
train = df[df.index < test_start]['Close'].copy()
test = df[df.index >= test_start]['Close'].copy()

if len(train) < 200:
    st.warning(f"Training series is short ({len(train)} observations). ARIMA may not perform well. Proceeding with available history.")

st.write(f"Data range fetched: {df.index[0].date()} → {df.index[-1].date()}")
st.write(f"Train size: {len(train)}  |  Test size (last 6m): {len(test)}")

# -------------------------
# Stationarity (ADF)
# -------------------------
st.subheader("Stationarity checks")
adf_res = adf_test(train)
st.write(f"ADF statistic: {adf_res['adf_stat']:.4f}, p-value: {adf_res['pvalue']:.4f}")
st.write("Critical values:", {k: float(v) for k,v in adf_res['crit'].items()})

if adf_res['pvalue'] < 0.05:
    d = 0
    st.info("Train appears stationary (ADF p < 0.05). Setting d = 0.")
else:
    d = 1
    st.info("Train appears non-stationary (ADF p >= 0.05). Setting d = 1 (first difference).")

# Option: let user force log transform
use_log = st.checkbox("Use log transform for modeling (fit on log prices)", value=False)
if use_log:
    model_train_series = np.log(train)
    model_test_series = np.log(test)
else:
    model_train_series = train
    model_test_series = test

# -------------------------
# Select (p,d,q)
# -------------------------
st.subheader("Selecting ARIMA order (p,d,q)")
with st.spinner("Selecting model order..."):
    order = select_order_with_pmdarima(model_train_series, d)
    if order is not None:
        st.success(f"auto_arima suggested order: {order}")
    else:
        st.info("pmdarima not available or failed. Running quick grid search (p,q in 0..3).")
        order = simple_grid_search_order(model_train_series, d, max_p=3, max_q=3)
        st.success(f"Grid search selected order: {order}")

# Small safety: ensure d in order matches chosen d
order = (1, 1, 1) if d == 1 else (order[0], d, order[2])
st.write(f"Final order used for walk-forward: {order}")

# -------------------------
# Walk-forward forecasting (test period)
# -------------------------
st.subheader("Walk-forward forecasting on test window (last 6 months)")
history = model_train_series[ticker].copy().tolist()  # model series (log or raw)
predictions = []
failed_steps = 0

progress_bar = st.progress(0)
for t in range(len(model_test_series)):
    try:
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=1)[0]
        predictions.append(yhat)
    except Exception as e:
        # fallback: use last observed value (naive)
        failed_steps += 1
        predictions.append(history[-1])
    # append the actual observed (transformed) test value for next iteration
    history.append(model_test_series[ticker].iloc[t])
    progress_bar.progress((t+1)/len(model_test_series))

if failed_steps > 0:
    st.warning(f"{failed_steps} steps in walk-forward failed to fit model; naive fallback used for those steps.")

# Invert transform if log used
if use_log:
    pred_series = pd.Series(np.exp(predictions), index=test.index)
else:
    pred_series = pd.Series(predictions, index=test.index)

# Calculate metrics on original scale
metrics = evaluate_metrics(test.values, pred_series.values)
# Directional accuracy
actual_dir = np.sign(test[ticker].diff().dropna())
pred_dir = np.sign(pred_series.diff().dropna())
directional_acc = (actual_dir == pred_dir).mean()

# Residuals & Ljung-Box
residuals = test[ticker] - pred_series
lb_test = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
lb_pvalue = lb_test['lb_pvalue'].iloc[0] if not lb_test.empty else np.nan

# -------------------------
# Show metrics & plots
# -------------------------
st.subheader("Test performance (last 6 months)")
colm1, colm2, colm3, colm4 = st.columns(4)
colm1.metric("RMSE", f"{metrics['RMSE']:.4f}")
colm2.metric("MAE", f"{metrics['MAE']:.4f}")
colm3.metric("MAPE (%)", f"{metrics['MAPE(%)']:.2f}%")
colm4.metric("Directional Acc.", f"{directional_acc:.2%}")

st.write(f"Ljung-Box (lag=10) p-value on residuals: {lb_pvalue:.4f}")

# Plot: Train + Test + Predictions (test window)
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(train.index, train.values, label='Train', color='black', alpha=0.6)
ax.plot(test.index, test.values, label='Test (actual)', color='blue')
ax.plot(pred_series.index, pred_series.values, label='ARIMA Predicted (walk-forward)', color='red', linestyle='--')
ax.set_title(f"{ticker.upper()}  Close: Train / Test / ARIMA Predictions")
ax.legend()
st.pyplot(fig)

# Zoomed: Test only
fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.plot(test.index, test.values, label='Test (actual)', color='blue')
ax2.plot(pred_series.index, pred_series.values, label='Predicted', color='red', linestyle='--')
ax2.set_title("Test window (last 6 months): Actual vs Predicted")
ax2.legend()
st.pyplot(fig2)

# Residual plot
fig3, ax3 = plt.subplots(figsize=(12,3))
ax3.plot(residuals.index, residuals.values)
ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_title("Residuals (Test - Predicted)")
st.pyplot(fig3)

# -------------------------
# Refit on full data and forecast future
# -------------------------
st.subheader("Refit on full series and forecast future horizon using SARIMA")
full_series = np.log(df['Close']) if use_log else df['Close']

with st.spinner("Refitting ARIMA on full data and producing future forecast..."):
    try:
        # model_full = ARIMA(full_series, order=order)
        # fit_full = model_full.fit()
        P, D, Q, s = order[0], order[1], order[2], 5
        model_full = SARIMAX(full_series, order=order, seasonal_order=(P,D,Q,s))
        fit_full = model_full.fit()
    except Exception as e:
        st.error(f"Full-fit failed: {e}")
        st.stop()

# number of business days to forecast
future_bdays = business_days_between(last_date, horizon_days)

future_bdays = pd.to_datetime(future_bdays).date

n_future = len(future_bdays)
if n_future == 0:
    st.error("Forecast horizon resulted in zero business days. Increase horizon.")
    st.stop()



future_forecast_transformed = fit_full.forecast(steps=n_future)

# Convert to numpy array to drop ARIMA's own index
future_forecast_values = future_forecast_transformed.to_numpy()

if use_log:
    future_forecast = np.exp(future_forecast_values)
else:
    future_forecast = future_forecast_values

future_df = pd.Series(future_forecast, index=future_bdays)
future_df.index.name = 'Date'
future_df.name = 'Forecasted Close'

fig4, ax4 = plt.subplots(figsize=(12,5))
ax4.plot(df.index[-25:], df['Close'].values[-25:], label='Recent actual', color='black')
ax4.plot([df.index[-1], future_df.index[0]], [float(df['Close'].iloc[-1]), float(future_df.iloc[0])], color='black', linestyle='--')
ax4.plot(future_df.index, future_df.values, label=f'Forecast next {horizon_days} days', color='green', marker='o')
ax4.set_title("Future forecast")
ax4.legend()
st.pyplot(fig4)

st.subheader("Future forecast values")
st.dataframe(future_df)

# -------------------------
# Download CSV
# -------------------------

st.subheader("Download results")
out_df = pd.DataFrame({
    'test_date': test.index,
    'actual_test': test[ticker],
    'predicted_test': pred_series.values
})
future_out = pd.DataFrame({'future_date': future_df.index, 'future_predicted': future_df.values})

csv_buf = io.StringIO()
out_df.to_csv(csv_buf, index=False)
csv_data = csv_buf.getvalue().encode('utf-8')

csv_buf2 = io.StringIO()
future_out.to_csv(csv_buf2, index=False)
csv_future_data = csv_buf2.getvalue().encode('utf-8')

st.download_button("Download test predictions CSV", data=csv_data, file_name=f"{ticker}_test_predictions.csv", mime="text/csv")
st.download_button("Download future forecast CSV", data=csv_future_data, file_name=f"{ticker}_future_forecast.csv", mime="text/csv")

st.success("Done — model trained & evaluated. Scroll up for outputs and charts.")