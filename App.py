import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
from datetime import datetime, timedelta
import ta
import streamlit as st
import time



# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

# Define default values
default_ticker = 'AAPL'
default_interval = '30m'
interval_options = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1d', '5d', '1wk', '1mo', '3mo']

# Streamlit app title
st.title("Stock Trend Prediction")

# User input for the stock ticker
ticker = st.text_input('Enter Stock Ticker', default_ticker)

# Date range selection
today = datetime.today()
start_date = st.slider("Start Date", min_value=today - timedelta(days=365), max_value=today, value=today - timedelta(days=30), format="YYYY-MM-DD")
end_date = st.slider("End Date", min_value=start_date, max_value=today, value=today, format="YYYY-MM-DD")

# Time interval selection
interval = st.selectbox("Select Time Interval", interval_options, index=interval_options.index(default_interval))

# Download the data using the user-inputted ticker, date range, and interval
data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

# Display raw data
st.subheader("Raw Data")
st.write(data)

# Display statistical summary
st.subheader("Statistical Data")
st.write(data.describe())

# Trend line selection
trend_option = st.selectbox("Select Trend Line Option", ['Close', 'Open', 'High', 'Low'])

# Visualization
st.subheader(f"{trend_option} Price Trend Over Time")
st.line_chart(data[trend_option])

# Optional: Use matplotlib for a customized plot
fig, ax = plt.subplots()
ax.plot(data.index, data[trend_option], label=trend_option)
ax.set_xlabel('Date')
ax.set_ylabel('Price')
fig.suptitle(f"{ticker} {trend_option} Price Trend Over Time")

# Format date
date_format = mpl_dates.DateFormatter('%d-%m-%Y %H:%M')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

# Calculate technical indicators
data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
macd = ta.trend.MACD(data['Close'])
data['macd'] = macd.macd()
data['macd_signal'] = macd.macd_signal()
data['macd_diff'] = macd.macd_diff()
stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
data['stoch_k'] = stoch.stoch()
data['stoch_d'] = stoch.stoch_signal()
data['adx'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
data['cci'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
data['roc'] = ta.momentum.ROCIndicator(data['Close']).roc()
data['williamsr'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
bbands = ta.volatility.BollingerBands(data['Close'])
data['bbands_upper'] = bbands.bollinger_hband()
data['bbands_middle'] = bbands.bollinger_mavg()
data['bbands_lower'] = bbands.bollinger_lband()
data['psar'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
data['ema'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
data['sma'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()

# Calculate CMO manually
def CMO(data, period=14):
    diff = data.diff(1)
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    sum_gain = gain.rolling(window=period).sum()
    sum_loss = loss.rolling(window=period).sum()
    cmo = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)
    return cmo

data['cmo'] = CMO(data['Close'])

kc = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'])
data['kc_upper'] = kc.keltner_channel_hband()
data['kc_middle'] = kc.keltner_channel_mband()
data['kc_lower'] = kc.keltner_channel_lband()

vwap = ta.volume.VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
data['vwap'] = vwap.vwap

# Manually calculate TEMA
def TEMA(series, window):
    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

# Calculate TEMA
window = 30  # Example window period
data['tema'] = TEMA(data['Close'], window)

data['mfi'] = ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume']).money_flow_index()
data['fi'] = ta.volume.ForceIndexIndicator(data['Close'], data['Volume']).force_index()
data['adi'] = ta.volume.AccDistIndexIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()
data['obv'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
data['eom'] = ta.volume.EaseOfMovementIndicator(data['High'], data['Low'], data['Close'], data['Volume']).ease_of_movement()
data['dpo'] = ta.trend.DPOIndicator(data['Close']).dpo()

# Calculate DMI
dmi = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=14)
data['adx'] = dmi.adx()
data['dmi_pos'] = dmi.adx_pos()
data['dmi_neg'] = dmi.adx_neg()

# Initialize empty lists for buy, sell, and neutral counts
buy_counts = []
sell_counts = []
neutral_counts = []

# Determine buy, sell, or neutral for each indicator
for index, row in data.iterrows():
    buy = 0
    sell = 0
    neutral = 0
    
    # RSI
    if row['rsi'] < 30:
        buy += 1
    elif row['rsi'] > 70:
        sell += 1
    else:
        neutral += 1
    
    # MACD
    if row['macd'] > row['macd_signal']:
        buy += 1
    elif row['macd'] < row['macd_signal']:
        sell += 1
    else:
        neutral += 1
    
    # Stochastic Oscillator
    if row['stoch_k'] < 20:
        buy += 1
    elif row['stoch_k'] > 80:
        sell += 1
    else:
        neutral += 1
    
    # ADX
    if row['adx'] > 25:
        neutral += 1  # ADX > 25 is a trend indicator, no direct buy/sell signal
    
    # CCI
    if row['cci'] < -100:
        buy += 1
    elif row['cci'] > 100:
        sell += 1
    else:
        neutral += 1
    
    # ROC
    if row['roc'] > 0:
        buy += 1
    elif row['roc'] < 0:
        sell += 1
    else:
        neutral += 1
    
    # Williams %R
    if row['williamsr'] < -80:
        buy += 1
    elif row['williamsr'] > -20:
        sell += 1
    else:
        neutral += 1
    
    # Bollinger Bands
    if row['Close'] < row['bbands_lower']:
        buy += 1
    elif row['Close'] > row['bbands_upper']:
        sell += 1
    else:
        neutral += 1
    
    # PSAR
    if row['Close'] > row['psar']:
        buy += 1
    elif row['Close'] < row['psar']:
        sell += 1
    else:
        neutral += 1
    
    # EMA
    if row['Close'] > row['ema']:
        buy += 1
    elif row['Close'] < row['ema']:
        sell += 1
    else:
        neutral += 1
    
    # SMA
    if row['Close'] > row['sma']:
        buy += 1
    elif row['Close'] < row['sma']:
        sell += 1
    else:
        neutral += 1
    
    # CMO
    if row['cmo'] < -50:
        buy += 1
    elif row['cmo'] > 50:
        sell += 1
    else:
        neutral += 1
    
    # Keltner Channel
    if row['Close'] < row['kc_lower']:
        buy += 1
    elif row['Close'] > row['kc_upper']:
        sell += 1
    else:
        neutral += 1
    
    # VWAP
    if row['Close'] > row['vwap']:
        buy += 1
    elif row['Close'] < row['vwap']:
        sell += 1
    else:
        neutral += 1
    
    # TEMA
    if row['Close'] > row['tema']:
        buy += 1
    elif row['Close'] < row['tema']:
        sell += 1
    else:
        neutral += 1
    
    # MFI
    if row['mfi'] < 20:
        buy += 1
    elif row['mfi'] > 80:
        sell += 1
    else:
        neutral += 1
    
    # Force Index
    if row['fi'] > 0:
        buy += 1
    elif row['fi'] < 0:
        sell += 1
    else:
        neutral += 1
    
    # Accumulation/Distribution Index
    if row['adi'] > 0:
        buy += 1
    elif row['adi'] < 0:
        sell += 1
    else:
        neutral += 1
    
    # On Balance Volume
    if row['obv'] > data['obv'].shift(1)[index]:
        buy += 1
    elif row['obv'] < data['obv'].shift(1)[index]:
        sell += 1
    else:
        neutral += 1
    
    # Ease of Movement
    if row['eom'] > 0:
        buy += 1
    elif row['eom'] < 0:
        sell += 1
    else:
        neutral += 1
    
    # Detrended Price Oscillator
    if row['dpo'] > 0:
        buy += 1
    elif row['dpo'] < 0:
        sell += 1
    else:
        neutral += 1
    
    # Directional Movement Index
    if row['dmi_pos'] > row['dmi_neg']:
        buy += 1
    elif row['dmi_pos'] < row['dmi_neg']:
        sell += 1
    else:
        neutral += 1
        
    buy_counts.append(buy)
    sell_counts.append(sell)
    neutral_counts.append(neutral)

# Create results DataFrame
results = pd.DataFrame({
    'datetime': data.index,
    'closing price': data['Close'],
    'Buy': buy_counts,
    'Sell': sell_counts,
    'Neutral': neutral_counts
})

# Display the results in Streamlit
st.subheader("Stock Predictions")
st.write(results)
