# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
from prophet import Prophet
#from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import ccxt
import datetime
import pandas as pd


START = "2022-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('BTC/USDT', 'AAPL')
symbol = st.selectbox('Select dataset for prediction', stocks)

timeF = ('1h', '4h', '1d')
timeF2 = st.selectbox('Select Timeframe', timeF)

n_years = st.slider('Months of prediction:', 1, 12)
period = n_years * 30


exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': st.secrets["k"],
    'secret': st.secrets["s"],
    'timeout': 30000,
    'enableRateLimit': True,
})
exchange.load_markets()
if exchange.has['fetchOHLCV']:

    ohlcv = exchange.fetch_ohlcv(symbol, timeF2, limit=10000)

D = []
for t in ohlcv:
  lst1 =  [(datetime.datetime.fromtimestamp(int(t[0])/1000).strftime('%Y-%m-%d %H:%M:%S')), t[1], t[2], t[3], t[4], sum(t[1:4])/3]
  D.append(lst1)
df = pd.DataFrame(D, columns =['Date', 'Open', 'High','Low', 'Close', 'VWAP'])
print(df.tail())
		
st.subheader('Raw data')
st.write(df.tail(7))



# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = df[['Date','VWAP']]
df_train = df_train.rename(columns={"Date": "ds", "VWAP": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(7)
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
st.write(fig1)


# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail(7))

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
