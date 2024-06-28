import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from plotly import graph_objs as go

START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App - Prophet Model')

stocks = ('CVS', 'GM', 'UAL', 'F', 'DAL')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    # Ensure the Date column is datetime and other columns are numeric
    data['Date'] = pd.to_datetime(data['Date'])
    data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Handle missing values by filling them with the previous valid value
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Predict forecast with Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Ensure forecast data is clean
forecast['ds'] = pd.to_datetime(forecast['ds'])
forecast['yhat'] = pd.to_numeric(forecast['yhat'], errors='coerce')
forecast['yhat_lower'] = pd.to_numeric(forecast['yhat_lower'], errors='coerce')
forecast['yhat_upper'] = pd.to_numeric(forecast['yhat_upper'], errors='coerce')

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write('Forecast plot')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

st.write("Change Points Plot - Marking Trends")
fig3 = m.plot(forecast)
a = add_changepoints_to_plot(fig3.gca(), m, forecast)
st.pyplot(fig3)

st.write("Cross-Validation Performance Metrics")
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')
df_p = performance_metrics(df_cv)
st.write(df_p.head())

st.write("Cross-Validation MAPE Plot")
fig4 = plot_cross_validation_metric(df_cv, metric='mape')
st.pyplot(fig4)
