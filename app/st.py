import math
import streamlit as st
import os
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import mplfinance as mpf
from binance.client import Client
from binance import Client
from volatility import fig
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')

client = Client(API_KEY, API_SECRET)
st.set_page_config(page_title='Crypto Analysis',
                   layout='wide', page_icon=':dollar:')
st.experimental_memo(persist='disk')
st.title('Crypto Analysis')
st.markdown('---')
st.subheader('Top Volatile Symbols')
st.pyplot(fig)
# c1, c2, c3 = st.columns([1, 1, 1])
# with c1:
# with c2:
# with c3:
st.markdown('---')
st.subheader(f'Technical Analysis')

st.sidebar.subheader('Settings')
st.sidebar.caption('Adjust charts settings and then press apply')

with st.sidebar.form('settings_form'):
    symbol = st.selectbox('Choose stock symbol', options=[
                          'BTC', 'ETH', 'FTM', 'SOL'], index=1)
    date_from = st.date_input('Show data from', date(2022, 1, 1))
    show_data = st.checkbox('Show data table', False)

    # show_nontrading_days = st.checkbox('Show non-trading days', True)
    # https://github.com/matplotlib/mplfinance/blob/master/examples/styles.ipynb
    chart_styles = [
        'default', 'binance', 'blueskies', 'brasil',
        'charles', 'checkers', 'classic', 'yahoo',
        'mike', 'nightclouds', 'sas', 'starsandstripes'
    ]
    chart_style = st.selectbox(
        'Chart style', options=chart_styles, index=chart_styles.index('yahoo'))
    chart_types = [
        'candle', 'ohlc', 'line', 'renko'
    ]
    chart_type = st.selectbox(
        'Chart type', options=chart_types, index=chart_types.index('candle'))

    mav1 = int(st.number_input('Mav 1', min_value=1,
                               max_value=255, value=33, step=1))
    mav2 = int(st.number_input('Mav 2', min_value=1,
                               max_value=255, value=55, step=1))

    st.form_submit_button('Apply')


# Getting historical data
historical_data = client.get_historical_klines(
    f'{symbol}USDT', Client.KLINE_INTERVAL_12HOUR, str(date_from), datetime.date(datetime.now()).strftime('%m-%d-%y'))
df = pd.DataFrame(historical_data)
df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
              'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']
df['Open Time'] = pd.to_datetime(df['Open Time']/1000, unit='s')
df['Close Time'] = pd.to_datetime(df['Close Time']/1000, unit='s')
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                   'Quote Asset Volume', 'TB Base Volume', 'TB Quote Volume']
df[numeric_columns] = df[numeric_columns].apply(
    pd.to_numeric, axis=1)
df['MA 33'] = df['Close'].rolling(mav1).mean()
df['MA 55'] = df['Close'].rolling(mav2).mean()

# Crossover signal calculation
previous_33 = df['MA 33'].shift(1)
previous_55 = df['MA 55'].shift(1)
crossing_down = ((df['MA 33'] <= df['MA 55'])
                 & (previous_33 >= previous_55))
crossing_up = ((df['MA 33'] >= df['MA 55'])
               & (previous_33 <= previous_55))
crossing_down_dates = df.loc[crossing_down, 'Close Time']
crossing_up_dates = df.loc[crossing_up, 'Close Time']

df['Short Signals'] = crossing_down_dates
df['Long Signals'] = crossing_up_dates
df['Long Signals'] = df['Long Signals'].astype(
    'int').where(df['Long Signals'].notnull(), np.nan)
df['Short Signals'] = df['Short Signals'].astype(
    'int').where(df['Short Signals'].notnull(), np.nan)

# MACD
exp12 = df['Close'].ewm(span=12, adjust=False).mean()
exp26 = df['Close'].ewm(span=26, adjust=False).mean()
macd = exp12 - exp26
signal = macd.ewm(span=9, adjust=False).mean()
histogram = macd - signal

# Plot
signals = [
    mpf.make_addplot(df['Long Signals'], type='bar',
                     color='g',  y_on_right=False),
    mpf.make_addplot(df['Short Signals'], type='bar',
                     color='r', y_on_right=False),
    # mpf.make_addplot(exp12,color='lime'),
    # mpf.make_addplot(exp26,color='c'),
    mpf.make_addplot(signal, panel=1, color='red', secondary_y=True),
    mpf.make_addplot(histogram, type='bar', width=0.7, panel=1,
                     color='dimgray', alpha=1, secondary_y=False),
    mpf.make_addplot(macd, panel=1, color='green', secondary_y=True)
]
# style  = mpf.make_mpf_style(base_mpl_style='yahoo')
# mpf.plot(tdf,addplot=apd)

renko_kwargs = dict(type=chart_type, style=chart_style, volume=True, mav=(mav1, mav2), tight_layout=True,
                    datetime_format='%b-%d-%Y', returnfig=True)
chart_kwargs = renko_kwargs | dict(
    volume_panel=2, panel_ratios=(6, 3, 2), addplot=signals)
if chart_type == 'renko' or chart_type == 'pnf':
    fig, ax = mpf.plot(df.set_index('Close Time'), **renko_kwargs)
else:
    fig, ax = mpf.plot(df.set_index('Close Time'), **chart_kwargs)


with st.container():
    st.pyplot(fig)


df_signals = df.loc[(df['Long Signals'] > 0)
                    | (df['Short Signals'] > 0)]
# df_signals.style.format({'Close Time': lambda t: t.strftime("%m/%d/%Y")})
# pd.to_datetime(hist_df['Open Time']/1000, unit='s')
pd.options.mode.chained_assignment = None
df_signals['Signals'] = ['Long' if not math.isnan(
    s) else 'Short' for s in df_signals['Long Signals']]
# df_signals['Signals'] = ['Short' if s != np.nan else '' for s in df_signals['Short Signals']]

df_signals["Time"] = pd.to_datetime(
    df_signals["Close Time"]).dt.strftime('%b %d %Y')
signals_data = df_signals[['Time', 'Open',
                           'Close', 'Number of Trades', 'Signals']]
# result.reset_index(drop=True)

if show_data:
    st.markdown('---')
    st.dataframe(signals_data.reset_index(drop=True))
