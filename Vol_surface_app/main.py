import yfinance as yf
import pandas as pd
import numpy as np
import functions as fn
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import griddata
import os
import py_vollib_vectorized as pvv
import glob

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'DIS']

def pull_stock_data(ticker = 'AAPL'):
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="1mo", interval="5m")
    # targets = stock.analyst_price_targets
    return hist_data

def load_options_data(ticker = 'AAPL', report_date = "2026-01-26 11:00:00"):
    # report_date = "2026-01-26 11:00:00"
    file_path = f"/Volumes/SEAGATE/crondata/{report_date[:10]}"
    # expiry_date = "2026-01-30"
    hhmm = report_date[11:16].replace(":", "")

    # Use glob to find matching files
    pattern = os.path.join(file_path, f"{ticker}_options_{hhmm}*.parquet")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"No options files found matching pattern: {pattern}")
    
    # Use the first matching file (or most recent if multiple)
    options_file = matching_files[0]
    options_data = pd.read_parquet(options_file)
    return options_data

# stock, spot_prices, spot_price = m.parse_stock_data(ticker)
def parse_stock_data(ticker='AAPL', report_date='2026-01-26 11:00:00'):
    hist_data = pull_stock_data(ticker)
    stock_price = hist_data[hist_data.index == f'{report_date}-05:00']
    if stock_price.empty:
        raise ValueError(f"No stock price data found for {ticker} at {report_date}")
    return stock_price['Open'].values[0]

def clean_options_data(df, report_date):
    df = df[df['lastPrice'] > 0]
    df = df[df['volume'] > df['volume'].quantile(0.5)]
    df = df[['contractSymbol', 'lastPrice', 'strike', 'bid', 'ask', 'type', 'impliedVolatility', 'volume']]
    df['type'] = df['type'].map({ 'call': 'c', 'put': 'p' })
    df['expiration'] = pd.to_datetime(df['contractSymbol'].str.extract(r'(\d{6})')[0], format='%y%m%d') + pd.DateOffset(hours=16, minutes=0, seconds=0)
    df['TTE'] = (df['expiration'] - pd.to_datetime(f"{report_date}")).dt.days / 365
    return df

# calls_data, expiration_dates = m.parse_options_data(ticker, stock.index[-1].strftime("%Y-%m-%d %H:%M:%S"))
def parse_options_data(ticker='AAPL', report_date='2026-01-26 11:00:00'):
    options_data = load_options_data(ticker, report_date)
    if options_data is None:
        raise ValueError(f"No options data found for {ticker} at {report_date}")
    options_data_cleaned = clean_options_data(options_data, report_date)
    return options_data_cleaned

def set_thresholds(row, stock_price, risk_free_rate=0.035, strike='strike', TTE='TTE', option_type='type'):
    if row[option_type] == 'c':
        return max(0, stock_price - row[strike] * np.exp(-risk_free_rate * row[TTE]))
    else:
        return max(0, row[strike] * np.exp(-risk_free_rate * row[TTE]) - stock_price)
    
def substitute_outdated_prices(row, last_price='lastPrice', threshold_price='thresholdPrice'):
    if row[last_price] < row[threshold_price]:
        return (row['bid'] + row['ask']) / 2
    else:
        return row[last_price]

# imp_vol_data = m.calculate_implied_volatility(
#     filtered_calls_data,
#     spot_price,
#     model,
#     risk_free_rate,
#     dividend_yield
# )
def calculate_implied_volatility(options_data, stock_price, model='black_scholes', risk_free_rate=0.035, 
                                 dividend_yield=0.0, autofill_outdated_options=False,
                                 last_price='lastPrice', strike='strike', 
                                 TTE='TTE', option_type='type'):
    if options_data is None or options_data.empty:
        raise ValueError("Invalid options data")
    
    #Setup threshold prices for calls and puts (Intrinsic value) and replace by mid-price if last price is below threshold
    if autofill_outdated_options:
        options_data['thresholdPrice'] = options_data.apply(
            lambda row: set_thresholds(row, stock_price, risk_free_rate, strike, TTE, option_type), axis=1)
        options_data['validPrice'] = options_data.apply(
            lambda row: substitute_outdated_prices(row, last_price, 'thresholdPrice'), axis=1)
        last_price = 'validPrice'
    
    options_data[f'{model}_iv'] = pvv.implied_volatility.vectorized_implied_volatility(
        price=options_data[last_price].values,
        S=stock_price,
        K=options_data[strike].values,
        t=options_data[TTE].values,
        r=risk_free_rate,
        q=dividend_yield,
        flag=options_data[option_type].values,
        model=model,
        return_as='numpy'
    )
    return options_data

def get_plot_data(filtered_df, TTE='TTE', strike='strike', iv='black_scholes_iv'):
    X = filtered_df[TTE].values
    Y = filtered_df[strike].values
    Z = filtered_df[iv].values * 100

    return X, Y, Z

def filter_options_data(df, option_type, min_strike_price, max_strike_price):
    if option_type not in ['c', 'p', 'b']:
        raise ValueError("option_type must be 'c' for calls, 'p' for puts, or 'b' for both")
    
    if option_type == 'b':
        types_to_include = ['c', 'p']
    else:
        types_to_include = [option_type]
    filtered_df = df[
        (df['strike'] >= min_strike_price) &
        (df['strike'] <= max_strike_price) &
        (df['type'].isin(types_to_include))
    ]
    return filtered_df.reset_index(drop=True)

# def plot_implied_volatility(X, Y, Z):
    # Optional: Will check back later
