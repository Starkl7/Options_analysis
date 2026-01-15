import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import polars as pl

period = "1mo"
interval = "1h"
aapl = yf.Ticker("AAPL")
aapl_stock = aapl.history(period=period, interval=interval)

csv_file = './options/doltdump/option_chain.csv'
pq_file = './options/doltdump/option_chain.parquet'

opt_chain = aapl.option_chain(aapl.options[0])

stock_data = {
    "aapl": { 
        "stock": aapl_stock,
        "period": period,
        "interval": interval
    }
}

options_data = {
    "aapl": {
        "option_chain": opt_chain,
        "expiration_date": aapl.options[0]
    }
}

def get_hist_options(ticker):
    df = pl.scan_parquet(pq_file)
    filtered_df = df.filter(pl.col("act_symbol") == ticker).collect()
    return filtered_df

def get_data(ticker):
    return {
        "stock": stock_data.get(ticker),
        "curr_options": options_data.get(ticker),
        "hist_options": get_hist_options(ticker.upper())
    }

# def get_hist_options(ticker):
#     return " Data retrieval function placeholder"