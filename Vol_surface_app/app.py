import streamlit as st
import main as m
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go

st.title("Implied Volatility Surface Viewer")
st.sidebar.header("User Inputs")

ticker = st.sidebar.selectbox("Select Ticker", m.tickers)

risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value = 0.0, max_value = 1.0, value = .035, format="%.4f")

dividend_yield = st.sidebar.slider("Dividend Yield (%)", min_value = 0.0, max_value = 1.0, value = 0.0, format="%.4f")

option_type = st.sidebar.selectbox("Select Option Type", ['Call', 'Put', 'Both'])
option_type = option_type.lower()[0]

report_date = st.sidebar.text_input("Report Date (YYYY-MM-DD HH:MM:SS)", value="2026-01-26 11:00:00")

model = st.sidebar.selectbox("Select Option Pricing Model", ['black_scholes'])

# autofill_outdated_options = st.sidebar.selectbox("Autofill Outdated Options Data", ['True', 'False'])

# Double check function return parameters
spot_price = m.parse_stock_data(ticker)

# Double check function return parameters
calls_data = m.parse_options_data(ticker, report_date)

dynamic_min_percentage = 20
dynamic_max_percentage = 200
default_min_percentage = 70
default_max_percentage = 130

strike_price_range_percentage = st.sidebar.slider(
    "Strike Price Range (% of Spot Price)", 
    min_value=dynamic_min_percentage, 
    max_value=dynamic_max_percentage, 
    value=(default_min_percentage, default_max_percentage)
)

min_strike_price = spot_price * (strike_price_range_percentage[0] / 100)
max_strike_price = spot_price * (strike_price_range_percentage[1] / 100)

# Double check function return parameters
filtered_options_data = m.filter_options_data(
    calls_data,
    option_type,
    min_strike_price,
    max_strike_price
)

if filtered_options_data.empty:
    st.warning("No options data available for the selected criteria.")
    st.stop()

# Double check function return parameters
imp_vol_data = m.calculate_implied_volatility(
    filtered_options_data,
    spot_price,
    model,
    risk_free_rate,
    dividend_yield,
    # autofill_outdated_options
)

if imp_vol_data.empty:
    st.warning("Implied volatility calculation failed for the selected criteria.")
    st.stop()

X = imp_vol_data['TTE'].values
Z = imp_vol_data[f'{model}_iv'].values * 100

# if option_type == 'Strike Price':
Y = imp_vol_data['strike'].values
y_label = 'Strike Price ($)'
# elif option_type == 'Moneyness':
#     T = imp_vol_data['TTE'].values
#     F = spot_price * np.exp((risk_free_rate - dividend_yield) * T)
#     F = np.maximum(F, 1e-12)

#     imp_vol_data['LogMoneyness'] = np.log(imp_vol_data['strike'].values / F)
#     Y = imp_vol_data['LogMoneyness'].values
#     y_label = 'Log Moneyness ln(K/F)'

if len(np.unique(X)) < 2 or len(np.unique(Y)) < 2:
    st.warning("Not enough unique data points to create a surface plot.")
    st.stop()

xi = np.linspace(X.min(), X.max(), 50)
yi = np.linspace(Y.min(), Y.max(), 50)
X_grid, Y_grid = np.meshgrid(xi, yi)

zi = griddata((X, Y), Z, (X_grid, Y_grid), method='linear')
zi2 = griddata((X, Y), Z, (X_grid, Y_grid), method='nearest')

zi = np.where(np.isnan(zi), zi2, zi)

fig = go.Figure(
    data=[go.Surface(z=zi, x=X_grid, y=Y_grid, colorscale='turbo')]
)
fig.update_layout(
    title=f'Implied Volatility Surface for {ticker}',
    scene = dict(
        xaxis_title='Time to Expiry (Years)',
        yaxis_title=y_label,
        zaxis_title='Implied Volatility (%)',
    ),
    autosize=True,
    width=1000,
    height=800,
    margin=dict(l=65, r=50, b=65, t=90)
)

st.plotly_chart(fig)