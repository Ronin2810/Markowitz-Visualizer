import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

st.title("Markowitz Model Visualizer")

# Step 1: Input number of assets
n_assets = st.number_input("Enter the number of assets:", min_value=2, value=2, step=1)

# Step 2: Input expected returns
st.header("Expected Returns")
expected_returns = []
for i in range(n_assets):
    ret = st.number_input(f"Return for asset {i+1}:", value=0.1, step=0.01, format="%.4f", key=f"return_{i}")
    expected_returns.append(ret)

# Step 3: Input variance-covariance matrix
st.header("Variance-Covariance Matrix")
cov_matrix = np.zeros((n_assets, n_assets))

matrix_input = []
for i in range(n_assets):
    row = []
    cols = st.columns(n_assets)
    for j in range(n_assets):
        value = cols[j].number_input(f"Cov({i+1},{j+1})", value=0.01, step=0.01, format="%.4f", key=f"cov_{i}_{j}")
        row.append(value)
    matrix_input.append(row)

cov_matrix = np.array(matrix_input)
expected_returns = np.array(expected_returns)

# Step 4: Allow short selling toggle
allow_short_selling = st.checkbox("Allow Short Selling", value=False)

# Step 5: Input risk-free rate
risk_free_rate = st.number_input("Enter the risk-free rate:", value=0.01, step=0.01, format="%.4f")

# Step 6: Show Capital Market Line (CML) toggle
show_cml = st.checkbox("Show Capital Market Line (CML)", value=False)

def portfolio_performance(weights, expected_returns, cov_matrix):
    returns = np.sum(weights * expected_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    p_returns, p_risk = portfolio_performance(weights, expected_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_risk

def max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate, allow_short_selling):
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    if allow_short_selling:
        bounds = tuple((-1, 1) for asset in range(num_assets))
    else:
        bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets*[1./num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def min_variance(expected_returns, cov_matrix, allow_short_selling):
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    if allow_short_selling:
        bounds = tuple((-1, 1) for asset in range(num_assets))
    else:
        bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(portfolio_variance, num_assets*[1./num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_variance(weights, expected_returns, cov_matrix):
    return portfolio_performance(weights, expected_returns, cov_matrix)[1]**2

def plot_efficient_frontier(expected_returns, cov_matrix, num_portfolios, risk_free_rate, allow_short_selling, show_cml):
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        if allow_short_selling:
            weights = np.random.uniform(-1, 1, len(expected_returns))
        else:
            weights = np.random.random(len(expected_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_std_dev = portfolio_performance(weights, expected_returns, cov_matrix)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev

    max_sharpe = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate, allow_short_selling)
    ms_weights = max_sharpe['x']
    ms_returns, ms_risk = portfolio_performance(ms_weights, expected_returns, cov_matrix)

    min_vol = min_variance(expected_returns, cov_matrix, allow_short_selling)
    mv_weights = min_vol['x']
    mv_returns, mv_risk = portfolio_performance(mv_weights, expected_returns, cov_matrix)

    fig = go.Figure()
    
    # Scatter plot for all portfolios
    fig.add_trace(go.Scatter(
        x=results[1, :],
        y=results[0, :],
        mode='markers',
        marker=dict(
            color=results[2, :],
            colorbar=dict(title='Sharpe Ratio'),
            colorscale='Viridis',
            size=5
        ),
        name='Portfolios'
    ))

    # Max Sharpe Ratio Portfolio
    fig.add_trace(go.Scatter(
        x=[ms_risk],
        y=[ms_returns],
        mode='markers',
        marker=dict(color='red', size=10, symbol='star'),
        name='Max Sharpe Ratio'
    ))

    # Minimum Volatility Portfolio
    fig.add_trace(go.Scatter(
        x=[mv_risk],
        y=[mv_returns],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='star'),
        name='Minimum Volatility'
    ))

    # Capital Market Line (CML)
    if show_cml:
        cml_x = np.linspace(0, max(results[1, :]), 100)
        cml_y = risk_free_rate + (cml_x / ms_risk) * (ms_returns - risk_free_rate)
        fig.add_trace(go.Scatter(
            x=cml_x,
            y=cml_y,
            mode='lines',
            name='Capital Market Line (CML)',
            line=dict(color='green', dash='dash')
        ))

    fig.update_layout(
        title='Efficient Frontier',
        xaxis=dict(title='Volatility (Std. Deviation)'),
        yaxis=dict(title='Return'),
        showlegend=True
    )

    st.plotly_chart(fig)

# Display the efficient frontier if inputs are valid
if len(expected_returns) == n_assets and cov_matrix.shape == (n_assets, n_assets):
    num_portfolios = 10000  # Number of portfolios to simulate
    plot_efficient_frontier(expected_returns, cov_matrix, num_portfolios, risk_free_rate, allow_short_selling, show_cml)
