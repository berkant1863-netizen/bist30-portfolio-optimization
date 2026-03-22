# BIST-30 Portfolio Optimization

I optimized a portfolio of 10 stocks from the BIST-30 index using the Markowitz mean-variance model. The goal was to get better risk-adjusted returns compared to an equal-weight benchmark.

# Project Structure

I split the project into 5 steps to keep things clean:

- Step 1 : Data: Pull price data and calculate daily returns
- Step 2 :  Risk Metrics: Volatility, VaR, Sharpe, correlation matrix
- Step 3 : Optimization: Markowitz mean-variance, efficient frontier
- Step 4 : Backtesting: 70/30 train-test split, out-of-sample validation
- Step 5 : Binance API: Real-time crypto data and live VaR monitoring

# What does it do?

I pulled 2 years of daily price data for 10 stocks (THYAO, GARAN, AKBNK, EREGL, BIMAS, SISE, KCHOL, SAHOL, ARCLK, TUPRS) from Yahoo Finance. Then step by step:

1. Calculated risk metrics for each stock (volatility, VaR, Sharpe, Max Drawdown)
2. Built a correlation matrix to see how stocks move together
3. Ran Markowitz optimization to find the best weights (max Sharpe portfolio)
4. Plotted the efficient frontier
5. Backtested with a 70/30 train-test split to check if it actually works out-of-sample
6. Pulled real-time crypto data (BTC, ETH) from Binance API and compared risk metrics

# Results

- Equal-weight portfolio Sharpe: -0.03
- Optimized portfolio Sharpe: 0.81
- Risk-free rate: 40% (Turkish market conditions)
- Position limit: max 40% per stock

# Efficient Frontier

[Efficient Frontier](efficient_frontier.png)

# Risk Metrics & Correlation

[Risk Metrics](risk_metrikleri.png)

# Backtesting (Out-of-Sample)

[Backtesting](backtesting.png)

# Files

- `main.py` — runs the full analysis
- `fiyat_verisi.csv` — raw price data
- `log_getiri.csv` — daily log returns
- `risk_metrikleri.csv` — risk metrics table
- `korelasyon.csv` — correlation matrix
- `*.png` — chart outputs

# How to run

```
pip install yfinance pandas numpy matplotlib seaborn scipy python-binance
python main.py
```

## Libraries used

- yfinance (data)
- pandas, numpy (data processing)
- scipy (optimization)
- matplotlib, seaborn (charts)
- python-binance (crypto data)
