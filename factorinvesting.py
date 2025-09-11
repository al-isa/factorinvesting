import pandas as pd
import numpy as np
import seaborn
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
#from pandas_datareader import data as pdr
from datetime import datetime
tickers = ['AMZN']
#rate limit
#'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V', 'XOM', 'UNH'

start_date = '2022-01-01'
end_date = '2024-12-31'

price_data = yf.download(tickers, start=start_date, end=end_date)
price_data = price_data['Close']

spy_data = yf.download('SPY', start=start_date, end=end_date)
spy_data = spy_data['Close']
spy_returns = spy_data.pct_change().fillna(0)
spy_cumulative = (1 + spy_returns).cumprod()


monthly_prices = price_data.resample('ME').last()

returns = monthly_prices.pct_change()
volatility = returns.rolling(window=12).std()

returns_list = []
dates = []

for i in range(12, len(monthly_prices)-1):
    date = monthly_prices.index[i]
    next_date = monthly_prices.index[i + 1]


    current_prices = monthly_prices.iloc[i]
    prev_prices = monthly_prices.iloc[i - 12]
    next_prices = monthly_prices.iloc[i + 1]

    #factor scores
    momentum = (current_prices - prev_prices) / prev_prices
    value = 1/current_prices #inverse price
    size = -volatility.iloc[i]

    #create rankings; low pe and caps are good, high momentum is good
    ranks = pd.DataFrame({
        'momentum': momentum.rank(ascending=False),
        'value': value.rank(ascending=True),
        'size': size.rank(ascending=True)
    })

    ranks['score'] = ranks.mean(axis=1)
    top3 = ranks.nsmallest(3, 'score').index

    #calcultae return over next month
    ret = (next_prices[top3] - current_prices[top3]) /current_prices[top3]
    portfolio_return = ret.mean()

    returns_list.append(portfolio_return)
    dates.append(next_date)

results = pd.DataFrame({'Date': dates, 'Return': returns_list})
results['Cumulative'] = (1 + results['Return']).cumprod()

cum_return = (1 + returns).cumprod()
running_max = cum_return.cummax()
drawdown = cum_return / running_max - 1


# Plot the equity curve
plt.figure(figsize=(14,6))
plt.plot(results['Date'], results['Cumulative'], label=tickers[0])
plt.plot(spy_cumulative.index, spy_cumulative, label='SPY Benchmark')
plt.title('Factor Investing Strategy: Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
# Annotate last point
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

'''
plt.figure(figsize=(14,5))
plt.plot(drawdown, color='red', label='Drawdown')
plt.title('Drawdown Curve')
plt.legend()
plt.grid()
'''


# Print performance metrics
total_return = results['Cumulative'].iloc[-1] - 1
annualized_return = (1 + total_return) ** (12 / len(results)) - 1
sharpe = np.mean(results['Return']) / np.std(results['Return']) * np.sqrt(12)
print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
