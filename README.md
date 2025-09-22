# factorinvesting

# Monte Carlo Stock Price Simulation
# This Python script implements a Monte Carlo simulation to forecast future stock price movements using geometric Brownian motion. 
# The model downloads historical stock data for a company from 2020-2024, 
# calculates key statistical parameters (daily mean return and volatility), 
# and uses these to simulate 1,000 possible price paths over the next 252 trading days (one year). 
# Each simulation applies random shocks from a normal distribution to model the unpredictable nature 
# of market movements while incorporating the stock's historical drift and volatility patterns.
# The simulation engine runs 1,000 independent forecasts, 
# where each day's price is calculated using the geometric Brownian motion formula: 
# S(t+1) = S(t) × e^[(μ - 0.5σ²) + σ×Z]. 
# This mathematical model captures both the expected return component (drift) 
# and the random volatility component that characterizes real stock price behavior. 
# The code tracks all simulated price paths and stores the final prices from each simulation 
# to analyze the distribution of possible outcomes after one year.
# The results provide valuable risk management insights including the expected final stock price, 
# the 5% Value at Risk (VaR) representing the worst-case scenario for 95% confidence, 
# and the probability of experiencing a loss. Two visualizations are generated: a line chart 
# showing all 1,000 simulated price trajectories, and a histogram displaying the distribution of final prices 
# with reference lines for current price, mean expected price, and VaR threshold. This approach helps investors 
# understand the range of possible outcomes and make more informed decisions based on probabilistic rather than 
# deterministic forecasts.