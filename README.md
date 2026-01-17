Data source - yfinance
- For historical option chain -
  Cronjobs have been setup for pulling yfinance option chain data every 15 mins (9:00 - 17:00 24*5) from 19/1/26
  Script needs a bit more optimization, making an API call for all tickers at once instead of calls seperately for each individual ticker.

Heston, Merton and Hull-White require parameter calibration to market data. Should be easier once the cronjobs for yfinance are up and running.

Heston, Merton and Hull-White require Monte Carlo simulations to model the stochastic processes.
Heston models volatility as a stochastic process.
Merton extends Black-Scholes by adding jumps, hence no closed form solution.
Hull-White model implementation needs to be improved to model stochastic interest rates, and then combined with Heston's stochastic volatility.

For Monte-Carlo simulations, 10^4 sims and 10^2 steps
- hist_sigma : Standard deviation of returns scaled to annual rate
- r          : risk-free interest rate (.035)

For models,
  - Binomial Tree - 10^2 time steps till expiry. In Q-measure, up factor = e^(sigma.t) and down factor = e^(-sigma.t)
  - Black-Scholes - Closed form solution, hence no MC necessary. Volatility (hist_sigma) is assumed constant.
  - Heston - Models volatility as a stochastic process. 
    Initial params = {
                        kappa : 2.0
                        theta : hist_sigma ^ 2
                        sigma : 0.1
                        v0    : hist_sigma ^ 2
                        rho   : -0.7
                      }
  - Merton Jump-Diffusion - Extends Black-Scholes to include jumps.
    Initial params = {
                        lambda  : 0.1
                        mu      : -0.2
                        delta   : 0.1
                        sigma   : hist_sigma
                      }
  - Hull-White - Implementation needs to be revised. Originally, should model interest rate as a stochastic process but right now, is similar to Heston model.
