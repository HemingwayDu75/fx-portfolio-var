# fx-portfolio-var
Python tool for calculating FX portfolio Value-at-Risk (VaR) using historical spot data, EWMA volatility, and delta exposure from options and forwards.

FX Portfolio Value-at-Risk Simulation
This Python tool calculates the Value at Risk (VaR) of an FX options and forwards portfolio based on imported historical spot data and portfolio positions. It integrates exponential weighted volatility (EWMA), delta exposure, and statistical confidence intervals to quantify FX market risk.

Objective

  To compute the 1-day parametric VaR of a portfolio expressed in a risk currency (e.g., USD) and converted to a functional currency (e.g., EUR), taking into account:

    Option delta sensitivity

      Directional exposure of forward and option positions
      
      Market volatility derived from historical spot prices
      
      User-defined confidence level and interest rate inputs

Input Files

  SPOT File: Historical exchange rates including at least columns: date, clot
    
    Portfolio File: Position data including at least columns: Amount1, Fwd/Strike, Type, Direction, Trade Date, ValueDate, Ccy1, Ccy2

   Input via graphical interface: Both files are selected using file dialog boxes (Tkinter).

Methodology

    Log returns computed from historical spot data
    
    Volatility calculated via EWMA (lambda = 0.97)
    
    Delta per option based on Black-Scholes-like formula (adjusted for FX)
    
    Delta-equivalent exposure aggregated across the portfolio
    
    1-day VaR = z-score × volatility × √(horizon)
    
    Result converted to functional currency using the latest spot

Customizable Parameters in the Script

      ini
      Copier
      Modifier
      confidence_level = 0.95
      time_horizon = 1  # in days
      risk_currency = 'USD'
      functional_currency = 'EUR'
      risk_free_rate_risk_currency = 0.05
      risk_free_rate_functional_currency = 0.0335
Requirements

      Python 3.8+
      
      Libraries:
      
      pandas
      
      numpy
      
      matplotlib
      
      seaborn
      
      scipy

      tkinter (included by default)

To install dependencies:


      pip install pandas numpy matplotlib seaborn scipy

How to Run

      Save the script as fx_var_calculator.py
      
      Execute the script in a Python environment:
      
      python fx_var_calculator.py

Select the historical spot data file, then the portfolio file

Results are printed in the terminal, including:

      Current volatility
      
      Delta-adjusted exposure
      
      VaR in both currencies
      
      Position-level VaR breakdown
      
      Top 5 risk-contributing positions

Output Summary

Console report with aggregate risk indicators

Position-level VaR and delta

Top 5 positions by absolute delta exposure
