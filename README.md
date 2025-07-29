# Time Varying Equity Premium  
#### Birju Patel

This project explores the time-varying nature of the equity risk premium using international data. I use the [Jordà-Schularick-Taylor Macrohistory Database](https://www.macrohistory.net/data/), which contains more than a century of macro-financial data across 16 developed countries.

The work is inspired by **Campbell and Shiller (1998), "Stock Prices, Earnings, and Expected Dividends"**. I replicate their finding on international data. Below are the key empirical findings.

- Smoothed dividend-price (D/P) ratios have strong predictive power on future long-term equity returns.
- D/P ratios have no predictive power on future long-term dividend growth.

This contradicts the hypothesis that equity market prices contain rational expectations of future dividend growth.

### Regression Summary

| Model                                 | R²     | Sample Size |
|--------------------------------------|--------|--------------|
| Forward Real Dividend Growth         | 0.048  | 142          |
| Forward Average Total Returns        | 0.196  | 142          |
