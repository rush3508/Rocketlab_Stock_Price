# Rocket Lab Stock Price Analysis
## Overview
This jpynb aims to analyze the relationship between Rocket Lab's Electron launch activities and its stock price performance. The analysis includes scraping and processing data on Rocket Lab's Electron launches, historical stock prices, and the NASDAQ index. We perform correlation analysis to understand how successful and failed launches impact Rocket Lab's stock price. Additionally, we incorporate financial metrics from Rocket Lab's quarterly reports to provide a comprehensive overview of the factors influencing its stock performance. Visualizations are created to illustrate the trends and correlations, offering insights into the financial and operational dynamics of Rocket Lab.

## Electron Launch Data Scraper
Python script that scrapes the list of Electron launches from the Wikipedia page and processes the data into a clean CSV file. It uses requests, BeautifulSoup, and pandas for web scraping and data manipulation.

## Incorporate Rocket Lab Stock Price, Nasdaq Index Price, Launch Success/Failure to Study Any Correlation

* Acquire NASDAQ Data: Download historical NASDAQ index data.
* Merge NASDAQ Data with Existing Data: Merge the NASDAQ index data with the existing merged DataFrame.
* Analyze Correlation: Calculate and analyze the correlation between the NASDAQ index and Rocket Lab's stock price changes.

# Rocket Lab Stock Price Prediction Using LTSM
## Overview
This jpynb aims to use LTSM to predict stock price movement of RKLB.
