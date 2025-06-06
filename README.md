# Stock Analyzer Dashboard 📈

A professional stock analysis tool that generates comprehensive technical analysis dashboards using matplotlib and real-time market data.

## 🚀 Features

- **Real-time Data**: Fetches live market data using Twelve Data API
- **Technical Indicators**:
  - Moving Averages (20, 50, 200-day)
  - Bollinger Bands
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Volume Analysis
- **Statistical Analysis**:
  - Returns distribution
  - Volatility calculations
  - Sharpe ratio
  - Maximum drawdown
  - Value at Risk (VaR)
- **Professional Visualization**: High-quality dashboard output with customizable time periods

## 📊 Sample Output

The dashboard generates a comprehensive multi-panel analysis:
- Price action with moving averages
- Bollinger Bands analysis
- Volume trends with color coding
- RSI momentum indicator
- MACD trend analysis
- Returns distribution histogram
- Key metrics summary table

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Boakye-20/stock-analyzer-dashboard.git
cd stock-analyzer-dashboard
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Get your free API key from [Twelve Data](https://twelvedata.com/) (800 calls/day free tier)

4. Update the API key in `stock_analyzer.py`:
```python
API_KEY = 'your_api_key_here'
```

## 💻 Usage

Basic usage:
```bash
python stock_analyzer.py AAPL
```

With custom time period:
```bash
python stock_analyzer.py MSFT --period 1y
```

Save output to specific location:
```bash
python stock_analyzer.py GOOGL --period 6mo --save reports/google_analysis.png
```

### Available Time Periods
- `1mo` - 1 month
- `3mo` - 3 months
- `6mo` - 6 months
- `1y` - 1 year
- `2y` - 2 years (default)
- `5y` - 5 years

## 📋 Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
twelvedata>=1.2.0
scipy>=1.7.0
```

## 🔧 Configuration

The analyzer supports various financial instruments:
- US Stocks (AAPL, MSFT, GOOGL, etc.)
- ETFs (SPY, QQQ, etc.)
- Banking sector stocks (JPM, BAC, GS, etc.)

## 📈 Technical Details

### Indicators Explained

- **Moving Averages**: Trend identification using 20, 50, and 200-day periods
- **Bollinger Bands**: Volatility measurement with 2 standard deviations
- **RSI**: Momentum oscillator (14-day period)
- **MACD**: Trend-following momentum indicator (12, 26, 9 periods)

### Metrics Calculated

- **Total Return**: Overall performance for the selected period
- **Annualized Return**: Year-over-year growth rate
- **Volatility**: Annualized standard deviation
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **VaR (95%)**: Value at Risk at 95% confidence level

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Paul Kwarteng**

- LinkedIn: [Paul Kwarteng](https://www.linkedin.com/in/paul-kwarteng-22a71b196/)
- GitHub: [@Boakye-20](https://github.com/Boakye-20)

## 🙏 Acknowledgments

- [Twelve Data](https://twelvedata.com/) for providing the market data API
- Matplotlib documentation for visualization best practices
- The Python finance community for inspiration

---

⭐ If you find this project useful, please consider giving it a star!
