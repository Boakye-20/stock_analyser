#!/usr/bin/env python3
"""
Professional Stock Analysis Dashboard
Rewritten using matplotlib tutorial best practices
- Clean GridSpec layout
- Proper font sizing
- Professional spacing
- Tutorial-style tick control
"""

# Import libraries following tutorial order
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from twelvedata import TDClient
import argparse
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set universal matplotlib parameters (tutorial best practice)
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['pdf.fonttype'] = 42  # For Illustrator compatibility

# Your Twelve Data API key
API_KEY = '375b83e1748244b8978b170a98761eee'

class ProfessionalStockAnalyzer:
    """Stock analyzer following matplotlib tutorial best practices"""
    
    def __init__(self, ticker, period='2y'):
        self.ticker = ticker.upper()
        self.period = period
        self.td = TDClient(apikey=API_KEY)
        
        # Define sector context
        self.sector_context = {
            'JPM': 'Banking', 'GS': 'Investment Banking', 'MS': 'Wealth Management',
            'BAC': 'Retail Banking', 'WFC': 'Commercial Banking', 'BLK': 'Asset Management',
            'V': 'Payments', 'MA': 'Payments', 'AAPL': 'Technology', 
            'MSFT': 'Software', 'GOOGL': 'Technology', 'SPY': 'S&P 500 ETF'
        }
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {self.ticker} - {self.sector_context.get(self.ticker, 'Equity')}")
        print(f"Period: {period} | API: Twelve Data (800 calls/day)")
        print(f"{'='*60}\n")
        
        self.fetch_data()
        
    def fetch_data(self):
        """Fetch data following tutorial's clean approach"""
        try:
            # Convert period to days
            period_days = {
                '1mo': 30, '3mo': 90, '6mo': 180,
                '1y': 365, '2y': 730, '5y': 1825
            }.get(self.period, 730)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            print("Fetching price data...")
            ts = self.td.time_series(
                symbol=self.ticker,
                interval="1day",
                outputsize=min(period_days, 5000),
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            self.data = ts.as_pandas()
            
            if self.data.empty:
                print(f"Error: No data returned for {self.ticker}")
                sys.exit(1)
                
            # Clean column names
            self.data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.data = self.data.sort_index()
            
            print(f"Success: Loaded {len(self.data)} days of data")
            
            # Fetch current quote
            self.fetch_quote()
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            sys.exit(1)
            
    def fetch_quote(self):
        """Fetch current quote data"""
        try:
            quote = self.td.quote(symbol=self.ticker).as_json()
            self.current_price = float(quote.get('close', 0))
            self.change = float(quote.get('change', 0))
            self.percent_change = float(quote.get('percent_change', 0))
        except:
            self.current_price = self.data['Close'].iloc[-1]
            self.change = 0
            self.percent_change = 0
            
    def calculate_indicators(self):
        """Calculate indicators following tutorial's clean style"""
        # Moving averages
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA200'] = self.data['Close'].rolling(window=200).mean()
        
        # Bollinger Bands
        rolling_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = self.data['MA20'] + (rolling_std * 2)
        self.data['BB_Lower'] = self.data['MA20'] - (rolling_std * 2)
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-10)
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Returns
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(30).std() * np.sqrt(252)
        
    def create_dashboard(self, save_path=None):
        """Create dashboard following tutorial's GridSpec approach"""
        self.calculate_indicators()
        
        # Create figure with square aspect ratio (tutorial preference)
        fig = plt.figure(figsize=(18, 22))
        
        # Use GridSpec for precise layout control
        gs = gridspec.GridSpec(4, 2, 
                              height_ratios=[1, 1, 1, 1.5],
                              hspace=0.35,  # Tutorial recommends 0.2-0.4
                              wspace=0.25)
        
        # Get colors from seaborn (tutorial best practice)
        colors = sns.color_palette("rocket", 6)
        
        # Define consistent styling
        title_size = 12
        label_size = 11
        tick_size = 10
        legend_size = 10
        
        # Common tick parameters (from tutorial)
        def style_axis(ax, title, ylabel=None, xlabel=None):
            """Apply consistent styling to axis"""
            ax.set_title(title, fontsize=title_size, fontweight='bold', pad=15)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=label_size)
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=label_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_size, length=10)
            ax.tick_params(axis='both', which='minor', labelsize=tick_size, length=5)
            ax.tick_params(direction='in', top=True, right=True)
            ax.grid(True, alpha=0.3)
            
        # Panel 1: Price & Moving Averages
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.data.index, self.data['Close'], label='Close', 
                linewidth=2.5, color=colors[0])
        ax1.plot(self.data.index, self.data['MA20'], label='MA20', 
                linewidth=1.5, color=colors[2], alpha=0.8)
        ax1.plot(self.data.index, self.data['MA50'], label='MA50', 
                linewidth=1.5, color=colors[3], alpha=0.8)
        ax1.plot(self.data.index, self.data['MA200'], label='MA200', 
                linewidth=1.5, color=colors[4], alpha=0.8)
        
        style_axis(ax1, f'{self.ticker} - Price & Moving Averages', 'Price ($)')
        ax1.legend(loc='best', fontsize=legend_size)
        
        # Manual x-axis formatting (tutorial approach)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Panel 2: Bollinger Bands
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.data.index, self.data['Close'], label='Close', 
                linewidth=2, color=colors[0])
        ax2.plot(self.data.index, self.data['BB_Upper'], '--', 
                label='Upper Band', color='red', alpha=0.7)
        ax2.plot(self.data.index, self.data['BB_Lower'], '--', 
                label='Lower Band', color='green', alpha=0.7)
        ax2.fill_between(self.data.index, self.data['BB_Lower'], 
                        self.data['BB_Upper'], alpha=0.1, color='gray')
        
        style_axis(ax2, f'{self.ticker} - Bollinger Bands', 'Price ($)')
        ax2.legend(loc='best', fontsize=legend_size)
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Panel 3: Volume (with color coding like tutorial)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Color bars based on price change
        colors_vol = []
        for i in range(len(self.data)):
            if i == 0:
                colors_vol.append('gray')
            else:
                if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                    colors_vol.append('green')
                else:
                    colors_vol.append('red')
                    
        ax3.bar(self.data.index, self.data['Volume'], color=colors_vol, 
                alpha=0.7, width=0.8)
        
        # Add volume moving average
        vol_ma = self.data['Volume'].rolling(20).mean()
        ax3.plot(self.data.index, vol_ma, color='blue', 
                linewidth=2, label='20-day MA')
        
        style_axis(ax3, f'{self.ticker} - Volume Analysis', 'Volume')
        ax3.legend(loc='best', fontsize=legend_size)
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Format y-axis for millions
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
        
        # Panel 4: RSI
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.data.index, self.data['RSI'], linewidth=2, color=colors[0])
        ax4.axhline(70, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax4.axhline(30, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax4.fill_between(self.data.index, 30, 70, alpha=0.1, color='blue')
        
        style_axis(ax4, f'{self.ticker} - RSI', 'RSI')
        ax4.set_ylim(0, 100)
        
        # Manual y-ticks (tutorial approach)
        ax4.set_yticks(np.arange(0, 101, 20))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add labels
        ax4.text(0.02, 0.95, 'Overbought', transform=ax4.transAxes, 
                fontsize=9, color='red', va='top')
        ax4.text(0.02, 0.15, 'Oversold', transform=ax4.transAxes, 
                fontsize=9, color='green')
        
        # Panel 5: MACD
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(self.data.index, self.data['MACD'], label='MACD', 
                linewidth=2, color='blue')
        ax5.plot(self.data.index, self.data['Signal'], label='Signal', 
                linewidth=2, color='red')
        ax5.bar(self.data.index, self.data['MACD'] - self.data['Signal'], 
                label='Histogram', alpha=0.3, color='gray')
        ax5.axhline(0, color='black', linewidth=0.5)
        
        style_axis(ax5, f'{self.ticker} - MACD', 'MACD')
        ax5.legend(loc='best', fontsize=legend_size)
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Panel 6: Returns Distribution
        ax6 = fig.add_subplot(gs[2, 1])
        returns = self.data['Returns'].dropna()
        
        # Histogram with KDE overlay (tutorial style)
        n, bins, patches = ax6.hist(returns, bins=50, alpha=0.7, 
                                   color=colors[3], edgecolor='black', 
                                   density=True)
        
        # Add normal distribution overlay
        from scipy import stats
        x = np.linspace(returns.min(), returns.max(), 100)
        ax6.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 
                'r-', linewidth=2, label='Normal')
        
        # Add vertical lines for mean and median
        ax6.axvline(returns.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {returns.mean():.4f}')
        ax6.axvline(returns.median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {returns.median():.4f}')
        
        style_axis(ax6, f'{self.ticker} - Returns Distribution', 
                  'Density', 'Daily Return')
        ax6.legend(loc='best', fontsize=legend_size)
        
        # Panel 7: Metrics Table (spanning full width)
        ax_table = fig.add_subplot(gs[3, :])
        ax_table.axis('off')
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Create table data
        table_data = []
        for i, (key, value) in enumerate(metrics.items()):
            if i % 2 == 0 and i > 0:
                table_data.append(['', ''])  # Empty row for spacing
            table_data.append([key, value])
        
        # Create table with better spacing
        table = ax_table.table(cellText=table_data,
                              colLabels=['Metric', 'Value'],
                              cellLoc='left',
                              loc='center',
                              colWidths=[0.5, 0.3])
        
        # Style table (tutorial approach)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.0)
        
        # Color styling
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')
                    # Highlight important metrics
                    if i > 0 and j == 0:
                        text = table_data[i-1][0]
                        if any(k in text for k in ['Current', 'Total Return', 'Sharpe']):
                            cell.set_text_props(weight='bold')
        
        # Main title
        fig.suptitle(f'{self.ticker} - {self.sector_context.get(self.ticker, "Equity")} Analysis',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save figure
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nDashboard saved to: {os.path.abspath(save_path)}")
        
        plt.show()
        
    def calculate_metrics(self):
        """Calculate financial metrics"""
        returns = self.data['Returns'].dropna()
        
        if len(returns) < 2:
            return {'Error': 'Insufficient data'}
            
        # Calculate metrics
        total_return = (self.data['Close'].iloc[-1] / self.data['Close'].iloc[0] - 1) * 100
        
        trading_days = len(self.data)
        years = trading_days / 252
        annualized_return = ((self.data['Close'].iloc[-1] / self.data['Close'].iloc[0]) ** (1/years) - 1)
        
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.04) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative / peak) - 1
        max_drawdown = drawdown.min()
        
        # VaR
        var_95 = np.percentile(returns, 5)
        
        return {
            'Current Price': f'${self.current_price:.2f}',
            'Change Today': f'{self.percent_change:+.2f}%',
            'Total Return': f'{total_return:.1f}%',
            'Annualized Return': f'{annualized_return:.1%}',
            'Volatility': f'{volatility:.1%}',
            'Sharpe Ratio': f'{sharpe_ratio:.2f}',
            'Max Drawdown': f'{max_drawdown:.1%}',
            'VaR (95%)': f'{var_95:.2%}',
            '52W Range': f'${self.data["Low"].min():.2f} - ${self.data["High"].max():.2f}'
        }

def main():
    """Main function following tutorial structure"""
    parser = argparse.ArgumentParser(
        description='Professional Stock Analysis Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--period', default='2y', 
                       choices=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                       help='Analysis period (default: 2y)')
    parser.add_argument('--save', help='Save path for dashboard')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ProfessionalStockAnalyzer(args.ticker, args.period)
    
    # Generate save path if not provided
    if args.save:
        save_path = args.save
    else:
        save_path = f'{args.ticker}_analysis_{datetime.now().strftime("%Y%m%d")}.png'
    
    # Create dashboard
    analyzer.create_dashboard(save_path)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Professional Stock Analysis Dashboard
Rewritten using matplotlib tutorial best practices
- Clean GridSpec layout
- Proper font sizing
- Professional spacing
- Tutorial-style tick control
"""

# Import libraries following tutorial order
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from twelvedata import TDClient
import argparse
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set universal matplotlib parameters (tutorial best practice)
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['pdf.fonttype'] = 42  # For Illustrator compatibility

# Your Twelve Data API key
API_KEY = '375b83e1748244b8978b170a98761eee'

class ProfessionalStockAnalyzer:
    """Stock analyzer following matplotlib tutorial best practices"""
    
    def __init__(self, ticker, period='2y'):
        self.ticker = ticker.upper()
        self.period = period
        self.td = TDClient(apikey=API_KEY)
        
        # Define sector context
        self.sector_context = {
            'JPM': 'Banking', 'GS': 'Investment Banking', 'MS': 'Wealth Management',
            'BAC': 'Retail Banking', 'WFC': 'Commercial Banking', 'BLK': 'Asset Management',
            'V': 'Payments', 'MA': 'Payments', 'AAPL': 'Technology', 
            'MSFT': 'Software', 'GOOGL': 'Technology', 'SPY': 'S&P 500 ETF'
        }
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {self.ticker} - {self.sector_context.get(self.ticker, 'Equity')}")
        print(f"Period: {period} | API: Twelve Data (800 calls/day)")
        print(f"{'='*60}\n")
        
        self.fetch_data()
        
    def fetch_data(self):
        """Fetch data following tutorial's clean approach"""
        try:
            # Convert period to days
            period_days = {
                '1mo': 30, '3mo': 90, '6mo': 180,
                '1y': 365, '2y': 730, '5y': 1825
            }.get(self.period, 730)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            print("Fetching price data...")
            ts = self.td.time_series(
                symbol=self.ticker,
                interval="1day",
                outputsize=min(period_days, 5000),
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            self.data = ts.as_pandas()
            
            if self.data.empty:
                print(f"Error: No data returned for {self.ticker}")
                sys.exit(1)
                
            # Clean column names
            self.data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.data = self.data.sort_index()
            
            print(f"Success: Loaded {len(self.data)} days of data")
            
            # Fetch current quote
            self.fetch_quote()
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            sys.exit(1)
            
    def fetch_quote(self):
        """Fetch current quote data"""
        try:
            quote = self.td.quote(symbol=self.ticker).as_json()
            self.current_price = float(quote.get('close', 0))
            self.change = float(quote.get('change', 0))
            self.percent_change = float(quote.get('percent_change', 0))
        except:
            self.current_price = self.data['Close'].iloc[-1]
            self.change = 0
            self.percent_change = 0
            
    def calculate_indicators(self):
        """Calculate indicators following tutorial's clean style"""
        # Moving averages
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA200'] = self.data['Close'].rolling(window=200).mean()
        
        # Bollinger Bands
        rolling_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = self.data['MA20'] + (rolling_std * 2)
        self.data['BB_Lower'] = self.data['MA20'] - (rolling_std * 2)
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-10)
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Returns
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(30).std() * np.sqrt(252)
        
    def create_dashboard(self, save_path=None):
        """Create dashboard following tutorial's GridSpec approach"""
        self.calculate_indicators()
        
        # Create figure with square aspect ratio (tutorial preference)
        fig = plt.figure(figsize=(18, 22))
        
        # Use GridSpec for precise layout control
        gs = gridspec.GridSpec(4, 2, 
                              height_ratios=[1, 1, 1, 1.5],
                              hspace=0.35,  # Tutorial recommends 0.2-0.4
                              wspace=0.25)
        
        # Get colors from seaborn (tutorial best practice)
        colors = sns.color_palette("rocket", 6)
        
        # Define consistent styling
        title_size = 12
        label_size = 11
        tick_size = 10
        legend_size = 10
        
        # Common tick parameters (from tutorial)
        def style_axis(ax, title, ylabel=None, xlabel=None):
            """Apply consistent styling to axis"""
            ax.set_title(title, fontsize=title_size, fontweight='bold', pad=15)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=label_size)
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=label_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_size, length=10)
            ax.tick_params(axis='both', which='minor', labelsize=tick_size, length=5)
            ax.tick_params(direction='in', top=True, right=True)
            ax.grid(True, alpha=0.3)
            
        # Panel 1: Price & Moving Averages
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.data.index, self.data['Close'], label='Close', 
                linewidth=2.5, color=colors[0])
        ax1.plot(self.data.index, self.data['MA20'], label='MA20', 
                linewidth=1.5, color=colors[2], alpha=0.8)
        ax1.plot(self.data.index, self.data['MA50'], label='MA50', 
                linewidth=1.5, color=colors[3], alpha=0.8)
        ax1.plot(self.data.index, self.data['MA200'], label='MA200', 
                linewidth=1.5, color=colors[4], alpha=0.8)
        
        style_axis(ax1, f'{self.ticker} - Price & Moving Averages', 'Price ($)')
        ax1.legend(loc='best', fontsize=legend_size)
        
        # Manual x-axis formatting (tutorial approach)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Panel 2: Bollinger Bands
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.data.index, self.data['Close'], label='Close', 
                linewidth=2, color=colors[0])
        ax2.plot(self.data.index, self.data['BB_Upper'], '--', 
                label='Upper Band', color='red', alpha=0.7)
        ax2.plot(self.data.index, self.data['BB_Lower'], '--', 
                label='Lower Band', color='green', alpha=0.7)
        ax2.fill_between(self.data.index, self.data['BB_Lower'], 
                        self.data['BB_Upper'], alpha=0.1, color='gray')
        
        style_axis(ax2, f'{self.ticker} - Bollinger Bands', 'Price ($)')
        ax2.legend(loc='best', fontsize=legend_size)
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Panel 3: Volume (with color coding like tutorial)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Color bars based on price change
        colors_vol = []
        for i in range(len(self.data)):
            if i == 0:
                colors_vol.append('gray')
            else:
                if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                    colors_vol.append('green')
                else:
                    colors_vol.append('red')
                    
        ax3.bar(self.data.index, self.data['Volume'], color=colors_vol, 
                alpha=0.7, width=0.8)
        
        # Add volume moving average
        vol_ma = self.data['Volume'].rolling(20).mean()
        ax3.plot(self.data.index, vol_ma, color='blue', 
                linewidth=2, label='20-day MA')
        
        style_axis(ax3, f'{self.ticker} - Volume Analysis', 'Volume')
        ax3.legend(loc='best', fontsize=legend_size)
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Format y-axis for millions
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
        
        # Panel 4: RSI
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.data.index, self.data['RSI'], linewidth=2, color=colors[0])
        ax4.axhline(70, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax4.axhline(30, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax4.fill_between(self.data.index, 30, 70, alpha=0.1, color='blue')
        
        style_axis(ax4, f'{self.ticker} - RSI', 'RSI')
        ax4.set_ylim(0, 100)
        
        # Manual y-ticks (tutorial approach)
        ax4.set_yticks(np.arange(0, 101, 20))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add labels
        ax4.text(0.02, 0.95, 'Overbought', transform=ax4.transAxes, 
                fontsize=9, color='red', va='top')
        ax4.text(0.02, 0.15, 'Oversold', transform=ax4.transAxes, 
                fontsize=9, color='green')
        
        # Panel 5: MACD
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(self.data.index, self.data['MACD'], label='MACD', 
                linewidth=2, color='blue')
        ax5.plot(self.data.index, self.data['Signal'], label='Signal', 
                linewidth=2, color='red')
        ax5.bar(self.data.index, self.data['MACD'] - self.data['Signal'], 
                label='Histogram', alpha=0.3, color='gray')
        ax5.axhline(0, color='black', linewidth=0.5)
        
        style_axis(ax5, f'{self.ticker} - MACD', 'MACD')
        ax5.legend(loc='best', fontsize=legend_size)
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Panel 6: Returns Distribution
        ax6 = fig.add_subplot(gs[2, 1])
        returns = self.data['Returns'].dropna()
        
        # Histogram with KDE overlay (tutorial style)
        n, bins, patches = ax6.hist(returns, bins=50, alpha=0.7, 
                                   color=colors[3], edgecolor='black', 
                                   density=True)
        
        # Add normal distribution overlay
        from scipy import stats
        x = np.linspace(returns.min(), returns.max(), 100)
        ax6.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 
                'r-', linewidth=2, label='Normal')
        
        # Add vertical lines for mean and median
        ax6.axvline(returns.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {returns.mean():.4f}')
        ax6.axvline(returns.median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {returns.median():.4f}')
        
        style_axis(ax6, f'{self.ticker} - Returns Distribution', 
                  'Density', 'Daily Return')
        ax6.legend(loc='best', fontsize=legend_size)
        
        # Panel 7: Metrics Table (spanning full width)
        ax_table = fig.add_subplot(gs[3, :])
        ax_table.axis('off')
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Create table data
        table_data = []
        for i, (key, value) in enumerate(metrics.items()):
            if i % 2 == 0 and i > 0:
                table_data.append(['', ''])  # Empty row for spacing
            table_data.append([key, value])
        
        # Create table with better spacing
        table = ax_table.table(cellText=table_data,
                              colLabels=['Metric', 'Value'],
                              cellLoc='left',
                              loc='center',
                              colWidths=[0.5, 0.3])
        
        # Style table (tutorial approach)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.0)
        
        # Color styling
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')
                    # Highlight important metrics
                    if i > 0 and j == 0:
                        text = table_data[i-1][0]
                        if any(k in text for k in ['Current', 'Total Return', 'Sharpe']):
                            cell.set_text_props(weight='bold')
        
        # Main title
        fig.suptitle(f'{self.ticker} - {self.sector_context.get(self.ticker, "Equity")} Analysis',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save figure
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nDashboard saved to: {os.path.abspath(save_path)}")
        
        plt.show()
        
    def calculate_metrics(self):
        """Calculate financial metrics"""
        returns = self.data['Returns'].dropna()
        
        if len(returns) < 2:
            return {'Error': 'Insufficient data'}
            
        # Calculate metrics
        total_return = (self.data['Close'].iloc[-1] / self.data['Close'].iloc[0] - 1) * 100
        
        trading_days = len(self.data)
        years = trading_days / 252
        annualized_return = ((self.data['Close'].iloc[-1] / self.data['Close'].iloc[0]) ** (1/years) - 1)
        
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.04) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative / peak) - 1
        max_drawdown = drawdown.min()
        
        # VaR
        var_95 = np.percentile(returns, 5)
        
        return {
            'Current Price': f'${self.current_price:.2f}',
            'Change Today': f'{self.percent_change:+.2f}%',
            'Total Return': f'{total_return:.1f}%',
            'Annualized Return': f'{annualized_return:.1%}',
            'Volatility': f'{volatility:.1%}',
            'Sharpe Ratio': f'{sharpe_ratio:.2f}',
            'Max Drawdown': f'{max_drawdown:.1%}',
            'VaR (95%)': f'{var_95:.2%}',
            '52W Range': f'${self.data["Low"].min():.2f} - ${self.data["High"].max():.2f}'
        }

def main():
    """Main function following tutorial structure"""
    parser = argparse.ArgumentParser(
        description='Professional Stock Analysis Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--period', default='2y', 
                       choices=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                       help='Analysis period (default: 2y)')
    parser.add_argument('--save', help='Save path for dashboard')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ProfessionalStockAnalyzer(args.ticker, args.period)
    
    # Generate save path if not provided
    if args.save:
        save_path = args.save
    else:
        save_path = f'{args.ticker}_analysis_{datetime.now().strftime("%Y%m%d")}.png'
    
    # Create dashboard
    analyzer.create_dashboard(save_path)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
