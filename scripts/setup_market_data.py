from src.utils.common_imports import *
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
            from src.data.real_market_data_integration import RealMarketDataIntegration
            from datetime import datetime, timedelta
from src.data.real_market_data_integration import RealMarketDataIntegration
from datetime import datetime, timedelta

#!/usr/bin/env python3
"""
Market Data Setup Script for QuantAI Trading Platform.

This script helps set up real market data integration
with multiple data sources and API keys.
"""


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = setup_logger()


class MarketDataSetup:
    """Setup and configure market data sources for QuantAI Trading Platform."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_file = self.project_root / "config" / "market_data_config.json"
        self.api_keys = {}
        
    def setup_environment(self):
        """Set up the environment for market data integration."""
        print("🔧 Setting up market data environment...")
        
        # Create necessary directories
        directories = [
            "config",
            "data",
            "logs",
            "cache",
            "results"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
        
        # Create .env file if it doesn't exist
        env_file = self.project_root / ".env"
        if not env_file.exists():
            self._create_env_file()
            print("✅ Created .env file")
        
        print("✅ Environment setup completed!")
    
    def _create_env_file(self):
        """Create .env file with API key placeholders."""
        env_content = """# QuantAI Trading Platform - Environment Variables
# Add your API keys here

# Yahoo Finance (FREE - No API key required)
# YFINANCE_ENABLED=true

# Alpha Vantage (FREE - 500 calls/day)
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Finnhub (FREE - 60 calls/minute)
# FINNHUB_API_KEY=your_finnhub_key_here

# Twelve Data (FREE - 800 calls/day)
# TWELVE_DATA_API_KEY=your_twelve_data_key_here

# IEX Cloud (FREE - 500,000 calls/month)
# IEX_API_KEY=your_iex_key_here

# News API (FREE - 1000 requests/month)
# NEWS_API_KEY=your_news_api_key_here

# Gemini API (for sentiment analysis)
# GEMINI_API_KEY=your_gemini_key_here

# Database
# DATABASE_URL=sqlite:///quantai.db
# REDIS_URL=redis://localhost:6379

# Logging
# LOG_LEVEL=INFO
# LOG_FILE=logs/quantai.log
"""
        
        env_file = self.project_root / ".env"
        with open(env_file, 'w') as f:
            f.write(env_content)
    
    def install_dependencies(self):
        """Install required dependencies for market data integration."""
        print("📦 Installing market data dependencies...")
        
        requirements_file = self.project_root / "requirements_market_data.txt"
        
        if requirements_file.exists():
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True)
                print("✅ Dependencies installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error installing dependencies: {e}")
                return False
        else:
            print("❌ requirements_market_data.txt not found!")
            return False
        
        return True
    
    def get_api_keys(self):
        """Get API keys from user input."""
        print("\n🔑 API Key Setup")
        print("=" * 50)
        print("Enter your API keys (press Enter to skip):")
        
        api_keys = {}
        
        # Yahoo Finance (no key needed)
        print("\n✅ Yahoo Finance - FREE (no API key required)")
        
        # Alpha Vantage
        alpha_key = input("\n🔹 Alpha Vantage API Key (FREE - 500 calls/day): ").strip()
        if alpha_key:
            api_keys['alpha_vantage'] = alpha_key
            print("✅ Alpha Vantage key saved")
        
        # Finnhub
        finnhub_key = input("\n🔹 Finnhub API Key (FREE - 60 calls/minute): ").strip()
        if finnhub_key:
            api_keys['finnhub'] = finnhub_key
            print("✅ Finnhub key saved")
        
        # Twelve Data
        twelve_key = input("\n🔹 Twelve Data API Key (FREE - 800 calls/day): ").strip()
        if twelve_key:
            api_keys['twelve_data'] = twelve_key
            print("✅ Twelve Data key saved")
        
        # IEX Cloud
        iex_key = input("\n🔹 IEX Cloud API Key (FREE - 500,000 calls/month): ").strip()
        if iex_key:
            api_keys['iex'] = iex_key
            print("✅ IEX Cloud key saved")
        
        # News API
        news_key = input("\n🔹 News API Key (FREE - 1000 requests/month): ").strip()
        if news_key:
            api_keys['news_api'] = news_key
            print("✅ News API key saved")
        
        # Gemini API
        gemini_key = input("\n🔹 Gemini API Key (for sentiment analysis): ").strip()
        if gemini_key:
            api_keys['gemini'] = gemini_key
            print("✅ Gemini API key saved")
        
        self.api_keys = api_keys
        return api_keys
    
    def save_config(self):
        """Save configuration to file."""
        config = {
            "data_sources": {
                "yfinance": {
                    "enabled": True,
                    "priority": 1,
                    "description": "Yahoo Finance - Free historical and real-time data"
                },
                "alpha_vantage": {
                    "enabled": bool(self.api_keys.get('alpha_vantage')),
                    "priority": 2,
                    "api_key": self.api_keys.get('alpha_vantage'),
                    "description": "Alpha Vantage - Technical indicators and fundamental data"
                },
                "finnhub": {
                    "enabled": bool(self.api_keys.get('finnhub')),
                    "priority": 3,
                    "api_key": self.api_keys.get('finnhub'),
                    "description": "Finnhub - Real-time data and news"
                },
                "twelve_data": {
                    "enabled": bool(self.api_keys.get('twelve_data')),
                    "priority": 4,
                    "api_key": self.api_keys.get('twelve_data'),
                    "description": "Twelve Data - Multi-asset data"
                },
                "iex": {
                    "enabled": bool(self.api_keys.get('iex')),
                    "priority": 5,
                    "api_key": self.api_keys.get('iex'),
                    "description": "IEX Cloud - Professional market data"
                },
                "news_api": {
                    "enabled": bool(self.api_keys.get('news_api')),
                    "priority": 6,
                    "api_key": self.api_keys.get('news_api'),
                    "description": "News API - News and sentiment data"
                }
            },
            "settings": {
                "min_data_points": 100,
                "max_retries": 3,
                "retry_delay": 1,
                "cache_duration": 3600,  # 1 hour
                "rate_limit_delay": 1.0
            },
            "symbols": {
                "focused_tickers": ["AMZN", "META", "NVDA", "GOOGL", "AAPL"],
                "additional_tickers": ["TSLA", "MSFT", "NFLX", "AMD", "INTC"],
                "sectors": ["Technology", "Communication Services", "Consumer Discretionary"]
            }
        }
        
        # Save to config file
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to {self.config_file}")
    
    def test_data_sources(self):
        """Test data sources to ensure they're working."""
        print("\n🧪 Testing data sources...")
        
        # Test symbols
        test_symbols = ["AAPL", "GOOGL"]
        
        try:
            # Import the real market data integration
            sys.path.append(str(self.project_root))
            
            # Initialize with API keys
            data_integration = RealMarketDataIntegration(
                alpha_vantage_key=self.api_keys.get('alpha_vantage'),
                finnhub_key=self.api_keys.get('finnhub'),
                twelve_data_key=self.api_keys.get('twelve_data'),
                iex_key=self.api_keys.get('iex')
            )
            
            # Test historical data
            print("📊 Testing historical data...")
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            
            historical_data = data_integration.get_historical_data(
                symbols=test_symbols,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
            
            if historical_data:
                print(f"✅ Historical data: {len(historical_data)} symbols")
                for symbol, data in historical_data.items():
                    print(f"   {symbol}: {len(data)} records")
            else:
                print("⚠️ No historical data retrieved")
            
            # Test real-time data
            print("\n🔄 Testing real-time data...")
            real_time_data = data_integration.get_real_time_data(test_symbols)
            
            if real_time_data:
                print(f"✅ Real-time data: {len(real_time_data)} symbols")
                for symbol, data in real_time_data.items():
                    print(f"   {symbol}: ${data['price']:.2f}")
            else:
                print("⚠️ No real-time data retrieved")
            
            print("✅ Data source testing completed!")
            
        except Exception as e:
            print(f"❌ Error testing data sources: {e}")
            return False
        
        return True
    
    def create_sample_usage(self):
        """Create sample usage script."""
        sample_script = '''#!/usr/bin/env python3
"""
Sample usage of QuantAI Real Market Data Integration.
"""


def main():
    # Initialize with your API keys
    data_integration = RealMarketDataIntegration(
        alpha_vantage_key="your_alpha_vantage_key",
        finnhub_key="your_finnhub_key",
        twelve_data_key="your_twelve_data_key",
        iex_key="your_iex_key"
    )
    
    # Define symbols
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    # Get historical data
    print("📊 Fetching historical data...")
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    historical_data = data_integration.get_historical_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval='1d'
    )
    
    # Get real-time data
    print("🔄 Fetching real-time data...")
    real_time_data = data_integration.get_real_time_data(symbols)
    
    # Get news data
    print("📰 Fetching news data...")
    news_data = data_integration.get_news_data(symbols, days=7)
    
    # Generate data quality report
    if historical_data:
        report = data_integration.get_data_quality_report(historical_data)
        print("\\n📈 Data Quality Report:")
        print(json.dumps(report, indent=2, default=str))
    
    print("\\n✅ Sample usage completed!")

if __name__ == "__main__":
    main()
'''
        
        sample_file = self.project_root / "examples" / "market_data_sample.py"
        sample_file.parent.mkdir(exist_ok=True)
        
        with open(sample_file, 'w') as f:
            f.write(sample_script)
        
        print(f"✅ Sample usage script created: {sample_file}")
    
    def run_setup(self):
        """Run the complete setup process."""
        print("🚀 QuantAI Trading Platform - Market Data Setup")
        print("=" * 60)
        
        # Step 1: Setup environment
        self.setup_environment()
        
        # Step 2: Install dependencies
        if not self.install_dependencies():
            print("❌ Setup failed at dependency installation")
            return False
        
        # Step 3: Get API keys
        self.get_api_keys()
        
        # Step 4: Save configuration
        self.save_config()
        
        # Step 5: Test data sources
        if self.test_data_sources():
            print("✅ All data sources tested successfully!")
        else:
            print("⚠️ Some data sources may not be working properly")
        
        # Step 6: Create sample usage
        self.create_sample_usage()
        
        print("\n🎉 Market data setup completed!")
        print("\n📋 Next Steps:")
        print("1. Review your configuration in config/market_data_config.json")
        print("2. Test the sample usage script: python examples/market_data_sample.py")
        print("3. Integrate with your backtesting systems")
        print("4. Start using real market data in your trading strategies")
        
        return True


def main():
    """Main setup function."""
    setup = MarketDataSetup()
    success = setup.run_setup()
    
    if success:
        print("\n✅ Setup completed successfully!")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
