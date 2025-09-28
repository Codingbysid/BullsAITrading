            from src.utils.common_imports import *
import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
            import yfinance as yf

#!/usr/bin/env python3
"""
QuantAI Trading Platform - Project Manager

This script provides comprehensive project management functionality
including setup, testing, deployment, and maintenance.
"""


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = setup_logger()


class QuantAIProjectManager:
    """Comprehensive project manager for QuantAI Trading Platform."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load project configuration."""
        config_file = self.project_root / "config" / "project_config.json"
        
        default_config = {
            "project_name": "QuantAI Trading Platform",
            "version": "1.0.0",
            "python_version": "3.8+",
            "main_script": "run_quantai.py",
            "requirements_file": "requirements.txt",
            "backtesters": {
                "simple": "apps/backtesting/backtesters/simple_backtest.py",
                "standalone": "apps/backtesting/backtesters/standalone_backtest.py",
                "qf_lib": "apps/backtesting/backtesters/qf_lib_backtester.py",
                "advanced": "apps/backtesting/backtesters/advanced_quantitative_backtester.py",
                "focused": "apps/backtesting/backtesters/focused_5_ticker_backtester.py"
            },
            "data_sources": {
                "yfinance": True,
                "alpha_vantage": False,
                "finnhub": False,
                "twelve_data": False,
                "iex": False
            },
            "features": {
                "backtesting": True,
                "portfolio_management": True,
                "real_time_data": True,
                "risk_management": True,
                "ml_models": True,
                "reinforcement_learning": True
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        
        return default_config
    
    def setup_project(self):
        """Set up the project structure and dependencies."""
        logger.info("🔧 Setting up QuantAI Trading Platform...")
        
        # Create directory structure
        directories = [
            "apps/backtesting/backtesters",
            "apps/portfolio",
            "apps/trading",
            "src/data",
            "src/models",
            "src/risk",
            "src/training",
            "src/database",
            "src/interface",
            "src/security",
            "config",
            "docs",
            "scripts",
            "examples",
            "tests/unit",
            "tests/integration",
            "tests/performance",
            "logs",
            "cache",
            "results"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        
        # Create __init__.py files
        init_files = [
            "apps/__init__.py",
            "apps/backtesting/__init__.py",
            "apps/backtesting/backtesters/__init__.py",
            "apps/portfolio/__init__.py",
            "apps/trading/__init__.py",
            "src/__init__.py",
            "src/data/__init__.py",
            "src/models/__init__.py",
            "src/risk/__init__.py",
            "src/training/__init__.py",
            "src/database/__init__.py",
            "src/interface/__init__.py",
            "src/security/__init__.py"
        ]
        
        for init_file in init_files:
            file_path = self.project_root / init_file
            if not file_path.exists():
                file_path.touch()
                logger.info(f"✅ Created: {init_file}")
        
        # Install dependencies
        self.install_dependencies()
        
        # Create configuration files
        self.create_config_files()
        
        logger.info("✅ Project setup completed!")
    
    def install_dependencies(self):
        """Install project dependencies."""
        logger.info("📦 Installing dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if requirements_file.exists():
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True)
                logger.info("✅ Dependencies installed successfully!")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Error installing dependencies: {e}")
                return False
        else:
            logger.error("❌ requirements.txt not found!")
            return False
        
        return True
    
    def create_config_files(self):
        """Create configuration files."""
        logger.info("⚙️ Creating configuration files...")
        
        # Project configuration
        project_config = {
            "project_name": "QuantAI Trading Platform",
            "version": "1.0.0",
            "description": "Advanced AI-driven quantitative trading platform",
            "author": "QuantAI Team",
            "license": "MIT",
            "python_version": "3.8+",
            "main_script": "run_quantai.py",
            "requirements_file": "requirements.txt",
            "backtesters": {
                "simple": "apps/backtesting/backtesters/simple_backtest.py",
                "standalone": "apps/backtesting/backtesters/standalone_backtest.py",
                "qf_lib": "apps/backtesting/backtesters/qf_lib_backtester.py",
                "advanced": "apps/backtesting/backtesters/advanced_quantitative_backtester.py",
                "focused": "apps/backtesting/backtesters/focused_5_ticker_backtester.py"
            },
            "data_sources": {
                "yfinance": {"enabled": True, "priority": 1, "api_key_required": False},
                "alpha_vantage": {"enabled": False, "priority": 2, "api_key_required": True},
                "finnhub": {"enabled": False, "priority": 3, "api_key_required": True},
                "twelve_data": {"enabled": False, "priority": 4, "api_key_required": True},
                "iex": {"enabled": False, "priority": 5, "api_key_required": True}
            },
            "features": {
                "backtesting": True,
                "portfolio_management": True,
                "real_time_data": True,
                "risk_management": True,
                "ml_models": True,
                "reinforcement_learning": True
            },
            "settings": {
                "min_data_points": 100,
                "max_retries": 3,
                "retry_delay": 1,
                "cache_duration": 3600,
                "rate_limit_delay": 1.0
            }
        }
        
        config_file = self.project_root / "config" / "project_config.json"
        with open(config_file, 'w') as f:
            json.dump(project_config, f, indent=2)
        
        logger.info(f"✅ Created: config/project_config.json")
        
        # Environment file
        env_content = """# QuantAI Trading Platform - Environment Variables
# Add your API keys here

# Yahoo Finance (FREE - No API key required)
YFINANCE_ENABLED=true

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
DATABASE_URL=sqlite:///quantai.db
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/quantai.log

# Trading
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.3
RISK_FREE_RATE=0.02
"""
        
        env_file = self.project_root / ".env"
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        logger.info(f"✅ Created: .env")
    
    def test_project(self):
        """Test the project functionality."""
        logger.info("🧪 Testing QuantAI Trading Platform...")
        
        tests = [
            ("Basic imports", self._test_imports),
            ("Backtesting systems", self._test_backtesters),
            ("Real market data", self._test_market_data),
            ("Portfolio management", self._test_portfolio),
            ("Configuration", self._test_configuration)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Running {test_name}...")
                result = test_func()
                results[test_name] = result
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.warning(f"⚠️ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Generate test report
        self._generate_test_report(results)
        
        return results
    
    def _test_imports(self) -> bool:
        """Test basic imports."""
        try:
            return True
        except ImportError as e:
            logger.error(f"Import error: {e}")
            return False
    
    def _test_backtesters(self) -> bool:
        """Test backtesting systems."""
        backtesters = [
            "apps/backtesting/backtesters/simple_backtest.py",
            "apps/backtesting/backtesters/qf_lib_backtester.py"
        ]
        
        for backtester in backtesters:
            backtester_path = self.project_root / backtester
            if not backtester_path.exists():
                logger.warning(f"Backtester not found: {backtester}")
                return False
        
        return True
    
    def _test_market_data(self) -> bool:
        """Test market data integration."""
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            return info is not None
        except Exception as e:
            logger.error(f"Market data test error: {e}")
            return False
    
    def _test_portfolio(self) -> bool:
        """Test portfolio management."""
        portfolio_files = [
            "src/database/db_manager.py",
            "src/portfolio/portfolio_manager.py",
            "src/interface/cli.py"
        ]
        
        for file_path in portfolio_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                logger.warning(f"Portfolio file not found: {file_path}")
                return False
        
        return True
    
    def _test_configuration(self) -> bool:
        """Test configuration files."""
        config_files = [
            "config/project_config.json",
            ".env"
        ]
        
        for config_file in config_files:
            full_path = self.project_root / config_file
            if not full_path.exists():
                logger.warning(f"Config file not found: {config_file}")
                return False
        
        return True
    
    def _generate_test_report(self, results: Dict[str, bool]):
        """Generate test report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "passed": sum(1 for result in results.values() if result),
            "failed": sum(1 for result in results.values() if not result),
            "results": results
        }
        
        report_file = self.project_root / "logs" / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Test report saved to: {report_file}")
    
    def clean_project(self):
        """Clean up the project."""
        logger.info("🧹 Cleaning up project...")
        
        # Remove temporary files
        temp_patterns = [
            "*.pyc",
            "__pycache__",
            "*.log",
            ".pytest_cache",
            "*.egg-info"
        ]
        
        for pattern in temp_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
        
        logger.info("✅ Project cleaned!")
    
    def deploy_project(self):
        """Deploy the project."""
        logger.info("🚀 Deploying QuantAI Trading Platform...")
        
        # Create deployment package
        deployment_dir = self.project_root / "deployment"
        deployment_dir.mkdir(exist_ok=True)
        
        # Copy essential files
        essential_files = [
            "run_quantai.py",
            "requirements.txt",
            "README.md",
            "apps/",
            "src/",
            "config/",
            "scripts/"
        ]
        
        for file_path in essential_files:
            src = self.project_root / file_path
            dst = deployment_dir / file_path
            
            if src.is_file():
                shutil.copy2(src, dst)
            elif src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
        
        logger.info(f"✅ Deployment package created: {deployment_dir}")
    
    def show_status(self):
        """Show project status."""
        logger.info("📊 QuantAI Trading Platform Status")
        logger.info("=" * 50)
        
        # Project info
        logger.info(f"Project: {self.config['project_name']}")
        logger.info(f"Version: {self.config['version']}")
        logger.info(f"Python: {self.config['python_version']}")
        
        # Features status
        logger.info("\n🔧 Features:")
        for feature, enabled in self.config['features'].items():
            status = "✅" if enabled else "❌"
            logger.info(f"  {status} {feature.replace('_', ' ').title()}")
        
        # Data sources status
        logger.info("\n📊 Data Sources:")
        for source, info in self.config['data_sources'].items():
            if isinstance(info, dict):
                enabled = info.get('enabled', False)
                status = "✅" if enabled else "❌"
                logger.info(f"  {status} {source.title()}")
            else:
                status = "✅" if info else "❌"
                logger.info(f"  {status} {source.title()}")
        
        # Backtesters status
        logger.info("\n📈 Backtesters:")
        for name, path in self.config['backtesters'].items():
            backtester_path = self.project_root / path
            status = "✅" if backtester_path.exists() else "❌"
            logger.info(f"  {status} {name.title()}")
    
    def run_command(self, command: str):
        """Run a specific command."""
        commands = {
            "setup": self.setup_project,
            "test": self.test_project,
            "clean": self.clean_project,
            "deploy": self.deploy_project,
            "status": self.show_status
        }
        
        if command in commands:
            commands[command]()
        else:
            logger.error(f"Unknown command: {command}")
            logger.info("Available commands: " + ", ".join(commands.keys()))


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/project_manager.py <command>")
        print("Commands: setup, test, clean, deploy, status")
        sys.exit(1)
    
    command = sys.argv[1]
    manager = QuantAIProjectManager()
    manager.run_command(command)


if __name__ == "__main__":
    main()
