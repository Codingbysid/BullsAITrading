"""
Unified configuration management for the QuantAI Trading Platform.

This module provides a single source of truth for all configuration
management, eliminating duplication across the codebase.
"""

import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import yaml
from dataclasses import dataclass, asdict
from .common_imports import setup_logger, save_json, load_json

logger = setup_logger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "postgresql://localhost:5432/quantai"
    redis_url: str = "redis://localhost:6379"
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = ""
    connection_pool_size: int = 10
    connection_timeout: int = 30


@dataclass
class APIConfig:
    """API configuration settings."""
    alpha_vantage_key: str = ""
    finazon_key: str = ""
    twelve_data_key: str = ""
    news_api_key: str = ""
    gemini_key: str = ""
    finnhub_key: str = ""
    iex_key: str = ""
    
    # Rate limits (requests per minute)
    alpha_vantage_rate_limit: int = 5
    finazon_rate_limit: int = 100
    twelve_data_rate_limit: int = 8
    news_api_rate_limit: int = 1000


@dataclass
class RiskConfig:
    """Risk management configuration settings."""
    max_position_size: float = 0.2
    max_drawdown: float = 0.1
    target_sharpe_ratio: float = 1.5
    max_portfolio_beta: float = 1.2
    min_portfolio_beta: float = 0.8
    risk_free_rate: float = 0.02
    var_confidence_level: float = 0.05
    kelly_confidence: float = 0.95


@dataclass
class TradingConfig:
    """Trading configuration settings."""
    trading_frequency: str = "daily"
    lookback_period: int = 252
    min_confidence_threshold: float = 0.6
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    target_symbols: List[str] = None
    
    def __post_init__(self):
        if self.target_symbols is None:
            self.target_symbols = ["GOOGL", "NVDA", "META", "AMZN", "AAPL"]


@dataclass
class ModelConfig:
    """Model configuration settings."""
    ensemble_weights: Dict[str, float] = None
    lookback_period: int = 252
    validation_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42
    n_jobs: int = -1
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "random_forest": 0.4,
                "xgboost": 0.35,
                "lstm": 0.25
            }


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "json"
    file_path: str = "logs/quantai.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    secret_key: str = "your_secret_key_here_change_this_in_production"
    jwt_secret: str = "your_jwt_secret_here_change_this_in_production"
    password_hash_algorithm: str = "pbkdf2_sha256"
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes


@dataclass
class AppConfig:
    """Application configuration settings."""
    name: str = "QuantAI Trading Platform"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False


class ConfigManager:
    """Unified configuration manager."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize default configurations
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.risk = RiskConfig()
        self.trading = TradingConfig()
        self.model = ModelConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.app = AppConfig()
        
        # Load configurations
        self.load_configurations()
    
    def load_configurations(self) -> None:
        """Load configurations from files and environment variables."""
        # Load from environment variables
        self._load_from_environment()
        
        # Load from config files
        self._load_from_files()
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # Database configuration
        if os.getenv("DATABASE_URL"):
            self.database.url = os.getenv("DATABASE_URL")
        if os.getenv("REDIS_URL"):
            self.database.redis_url = os.getenv("REDIS_URL")
        if os.getenv("INFLUXDB_URL"):
            self.database.influxdb_url = os.getenv("INFLUXDB_URL")
        if os.getenv("INFLUXDB_TOKEN"):
            self.database.influxdb_token = os.getenv("INFLUXDB_TOKEN")
        
        # API configuration
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            self.api.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if os.getenv("FINAZON_API_KEY"):
            self.api.finazon_key = os.getenv("FINAZON_API_KEY")
        if os.getenv("TWELVE_DATA_API_KEY"):
            self.api.twelve_data_key = os.getenv("TWELVE_DATA_API_KEY")
        if os.getenv("NEWS_API_KEY"):
            self.api.news_api_key = os.getenv("NEWS_API_KEY")
        if os.getenv("GEMINI_API_KEY"):
            self.api.gemini_key = os.getenv("GEMINI_API_KEY")
        if os.getenv("FINNHUB_API_KEY"):
            self.api.finnhub_key = os.getenv("FINNHUB_API_KEY")
        if os.getenv("IEX_API_KEY"):
            self.api.iex_key = os.getenv("IEX_API_KEY")
        
        # Risk configuration
        if os.getenv("MAX_POSITION_SIZE"):
            self.risk.max_position_size = float(os.getenv("MAX_POSITION_SIZE"))
        if os.getenv("MAX_DRAWDOWN"):
            self.risk.max_drawdown = float(os.getenv("MAX_DRAWDOWN"))
        if os.getenv("TARGET_SHARPE_RATIO"):
            self.risk.target_sharpe_ratio = float(os.getenv("TARGET_SHARPE_RATIO"))
        if os.getenv("RISK_FREE_RATE"):
            self.risk.risk_free_rate = float(os.getenv("RISK_FREE_RATE"))
        
        # Trading configuration
        if os.getenv("TRADING_FREQUENCY"):
            self.trading.trading_frequency = os.getenv("TRADING_FREQUENCY")
        if os.getenv("LOOKBACK_PERIOD"):
            self.trading.lookback_period = int(os.getenv("LOOKBACK_PERIOD"))
        if os.getenv("MIN_CONFIDENCE_THRESHOLD"):
            self.trading.min_confidence_threshold = float(os.getenv("MIN_CONFIDENCE_THRESHOLD"))
        
        # Logging configuration
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_FORMAT"):
            self.logging.format = os.getenv("LOG_FORMAT")
        if os.getenv("LOG_FILE"):
            self.logging.file_path = os.getenv("LOG_FILE")
        
        # Security configuration
        if os.getenv("SECRET_KEY"):
            self.security.secret_key = os.getenv("SECRET_KEY")
        if os.getenv("JWT_SECRET"):
            self.security.jwt_secret = os.getenv("JWT_SECRET")
        
        # App configuration
        if os.getenv("DEBUG"):
            self.app.debug = os.getenv("DEBUG").lower() in ("true", "1", "yes")
        if os.getenv("HOST"):
            self.app.host = os.getenv("HOST")
        if os.getenv("PORT"):
            self.app.port = int(os.getenv("PORT"))
    
    def _load_from_files(self) -> None:
        """Load configuration from files."""
        # Load from JSON files
        config_files = {
            "database.json": self.database,
            "api.json": self.api,
            "risk.json": self.risk,
            "trading.json": self.trading,
            "model.json": self.model,
            "logging.json": self.logging,
            "security.json": self.security,
            "app.json": self.app
        }
        
        for filename, config_obj in config_files.items():
            filepath = self.config_dir / filename
            if filepath.exists():
                try:
                    config_data = load_json(filepath)
                    if config_data:
                        # Update configuration object
                        for key, value in config_data.items():
                            if hasattr(config_obj, key):
                                setattr(config_obj, key, value)
                except Exception as e:
                    logger.error(f"Failed to load config from {filepath}: {e}")
        
        # Load from YAML files if they exist
        yaml_files = {
            "config.yaml": "all",
            "database.yaml": "database",
            "api.yaml": "api",
            "risk.yaml": "risk",
            "trading.yaml": "trading",
            "model.yaml": "model",
            "logging.yaml": "logging",
            "security.yaml": "security",
            "app.yaml": "app"
        }
        
        for filename, config_section in yaml_files.items():
            filepath = self.config_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        yaml_data = yaml.safe_load(f)
                    
                    if config_section == "all":
                        # Update all configurations
                        for section_name, section_data in yaml_data.items():
                            if hasattr(self, section_name):
                                config_obj = getattr(self, section_name)
                                for key, value in section_data.items():
                                    if hasattr(config_obj, key):
                                        setattr(config_obj, key, value)
                    else:
                        # Update specific configuration
                        if hasattr(self, config_section):
                            config_obj = getattr(self, config_section)
                            for key, value in yaml_data.items():
                                if hasattr(config_obj, key):
                                    setattr(config_obj, key, value)
                except Exception as e:
                    logger.error(f"Failed to load YAML config from {filepath}: {e}")
    
    def save_configurations(self) -> None:
        """Save current configurations to files."""
        # Save as JSON files
        configs = {
            "database": self.database,
            "api": self.api,
            "risk": self.risk,
            "trading": self.trading,
            "model": self.model,
            "logging": self.logging,
            "security": self.security,
            "app": self.app
        }
        
        for name, config_obj in configs.items():
            filepath = self.config_dir / f"{name}.json"
            try:
                save_json(asdict(config_obj), filepath)
            except Exception as e:
                logger.error(f"Failed to save config to {filepath}: {e}")
        
        # Save as YAML file
        yaml_filepath = self.config_dir / "config.yaml"
        try:
            yaml_data = {}
            for name, config_obj in configs.items():
                yaml_data[name] = asdict(config_obj)
            
            with open(yaml_filepath, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save YAML config to {yaml_filepath}: {e}")
    
    def get_config(self, section: str) -> Optional[Any]:
        """Get configuration section by name."""
        return getattr(self, section, None)
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update configuration section with new values."""
        config_obj = getattr(self, section, None)
        if config_obj is None:
            logger.error(f"Unknown configuration section: {section}")
            return False
        
        try:
            for key, value in updates.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {section}.{key}")
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration {section}: {e}")
            return False
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return list of issues."""
        issues = []
        
        # Validate database configuration
        if not self.database.url:
            issues.append("Database URL is not configured")
        
        # Validate API keys (optional)
        if not any([
            self.api.alpha_vantage_key,
            self.api.finazon_key,
            self.api.twelve_data_key
        ]):
            issues.append("No API keys configured (some features may not work)")
        
        # Validate risk configuration
        if self.risk.max_position_size <= 0 or self.risk.max_position_size > 1:
            issues.append("Max position size must be between 0 and 1")
        
        if self.risk.max_drawdown <= 0 or self.risk.max_drawdown > 1:
            issues.append("Max drawdown must be between 0 and 1")
        
        if self.risk.risk_free_rate < 0 or self.risk.risk_free_rate > 1:
            issues.append("Risk-free rate must be between 0 and 1")
        
        # Validate trading configuration
        if self.trading.lookback_period <= 0:
            issues.append("Lookback period must be positive")
        
        if self.trading.min_confidence_threshold < 0 or self.trading.min_confidence_threshold > 1:
            issues.append("Min confidence threshold must be between 0 and 1")
        
        # Validate model configuration
        if self.model.validation_split < 0 or self.model.validation_split > 1:
            issues.append("Validation split must be between 0 and 1")
        
        if self.model.test_split < 0 or self.model.test_split > 1:
            issues.append("Test split must be between 0 and 1")
        
        if self.model.validation_split + self.model.test_split >= 1:
            issues.append("Validation and test splits must sum to less than 1")
        
        # Validate security configuration
        if self.security.secret_key == "your_secret_key_here_change_this_in_production":
            issues.append("Default secret key detected - change for production")
        
        if self.security.jwt_secret == "your_jwt_secret_here_change_this_in_production":
            issues.append("Default JWT secret detected - change for production")
        
        return issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "database": {
                "url_configured": bool(self.database.url),
                "redis_configured": bool(self.database.redis_url),
                "influxdb_configured": bool(self.database.influxdb_url)
            },
            "api": {
                "alpha_vantage_configured": bool(self.api.alpha_vantage_key),
                "finazon_configured": bool(self.api.finazon_key),
                "twelve_data_configured": bool(self.api.twelve_data_key),
                "news_api_configured": bool(self.api.news_api_key),
                "gemini_configured": bool(self.api.gemini_key)
            },
            "risk": {
                "max_position_size": self.risk.max_position_size,
                "max_drawdown": self.risk.max_drawdown,
                "target_sharpe_ratio": self.risk.target_sharpe_ratio,
                "risk_free_rate": self.risk.risk_free_rate
            },
            "trading": {
                "frequency": self.trading.trading_frequency,
                "lookback_period": self.trading.lookback_period,
                "min_confidence_threshold": self.trading.min_confidence_threshold,
                "target_symbols": self.trading.target_symbols
            },
            "model": {
                "ensemble_weights": self.model.ensemble_weights,
                "lookback_period": self.model.lookback_period,
                "validation_split": self.model.validation_split,
                "test_split": self.model.test_split
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file_path": self.logging.file_path
            },
            "security": {
                "password_hash_algorithm": self.security.password_hash_algorithm,
                "session_timeout": self.security.session_timeout,
                "max_login_attempts": self.security.max_login_attempts
            },
            "app": {
                "name": self.app.name,
                "version": self.app.version,
                "debug": self.app.debug,
                "host": self.app.host,
                "port": self.app.port
            }
        }


# Global configuration manager instance
config_manager = ConfigManager()


# Convenience functions
def get_config(section: str) -> Optional[Any]:
    """Get configuration section by name."""
    return config_manager.get_config(section)

def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return config_manager.database

def get_api_config() -> APIConfig:
    """Get API configuration."""
    return config_manager.api

def get_risk_config() -> RiskConfig:
    """Get risk configuration."""
    return config_manager.risk

def get_trading_config() -> TradingConfig:
    """Get trading configuration."""
    return config_manager.trading

def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return config_manager.model

def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return config_manager.logging

def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return config_manager.security

def get_app_config() -> AppConfig:
    """Get application configuration."""
    return config_manager.app

def update_config(section: str, updates: Dict[str, Any]) -> bool:
    """Update configuration section with new values."""
    return config_manager.update_config(section, updates)

def save_config() -> None:
    """Save current configurations to files."""
    config_manager.save_configurations()

def validate_config() -> List[str]:
    """Validate current configuration and return list of issues."""
    return config_manager.validate_config()

def get_config_summary() -> Dict[str, Any]:
    """Get configuration summary."""
    return config_manager.get_summary()
