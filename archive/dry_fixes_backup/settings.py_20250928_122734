"""
Configuration settings for QuantAI Trading Platform.

This module handles all configuration management including environment variables,
API keys, and system parameters.
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "QuantAI Trading Platform"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # API Keys (Optional - FinViz uses web scraping, no API key needed)
    alpha_vantage_api_key: str = Field(default="", env="ALPHA_VANTAGE_API_KEY")
    finazon_api_key: str = Field(default="", env="FINAZON_API_KEY")
    twelve_data_api_key: str = Field(default="", env="TWELVE_DATA_API_KEY")
    news_api_key: str = Field(default="", env="NEWS_API_KEY")
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    
    # Database Configuration (Optional for basic usage)
    database_url: str = Field(default="postgresql://localhost:5432/quantai", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    influxdb_url: str = Field(default="http://localhost:8086", env="INFLUXDB_URL")
    influxdb_token: str = Field(default="", env="INFLUXDB_TOKEN")
    
    # Risk Management Parameters
    max_position_size: float = Field(default=0.2, env="MAX_POSITION_SIZE")
    max_drawdown: float = Field(default=0.1, env="MAX_DRAWDOWN")
    target_sharpe_ratio: float = Field(default=1.5, env="TARGET_SHARPE_RATIO")
    max_portfolio_beta: float = Field(default=1.2, env="MAX_PORTFOLIO_BETA")
    min_portfolio_beta: float = Field(default=0.8, env="MIN_PORTFOLIO_BETA")
    
    # Trading Parameters
    trading_frequency: str = Field(default="daily", env="TRADING_FREQUENCY")
    lookback_period: int = Field(default=252, env="LOOKBACK_PERIOD")
    min_confidence_threshold: float = Field(default=0.6, env="MIN_CONFIDENCE_THRESHOLD")
    
    # Target Assets (FAANG+)
    target_symbols: List[str] = [
        "GOOGL", "NVDA", "META", "AMZN", "AAPL"
    ]
    
    # API Rate Limits (requests per minute)
    alpha_vantage_rate_limit: int = Field(default=5, env="ALPHA_VANTAGE_RATE_LIMIT")
    finazon_rate_limit: int = Field(default=100, env="FINAZON_RATE_LIMIT")
    twelve_data_rate_limit: int = Field(default=8, env="TWELVE_DATA_RATE_LIMIT")
    news_api_rate_limit: int = Field(default=1000, env="NEWS_API_RATE_LIMIT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Security (Optional for basic usage)
    secret_key: str = Field(default="your_secret_key_here_change_this_in_production", env="SECRET_KEY")
    jwt_secret: str = Field(default="your_jwt_secret_here_change_this_in_production", env="JWT_SECRET")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from environment


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
