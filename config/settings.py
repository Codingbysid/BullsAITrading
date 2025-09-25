"""
Configuration settings for QuantAI Portfolio Manager.

This module contains all configuration settings for the application,
including database, API, and system settings.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '3306'))
    database: str = os.getenv('DB_NAME', 'quantai_portfolio')
    user: str = os.getenv('DB_USER', 'root')
    password: str = os.getenv('DB_PASSWORD', '')
    charset: str = 'utf8mb4'
    autocommit: bool = True
    use_unicode: bool = True

@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = os.getenv('API_HOST', '0.0.0.0')
    port: int = int(os.getenv('API_PORT', '8000'))
    debug: bool = os.getenv('API_DEBUG', 'False').lower() == 'true'
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ['*']

@dataclass
class TradingConfig:
    """Trading configuration settings"""
    supported_symbols: list = None
    max_position_size: float = 0.2  # 20% max per position
    max_portfolio_risk: float = 0.1  # 10% max portfolio risk
    kelly_fraction_limit: float = 0.25  # 25% max Kelly fraction
    min_confidence_threshold: float = 0.6  # 60% minimum confidence
    max_risk_score: float = 0.8  # 80% maximum risk
    
    def __post_init__(self):
        if self.supported_symbols is None:
            self.supported_symbols = ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']

@dataclass
class LearningConfig:
    """Reinforcement learning configuration"""
    feedback_evaluation_days: int = 30
    model_retrain_frequency: int = 7  # days
    min_feedback_samples: int = 10
    learning_rate: float = 0.01
    personalization_threshold: float = 0.7
    confidence_bands: list = None
    
    def __post_init__(self):
        if self.confidence_bands is None:
            self.confidence_bands = [0.6, 0.8, 1.0]

@dataclass
class RiskConfig:
    """Risk management configuration"""
    var_confidence_levels: list = None
    max_drawdown_threshold: float = 0.15  # 15% max drawdown
    concentration_risk_threshold: float = 0.3  # 30% max concentration
    correlation_threshold: float = 0.7  # 70% max correlation
    volatility_threshold: float = 0.3  # 30% max volatility
    
    def __post_init__(self):
        if self.var_confidence_levels is None:
            self.var_confidence_levels = [0.95, 0.99]

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_path: str = os.getenv('LOG_FILE', 'logs/quantai.log')
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class SecurityConfig:
    """Security configuration"""
    password_min_length: int = 8
    password_require_special: bool = True
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    encryption_key: str = os.getenv('ENCRYPTION_KEY', 'default_key_change_in_production')

class Settings:
    """Main settings class"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.trading = TradingConfig()
        self.learning = LearningConfig()
        self.risk = RiskConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            'logs',
            'models',
            'data',
            'backups',
            'reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return (f"mysql://{self.database.user}:{self.database.password}@"
                f"{self.database.host}:{self.database.port}/{self.database.database}")
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL for caching"""
        return os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return not self.is_production()
    
    def get_cors_origins(self) -> list:
        """Get CORS origins based on environment"""
        if self.is_production():
            return ['https://yourdomain.com']
        else:
            return ['*']
    
    def get_rate_limits(self) -> Dict[str, int]:
        """Get rate limits for API endpoints"""
        return {
            'auth': 5,  # 5 requests per minute
            'recommendations': 10,  # 10 requests per minute
            'portfolio': 20,  # 20 requests per minute
            'analytics': 5  # 5 requests per minute
        }
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags"""
        return {
            'personalized_recommendations': True,
            'reinforcement_learning': True,
            'risk_alerts': True,
            'portfolio_rebalancing': True,
            'advanced_analytics': True,
            'user_feedback': True,
            'learning_insights': True
        }
    
    def get_performance_targets(self) -> Dict[str, float]:
        """Get performance targets"""
        return {
            'recommendation_accuracy': 0.65,
            'user_acceptance_rate': 0.4,
            'response_time_ms': 500,
            'learning_improvement_days': 30,
            'min_sharpe_ratio': 1.5,
            'max_drawdown': 0.1
        }

# Global settings instance
settings = Settings()

# Environment-specific overrides
if settings.is_development():
    settings.database.host = 'localhost'
    settings.api.debug = True
    settings.logging.level = 'DEBUG'
elif settings.is_production():
    settings.database.host = os.getenv('DB_HOST', 'production-db-host')
    settings.api.debug = False
    settings.logging.level = 'INFO'
    settings.security.encryption_key = os.getenv('ENCRYPTION_KEY')
