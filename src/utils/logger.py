"""
Logging utilities for QuantAI platform.

This module provides structured logging with different levels and formats
for debugging, monitoring, and compliance requirements.
"""

import logging
import structlog
from typing import Dict, Any
from datetime import datetime
import json


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Setup structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ("json" or "text")
    """
    # Configure structlog
    if log_format == "json":
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Set logging level
    logging.basicConfig(level=getattr(logging, log_level.upper()))


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def log_trading_decision(
    symbol: str,
    action: str,
    confidence: float,
    reasoning: str,
    additional_context: Dict[str, Any] = None
) -> None:
    """
    Log trading decisions with structured format.
    
    Args:
        symbol: Stock symbol
        action: Trading action (BUY/SELL/HOLD)
        confidence: Model confidence (0-1)
        reasoning: Human-readable reasoning
        additional_context: Additional context data
    """
    logger = get_logger("trading")
    
    context = {
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "reasoning": reasoning,
        "timestamp": datetime.now().isoformat()
    }
    
    if additional_context:
        context.update(additional_context)
    
    logger.info("trading_decision", **context)


def log_model_performance(
    model_name: str,
    accuracy: float,
    sharpe_ratio: float,
    max_drawdown: float,
    additional_metrics: Dict[str, float] = None
) -> None:
    """
    Log model performance metrics.
    
    Args:
        model_name: Name of the model
        accuracy: Model accuracy
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown
        additional_metrics: Additional performance metrics
    """
    logger = get_logger("model_performance")
    
    context = {
        "model_name": model_name,
        "accuracy": accuracy,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "timestamp": datetime.now().isoformat()
    }
    
    if additional_metrics:
        context.update(additional_metrics)
    
    logger.info("model_performance", **context)


def log_risk_event(
    event_type: str,
    severity: str,
    description: str,
    portfolio_impact: float,
    additional_context: Dict[str, Any] = None
) -> None:
    """
    Log risk management events.
    
    Args:
        event_type: Type of risk event
        severity: Event severity (LOW/MEDIUM/HIGH/CRITICAL)
        description: Event description
        portfolio_impact: Impact on portfolio
        additional_context: Additional context data
    """
    logger = get_logger("risk_management")
    
    context = {
        "event_type": event_type,
        "severity": severity,
        "description": description,
        "portfolio_impact": portfolio_impact,
        "timestamp": datetime.now().isoformat()
    }
    
    if additional_context:
        context.update(additional_context)
    
    logger.warning("risk_event", **context)


def log_api_call(
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: float,
    user_id: str = None
) -> None:
    """
    Log API calls for monitoring and debugging.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        status_code: Response status code
        response_time_ms: Response time in milliseconds
        user_id: User ID (if authenticated)
    """
    logger = get_logger("api")
    
    context = {
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "response_time_ms": response_time_ms,
        "timestamp": datetime.now().isoformat()
    }
    
    if user_id:
        context["user_id"] = user_id
    
    logger.info("api_call", **context)


def log_data_quality_issue(
    data_source: str,
    issue_type: str,
    description: str,
    impact: str,
    additional_context: Dict[str, Any] = None
) -> None:
    """
    Log data quality issues.
    
    Args:
        data_source: Source of the data
        issue_type: Type of issue
        description: Issue description
        impact: Impact on system
        additional_context: Additional context data
    """
    logger = get_logger("data_quality")
    
    context = {
        "data_source": data_source,
        "issue_type": issue_type,
        "description": description,
        "impact": impact,
        "timestamp": datetime.now().isoformat()
    }
    
    if additional_context:
        context.update(additional_context)
    
    logger.warning("data_quality_issue", **context)
