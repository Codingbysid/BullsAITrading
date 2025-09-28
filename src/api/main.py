from src.utils.common_imports import *
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio
from ..config.settings import get_settings
from ..data.data_sources import data_manager
from ..data.feature_engineering import FeatureEngineer
from ..data.sentiment_analysis import sentiment_monitor
from ..models.trading_models import create_ensemble_model
from ..risk.risk_management import RiskManager
from ..backtesting.backtesting_engine import BacktestingEngine
    import uvicorn

"""
FastAPI main application for QuantAI Trading Platform.

This module provides the main API endpoints for:
- Market data access
- Model predictions
- Risk management
- Portfolio management
- Real-time monitoring
"""



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = setup_logger()

# Initialize FastAPI app
app = FastAPI(
    title="QuantAI Trading Platform API",
    description="Advanced AI-driven quantitative trading platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
settings = get_settings()
risk_manager = RiskManager()
backtesting_engine = BacktestingEngine()
feature_engineer = FeatureEngineer()
ensemble_model = create_ensemble_model()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "QuantAI Trading Platform API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: str = "finazon"
):
    """
    Get market data for a symbol.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        source: Data source preference
    """
    try:
        # Parse dates
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = datetime.now() - timedelta(days=365)
        
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_dt = datetime.now()
        
        # Fetch data
        data = await data_manager.get_market_data(
            symbol=symbol,
            start_date=start_dt,
            end_date=end_dt,
            preferred_source=source
        )
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Convert to JSON-serializable format
        result = {
            "symbol": symbol,
            "data": data.reset_index().to_dict('records'),
            "metadata": {
                "start_date": start_dt.isoformat(),
                "end_date": end_dt.isoformat(),
                "source": source,
                "rows": len(data)
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/{symbol}")
async def get_features(symbol: str, lookback_days: int = 252):
    """
    Get engineered features for a symbol.
    
    Args:
        symbol: Stock symbol
        lookback_days: Number of days to look back
    """
    try:
        # Get market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        data = await data_manager.get_market_data(symbol, start_date, end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Create features
        features = feature_engineer.create_all_features(data)
        
        if features.empty:
            raise HTTPException(status_code=500, detail="Failed to create features")
        
        # Return features
        result = {
            "symbol": symbol,
            "features": features.reset_index().to_dict('records'),
            "metadata": {
                "lookback_days": lookback_days,
                "feature_count": len(features.columns),
                "rows": len(features)
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating features for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sentiment/{symbol}")
async def get_sentiment(symbol: str, lookback_days: int = 7):
    """
    Get sentiment analysis for a symbol.
    
    Args:
        symbol: Stock symbol
        lookback_days: Number of days to look back
    """
    try:
        sentiment = await sentiment_monitor.get_live_sentiment(symbol)
        
        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/{symbol}")
async def predict_trading_signal(symbol: str):
    """
    Generate trading prediction for a symbol.
    
    Args:
        symbol: Stock symbol
    """
    try:
        # Get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        data = await data_manager.get_market_data(symbol, start_date, end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Create features
        features = feature_engineer.create_all_features(data)
        
        if features.empty:
            raise HTTPException(status_code=500, detail="Failed to create features")
        
        # Get latest features for prediction
        latest_features = features.iloc[-1:].drop(columns=['future_return', 'future_direction', 'future_volatility', 'future_sharpe'], errors='ignore')
        
        # Generate prediction
        prediction = ensemble_model.predict(latest_features)
        confidence = ensemble_model.predict_proba(latest_features)[0][1] if hasattr(ensemble_model, 'predict_proba') else 0.5
        
        # Get sentiment
        sentiment = await sentiment_monitor.get_live_sentiment(symbol)
        
        # Calculate position size
        predicted_return = float(prediction[0])
        predicted_volatility = features['volatility_20'].iloc[-1] if 'volatility_20' in features.columns else 0.1
        position_size = risk_manager.calculate_position_size(
            predicted_return=predicted_return,
            predicted_volatility=predicted_volatility,
            confidence=confidence,
            current_drawdown=0.0  # Would be calculated from portfolio state
        )
        
        # Generate signal
        if predicted_return > 0.02 and confidence > 0.6:
            signal = "BUY"
        elif predicted_return < -0.02 and confidence > 0.6:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        return {
            "symbol": symbol,
            "prediction": {
                "expected_return": predicted_return,
                "confidence": confidence,
                "signal": signal,
                "position_size": position_size
            },
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating prediction for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest")
async def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001
):
    """
    Run backtest for a symbol.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Initial capital
        transaction_cost: Transaction cost as fraction
    """
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Get market data
        data = await data_manager.get_market_data(symbol, start_dt, end_dt)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Create features
        features = feature_engineer.create_all_features(data)
        
        if features.empty:
            raise HTTPException(status_code=500, detail="Failed to create features")
        
        # Prepare data for backtesting
        X = features.drop(columns=['future_return', 'future_direction', 'future_volatility', 'future_sharpe'], errors='ignore')
        y = features['future_return']
        prices = data['close']
        
        # Run backtest
        results = backtesting_engine.run_backtest(
            model=ensemble_model,
            X=X,
            y=y,
            prices=prices,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost
        )
        
        return {
            "symbol": symbol,
            "backtest_results": {
                "total_return": results.metrics.get('total_return', 0),
                "annualized_return": results.metrics.get('annualized_return', 0),
                "sharpe_ratio": results.metrics.get('sharpe_ratio', 0),
                "max_drawdown": results.metrics.get('max_drawdown', 0),
                "win_rate": results.metrics.get('win_rate', 0),
                "profit_factor": results.metrics.get('profit_factor', 0),
                "calmar_ratio": results.metrics.get('calmar_ratio', 0),
                "num_trades": results.metrics.get('num_trades', 0)
            },
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/risk-metrics")
async def get_portfolio_risk_metrics():
    """Get current portfolio risk metrics."""
    try:
        # This would typically fetch from a database or portfolio state
        # For now, return placeholder data
        return {
            "risk_metrics": {
                "current_drawdown": 0.0,
                "max_drawdown": 0.0,
                "portfolio_beta": 1.0,
                "var_95": -0.02,
                "expected_shortfall": -0.03
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/performance")
async def get_model_performance():
    """Get current model performance metrics."""
    try:
        # This would typically fetch from model performance tracking
        return {
            "model_performance": {
                "ensemble_accuracy": 0.65,
                "ensemble_sharpe": 1.2,
                "last_retrain": datetime.now().isoformat(),
                "prediction_latency_ms": 45
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
