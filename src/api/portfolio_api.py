"""
FastAPI Web API for QuantAI Portfolio Manager & Trade Suggestion Bot.

This module provides a RESTful API for the portfolio management system,
enabling web-based access to all features.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Import portfolio components
try:
    from ..database.db_manager import QuantAIDatabase
    from ..portfolio.portfolio_manager import PortfolioManager
    from ..training.feedback_trainer import ReinforcementFeedbackTrainer
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    logging.warning("Portfolio modules not available, using fallback")

# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    risk_tolerance: str = "Moderate"
    investment_experience: str = "Beginner"

class UserLogin(BaseModel):
    username: str
    password: str

class PositionAdd(BaseModel):
    symbol: str
    shares: float
    price: float

class RecommendationResponse(BaseModel):
    recommendation_id: int
    accepted: bool
    actual_shares: Optional[float] = 0
    actual_price: Optional[float] = 0

# Initialize FastAPI app
app = FastAPI(
    title="QuantAI Portfolio Manager API",
    description="AI-powered portfolio management and trading recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
db = None
portfolio_manager = None
feedback_trainer = None
current_users = {}  # Simple session management

def get_database():
    """Get database connection"""
    global db
    if db is None:
        db_config = {
            'host': 'localhost',
            'database': 'quantai_portfolio',
            'user': 'root',
            'password': ''
        }
        try:
            db = QuantAIDatabase(db_config)
            db.create_tables()
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            # Use fallback
            db = None
    return db

def get_portfolio_manager():
    """Get portfolio manager"""
    global portfolio_manager
    if portfolio_manager is None:
        db = get_database()
        portfolio_manager = PortfolioManager(db)
    return portfolio_manager

def get_feedback_trainer():
    """Get feedback trainer"""
    global feedback_trainer
    if feedback_trainer is None:
        db = get_database()
        feedback_trainer = ReinforcementFeedbackTrainer(db)
    return feedback_trainer

# Authentication endpoints
@app.post("/auth/register")
async def register_user(user: UserCreate):
    """Register new user"""
    try:
        db = get_database()
        if db is None:
            raise HTTPException(status_code=500, detail="Database not available")
        
        user_id = db.create_user(
            username=user.username,
            password=user.password,
            email=user.email,
            risk_tolerance=user.risk_tolerance,
            experience=user.investment_experience
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "message": "User created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
async def login_user(user: UserLogin):
    """Login user"""
    try:
        db = get_database()
        if db is None:
            # Fallback authentication
            if user.username == "demo" and user.password == "demo":
                current_users["demo"] = 1
                return {
                    "success": True,
                    "user_id": 1,
                    "message": "Demo user logged in"
                }
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user_id = db.authenticate_user(user.username, user.password)
        if user_id:
            current_users[user.username] = user_id
            return {
                "success": True,
                "user_id": user_id,
                "message": "Login successful"
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Portfolio endpoints
@app.get("/portfolio/{user_id}")
async def get_portfolio(user_id: int):
    """Get user portfolio"""
    try:
        portfolio_manager = get_portfolio_manager()
        portfolio = portfolio_manager.get_user_portfolio_summary(user_id)
        return portfolio
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio/{user_id}/positions")
async def add_position(user_id: int, position: PositionAdd):
    """Add stock position"""
    try:
        if position.symbol not in ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']:
            raise HTTPException(status_code=400, detail="Invalid symbol")
        
        db = get_database()
        if db is None:
            raise HTTPException(status_code=500, detail="Database not available")
        
        portfolio_id = db.add_portfolio_position(
            user_id, position.symbol, position.shares, position.price
        )
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "message": f"Added {position.shares} shares of {position.symbol}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/portfolio/{user_id}/recommendations")
async def get_recommendations(user_id: int):
    """Get AI recommendations"""
    try:
        portfolio_manager = get_portfolio_manager()
        recommendations = portfolio_manager.generate_recommendations(user_id)
        return {
            "success": True,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio/{user_id}/recommendations/{recommendation_id}/respond")
async def respond_to_recommendation(
    user_id: int, 
    recommendation_id: int, 
    response: RecommendationResponse
):
    """Respond to recommendation"""
    try:
        portfolio_manager = get_portfolio_manager()
        result = portfolio_manager.execute_recommendation(
            recommendation_id, user_id, response.accepted, 
            response.actual_shares, response.actual_price
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/portfolio/{user_id}/analytics")
async def get_analytics(user_id: int, days: int = 30):
    """Get performance analytics"""
    try:
        portfolio_manager = get_portfolio_manager()
        analytics = portfolio_manager.get_performance_analytics(user_id, days)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio/update-prices")
async def update_prices():
    """Update all stock prices"""
    try:
        portfolio_manager = get_portfolio_manager()
        portfolio_manager.update_all_portfolios()
        return {
            "success": True,
            "message": "Prices updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# System endpoints
@app.get("/system/status")
async def get_system_status():
    """Get system status"""
    return {
        "status": "operational",
        "modules_available": MODULES_AVAILABLE,
        "database_connected": db is not None,
        "supported_symbols": ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL'],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/system/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# Learning endpoints
@app.post("/learning/train")
async def train_feedback_model():
    """Train feedback model"""
    try:
        feedback_trainer = get_feedback_trainer()
        results = feedback_trainer.train_feedback_model()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/performance")
async def get_learning_performance():
    """Get learning performance report"""
    try:
        feedback_trainer = get_feedback_trainer()
        report = feedback_trainer.get_model_performance_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "QuantAI Portfolio Manager API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
