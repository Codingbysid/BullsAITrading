"""
Web dashboard API for the QuantAI Trading Platform.

This module provides a FastAPI-based web dashboard for monitoring:
- Real-time performance metrics
- Risk alerts and monitoring
- Trading decisions and signals
- System health and status
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from .performance_monitor import performance_monitor, get_dashboard_data
from ..trading.decision_engine import decision_engine
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="QuantAI Trading Dashboard",
    description="Real-time monitoring dashboard for the QuantAI Trading Platform",
    version="1.0.0"
)

# Templates for HTML responses
templates = Jinja2Templates(directory="src/dashboard/templates")


@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "QuantAI Trading Dashboard"
    })


@app.get("/api/performance")
async def get_performance():
    """Get current performance metrics."""
    try:
        return JSONResponse(content=performance_monitor.get_performance_summary())
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk-alerts")
async def get_risk_alerts(severity: Optional[str] = None):
    """Get risk alerts, optionally filtered by severity."""
    try:
        alerts = performance_monitor.get_risk_alerts(severity)
        return JSONResponse(content=alerts)
    except Exception as e:
        logger.error(f"Error getting risk alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance-trends")
async def get_performance_trends(days: int = 30):
    """Get performance trends over time."""
    try:
        trends = performance_monitor.get_performance_trends(days)
        return JSONResponse(content=trends)
    except Exception as e:
        logger.error(f"Error getting performance trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-performance")
async def get_model_performance():
    """Get model performance metrics."""
    try:
        model_perf = performance_monitor.get_model_performance()
        return JSONResponse(content=model_perf)
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions")
async def get_recent_decisions(limit: int = 50):
    """Get recent trading decisions."""
    try:
        decisions = decision_engine.get_performance_metrics()
        recent_decisions = decisions.get('recent_decisions', [])[-limit:]
        return JSONResponse(content=recent_decisions)
    except Exception as e:
        logger.error(f"Error getting recent decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard-data")
async def get_dashboard_data_endpoint():
    """Get comprehensive dashboard data."""
    try:
        data = get_dashboard_data()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export-performance")
async def export_performance_data(background_tasks: BackgroundTasks):
    """Export performance data to file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_export_{timestamp}.json"
        filepath = f"exports/{filename}"
        
        # Create exports directory if it doesn't exist
        import os
        os.makedirs("exports", exist_ok=True)
        
        background_tasks.add_task(performance_monitor.export_performance_data, filepath)
        
        return JSONResponse(content={
            "message": "Export started",
            "filename": filename,
            "filepath": filepath
        })
    except Exception as e:
        logger.error(f"Error exporting performance data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        settings = get_settings()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "performance_monitor": "healthy",
                "decision_engine": "healthy",
                "risk_manager": "healthy"
            }
        }
        
        return JSONResponse(content=health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            },
            status_code=500
        )


@app.get("/api/system-status")
async def get_system_status():
    """Get detailed system status."""
    try:
        settings = get_settings()
        
        # Get performance metrics
        performance = performance_monitor.get_performance_summary()
        
        # Get decision engine metrics
        decisions = decision_engine.get_performance_metrics()
        
        # Get risk alerts
        alerts = performance_monitor.get_risk_alerts()
        high_risk_alerts = [a for a in alerts if a.get('severity') == 'HIGH']
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "operational",
            "performance": performance,
            "decisions": decisions,
            "risk_status": {
                "total_alerts": len(alerts),
                "high_risk_alerts": len(high_risk_alerts),
                "status": "HIGH_RISK" if len(high_risk_alerts) > 0 else "NORMAL"
            },
            "settings": {
                "max_position_size": settings.max_position_size,
                "max_drawdown": settings.max_drawdown,
                "target_sharpe_ratio": settings.target_sharpe_ratio
            }
        }
        
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time updates (if needed)
@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket):
    """WebSocket endpoint for real-time dashboard updates."""
    try:
        await websocket.accept()
        
        while True:
            # Get latest dashboard data
            data = get_dashboard_data()
            
            # Send data to client
            await websocket.send_json(data)
            
            # Wait before next update
            import asyncio
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


def run_dashboard(host: str = "0.0.0.0", port: int = 8001):
    """Run the dashboard server."""
    try:
        logger.info(f"Starting QuantAI Trading Dashboard on {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        raise


if __name__ == "__main__":
    run_dashboard()
