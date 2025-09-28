from src.utils.common_imports import *
import mysql.connector
from mysql.connector import Error
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
            import sqlite3

"""
Database Manager for QuantAI Portfolio Manager & Trade Suggestion Bot.

This module manages the core database operations for user management,
portfolio tracking, and recommendation feedback learning.
"""


class QuantAIDatabase:
    """Database manager for QuantAI Portfolio Manager"""
    
    def __init__(self, config: dict):
        self.config = config
        self.connection = None
        self.logger = setup_logger()
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.config.get('host', 'localhost'),
                database=self.config.get('database', 'quantai_portfolio'),
                user=self.config.get('user', 'root'),
                password=self.config.get('password', '')
            )
            self.logger.info("Database connection established")
        except Error as e:
            self.logger.error(f"Database connection failed: {e}")
            # For development, create in-memory SQLite fallback
            self._create_sqlite_fallback()
    
    def _create_sqlite_fallback(self):
        """Create SQLite fallback for development"""
        try:
            self.connection = sqlite3.connect('quantai_portfolio.db')
            self.logger.info("Using SQLite fallback database")
        except ImportError:
            self.logger.error("Neither MySQL nor SQLite available")
            raise
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        tables = {
            'users': """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    email VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    risk_tolerance ENUM('Conservative', 'Moderate', 'Aggressive') DEFAULT 'Moderate',
                    investment_experience ENUM('Beginner', 'Intermediate', 'Advanced') DEFAULT 'Beginner',
                    portfolio_value DECIMAL(15, 2) DEFAULT 0.00,
                    total_pnl DECIMAL(15, 2) DEFAULT 0.00,
                    is_active BOOLEAN DEFAULT TRUE,
                    preferences JSON,
                    INDEX idx_username (username),
                    INDEX idx_risk_tolerance (risk_tolerance)
                )
            """,
            'user_portfolios': """
                CREATE TABLE IF NOT EXISTS user_portfolios (
                    portfolio_id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    user_id INTEGER NOT NULL,
                    symbol ENUM('AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL') NOT NULL,
                    shares DECIMAL(10, 4) NOT NULL,
                    avg_purchase_price DECIMAL(10, 2) NOT NULL,
                    purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    current_price DECIMAL(10, 2),
                    market_value DECIMAL(12, 2),
                    unrealized_pnl DECIMAL(12, 2),
                    realized_pnl DECIMAL(12, 2) DEFAULT 0.00,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    position_type ENUM('LONG', 'SHORT') DEFAULT 'LONG',
                    stop_loss DECIMAL(10, 2),
                    take_profit DECIMAL(10, 2),
                    notes TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    INDEX idx_user_symbol (user_id, symbol),
                    INDEX idx_symbol (symbol),
                    INDEX idx_last_updated (last_updated)
                )
            """,
            'recommendations_feedback': """
                CREATE TABLE IF NOT EXISTS recommendations_feedback (
                    recommendation_id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    user_id INTEGER NOT NULL,
                    symbol ENUM('AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL') NOT NULL,
                    recommendation_type ENUM('BUY', 'SELL', 'HOLD') NOT NULL,
                    recommended_shares DECIMAL(10, 4),
                    recommended_price DECIMAL(10, 2),
                    confidence_score DECIMAL(3, 2),
                    recommendation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    user_accepted BOOLEAN DEFAULT FALSE,
                    user_response_timestamp TIMESTAMP NULL,
                    actual_shares_traded DECIMAL(10, 4) DEFAULT 0,
                    actual_price DECIMAL(10, 2),
                    
                    outcome_evaluated BOOLEAN DEFAULT FALSE,
                    outcome_pnl DECIMAL(12, 2),
                    outcome_return_pct DECIMAL(5, 4),
                    evaluation_timestamp TIMESTAMP NULL,
                    
                    model_prediction DECIMAL(10, 2),
                    model_features JSON,
                    risk_score DECIMAL(3, 2),
                    
                    recommendation_reasoning TEXT,
                    market_conditions JSON,
                    user_feedback_score INTEGER DEFAULT NULL,
                    user_feedback_comment TEXT,
                    
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    INDEX idx_user_recommendations (user_id, recommendation_timestamp),
                    INDEX idx_symbol_recommendations (symbol, recommendation_timestamp),
                    INDEX idx_outcome_evaluated (outcome_evaluated),
                    INDEX idx_confidence_score (confidence_score)
                )
            """
        }
        
        cursor = self.connection.cursor()
        for table_name, table_sql in tables.items():
            try:
                cursor.execute(table_sql)
                self.logger.info(f"Table {table_name} created/verified")
            except Error as e:
                self.logger.error(f"Failed to create table {table_name}: {e}")
                raise
        
        cursor.close()
        self.connection.commit()

    # User Management Methods
    def create_user(self, username: str, password: str, email: str = None, 
                   risk_tolerance: str = 'Moderate', experience: str = 'Beginner') -> int:
        """Create new user account"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        cursor = self.connection.cursor()
        query = """
            INSERT INTO users (username, password_hash, email, risk_tolerance, investment_experience)
            VALUES (%s, %s, %s, %s, %s)
        """
        try:
            cursor.execute(query, (username, password_hash, email, risk_tolerance, experience))
            user_id = cursor.lastrowid
            self.connection.commit()
            self.logger.info(f"User {username} created with ID {user_id}")
            return user_id
        except Error as e:
            self.logger.error(f"Failed to create user {username}: {e}")
            raise
        finally:
            cursor.close()
    
    def authenticate_user(self, username: str, password: str) -> Optional[int]:
        """Authenticate user and return user_id"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        cursor = self.connection.cursor()
        query = """
            SELECT user_id FROM users 
            WHERE username = %s AND password_hash = %s
        """
        try:
            cursor.execute(query, (username, password_hash))
            result = cursor.fetchone()
            
            if result:
                user_id = result[0]
                # Update last login
                update_query = "UPDATE users SET last_login = %s WHERE user_id = %s"
                cursor.execute(update_query, (datetime.now(), user_id))
                self.connection.commit()
                self.logger.info(f"User {username} authenticated successfully")
                return user_id
            else:
                self.logger.warning(f"Authentication failed for user {username}")
                return None
        except Error as e:
            self.logger.error(f"Authentication error for user {username}: {e}")
            return None
        finally:
            cursor.close()

    # Portfolio Management Methods
    def add_portfolio_position(self, user_id: int, symbol: str, shares: float, 
                             purchase_price: float) -> int:
        """Add or update portfolio position"""
        cursor = self.connection.cursor()
        
        # Check if position exists
        check_query = """
            SELECT portfolio_id, shares, avg_purchase_price 
            FROM user_portfolios 
            WHERE user_id = %s AND symbol = %s
        """
        cursor.execute(check_query, (user_id, symbol))
        existing = cursor.fetchone()
        
        try:
            if existing:
                # Update existing position (average down/up)
                existing_id, existing_shares, existing_avg = existing
                total_shares = existing_shares + shares
                new_avg_price = ((existing_shares * existing_avg) + (shares * purchase_price)) / total_shares
                
                update_query = """
                    UPDATE user_portfolios 
                    SET shares = %s, avg_purchase_price = %s, last_updated = %s
                    WHERE portfolio_id = %s
                """
                cursor.execute(update_query, (total_shares, new_avg_price, datetime.now(), existing_id))
                self.logger.info(f"Updated position for user {user_id}, symbol {symbol}")
                return existing_id
            else:
                # Create new position
                insert_query = """
                    INSERT INTO user_portfolios (user_id, symbol, shares, avg_purchase_price)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(insert_query, (user_id, symbol, shares, purchase_price))
                portfolio_id = cursor.lastrowid
                self.logger.info(f"Created new position for user {user_id}, symbol {symbol}")
                return portfolio_id
        except Error as e:
            self.logger.error(f"Failed to add portfolio position: {e}")
            raise
        finally:
            self.connection.commit()
            cursor.close()
    
    def update_portfolio_prices(self, price_data: Dict[str, float]):
        """Update current prices and calculate P&L"""
        cursor = self.connection.cursor()
        
        for symbol, current_price in price_data.items():
            try:
                update_query = """
                    UPDATE user_portfolios 
                    SET current_price = %s, 
                        market_value = shares * %s,
                        unrealized_pnl = (shares * %s) - (shares * avg_purchase_price),
                        last_updated = %s
                    WHERE symbol = %s
                """
                cursor.execute(update_query, (
                    current_price, current_price, current_price, datetime.now(), symbol
                ))
                self.logger.debug(f"Updated prices for {symbol}: ${current_price}")
            except Error as e:
                self.logger.error(f"Failed to update prices for {symbol}: {e}")
        
        self.connection.commit()
        cursor.close()

    def get_user_portfolio(self, user_id: int) -> List[Dict]:
        """Get user's complete portfolio"""
        cursor = self.connection.cursor(dictionary=True)
        query = """
            SELECT 
                portfolio_id, symbol, shares, avg_purchase_price, current_price,
                market_value, unrealized_pnl, realized_pnl, purchase_date, last_updated
            FROM user_portfolios 
            WHERE user_id = %s
            ORDER BY symbol
        """
        try:
            cursor.execute(query, (user_id,))
            portfolio = cursor.fetchall()
            self.logger.info(f"Retrieved portfolio for user {user_id}: {len(portfolio)} positions")
            return portfolio
        except Error as e:
            self.logger.error(f"Failed to get portfolio for user {user_id}: {e}")
            return []
        finally:
            cursor.close()

    # Recommendation Management Methods
    def save_recommendation(self, user_id: int, symbol: str, rec_type: str, 
                          shares: float, price: float, confidence: float,
                          model_prediction: float, features: dict, risk_score: float) -> int:
        """Save AI recommendation"""
        cursor = self.connection.cursor()
        query = """
            INSERT INTO recommendations_feedback (
                user_id, symbol, recommendation_type, recommended_shares, 
                recommended_price, confidence_score, model_prediction, 
                model_features, risk_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            cursor.execute(query, (
                user_id, symbol, rec_type, shares, price, confidence,
                model_prediction, json.dumps(features), risk_score
            ))
            rec_id = cursor.lastrowid
            self.connection.commit()
            self.logger.info(f"Saved recommendation {rec_id} for user {user_id}, symbol {symbol}")
            return rec_id
        except Error as e:
            self.logger.error(f"Failed to save recommendation: {e}")
            raise
        finally:
            cursor.close()
    
    def record_user_response(self, recommendation_id: int, accepted: bool, 
                           actual_shares: float = 0, actual_price: float = 0):
        """Record user response to recommendation"""
        cursor = self.connection.cursor()
        query = """
            UPDATE recommendations_feedback 
            SET user_accepted = %s, user_response_timestamp = %s, 
                actual_shares_traded = %s, actual_price = %s
            WHERE recommendation_id = %s
        """
        try:
            cursor.execute(query, (
                accepted, datetime.now(), actual_shares, actual_price, recommendation_id
            ))
            self.connection.commit()
            self.logger.info(f"Recorded user response for recommendation {recommendation_id}")
        except Error as e:
            self.logger.error(f"Failed to record user response: {e}")
            raise
        finally:
            cursor.close()
    
    def evaluate_recommendation_outcome(self, recommendation_id: int, days_after: int = 30) -> Dict:
        """Evaluate recommendation outcome after specified period"""
        cursor = self.connection.cursor(dictionary=True)
        
        # Get recommendation details
        query = """
            SELECT * FROM recommendations_feedback 
            WHERE recommendation_id = %s AND user_accepted = TRUE
        """
        cursor.execute(query, (recommendation_id,))
        rec = cursor.fetchone()
        
        if not rec:
            return {}
        
        try:
            # Calculate outcome based on time period
            eval_date = rec['user_response_timestamp'] + timedelta(days=days_after)
            
            # This would integrate with your price data to get actual performance
            # For now, we'll use a placeholder calculation
            outcome_pnl = 0  # Calculate based on actual price movement
            outcome_return_pct = 0  # Calculate return percentage
            
            # Update recommendation with outcome
            update_query = """
                UPDATE recommendations_feedback 
                SET outcome_evaluated = TRUE, outcome_pnl = %s, 
                    outcome_return_pct = %s, evaluation_timestamp = %s
                WHERE recommendation_id = %s
            """
            cursor.execute(update_query, (
                outcome_pnl, outcome_return_pct, datetime.now(), recommendation_id
            ))
            self.connection.commit()
            
            return {
                'recommendation_id': recommendation_id,
                'outcome_pnl': outcome_pnl,
                'outcome_return_pct': outcome_return_pct,
                'evaluation_date': eval_date
            }
        except Error as e:
            self.logger.error(f"Failed to evaluate recommendation outcome: {e}")
            return {}
        finally:
            cursor.close()

    def get_feedback_data(self, days: int = 90) -> List[Dict]:
        """Get feedback data for model retraining"""
        cursor = self.connection.cursor(dictionary=True)
        query = """
            SELECT 
                symbol, recommendation_type, confidence_score, user_accepted,
                actual_shares_traded, actual_price, outcome_pnl, outcome_return_pct,
                model_prediction, model_features, risk_score
            FROM recommendations_feedback 
            WHERE recommendation_timestamp >= %s AND outcome_evaluated = TRUE
            ORDER BY recommendation_timestamp DESC
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cursor.execute(query, (cutoff_date,))
            feedback_data = cursor.fetchall()
            self.logger.info(f"Retrieved {len(feedback_data)} feedback records")
            return feedback_data
        except Error as e:
            self.logger.error(f"Failed to get feedback data: {e}")
            return []
        finally:
            cursor.close()

    def get_user_statistics(self, user_id: int) -> Dict:
        """Get comprehensive user statistics"""
        cursor = self.connection.cursor(dictionary=True)
        
        try:
            # Get user basic info
            user_query = """
                SELECT username, risk_tolerance, investment_experience, 
                       portfolio_value, total_pnl, created_at, last_login
                FROM users WHERE user_id = %s
            """
            cursor.execute(user_query, (user_id,))
            user_info = cursor.fetchone()
            
            # Get portfolio summary
            portfolio_query = """
                SELECT 
                    COUNT(*) as total_positions,
                    SUM(market_value) as total_value,
                    SUM(unrealized_pnl) as total_unrealized_pnl,
                    SUM(realized_pnl) as total_realized_pnl
                FROM user_portfolios WHERE user_id = %s
            """
            cursor.execute(portfolio_query, (user_id,))
            portfolio_stats = cursor.fetchone()
            
            # Get recommendation stats
            rec_query = """
                SELECT 
                    COUNT(*) as total_recommendations,
                    SUM(user_accepted) as accepted_recommendations,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(CASE WHEN outcome_return_pct > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN outcome_evaluated = TRUE THEN 1 END) as evaluated_trades
                FROM recommendations_feedback WHERE user_id = %s
            """
            cursor.execute(rec_query, (user_id,))
            rec_stats = cursor.fetchone()
            
            return {
                'user_info': user_info,
                'portfolio_stats': portfolio_stats,
                'recommendation_stats': rec_stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get user statistics: {e}")
            return {}
        finally:
            cursor.close()
    
    def update_user_portfolio_value(self, user_id: int):
        """Update user's total portfolio value and P&L"""
        cursor = self.connection.cursor()
        
        try:
            # Calculate total portfolio value
            query = """
                SELECT 
                    SUM(market_value) as total_value,
                    SUM(unrealized_pnl) as total_pnl
                FROM user_portfolios WHERE user_id = %s
            """
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
            
            if result:
                total_value = result[0] or 0
                total_pnl = result[1] or 0
                
                # Update user record
                update_query = """
                    UPDATE users 
                    SET portfolio_value = %s, total_pnl = %s 
                    WHERE user_id = %s
                """
                cursor.execute(update_query, (total_value, total_pnl, user_id))
                self.connection.commit()
                
                self.logger.info(f"Updated portfolio value for user {user_id}: ${total_value}")
                
        except Exception as e:
            self.logger.error(f"Failed to update portfolio value: {e}")
        finally:
            cursor.close()
    
    def get_learning_data(self, days: int = 90) -> List[Dict]:
        """Get data for reinforcement learning model training"""
        cursor = self.connection.cursor(dictionary=True)
        
        query = """
            SELECT 
                rf.*,
                u.risk_tolerance,
                u.investment_experience,
                up.shares as current_shares,
                up.avg_purchase_price
            FROM recommendations_feedback rf
            JOIN users u ON rf.user_id = u.user_id
            LEFT JOIN user_portfolios up ON rf.user_id = up.user_id AND rf.symbol = up.symbol
            WHERE rf.recommendation_timestamp >= %s
            ORDER BY rf.recommendation_timestamp DESC
        """
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cursor.execute(query, (cutoff_date,))
            learning_data = cursor.fetchall()
            self.logger.info(f"Retrieved {len(learning_data)} learning records")
            return learning_data
        except Exception as e:
            self.logger.error(f"Failed to get learning data: {e}")
            return []
        finally:
            cursor.close()
    
    def record_user_feedback(self, recommendation_id: int, feedback_score: int, 
                           feedback_comment: str = None):
        """Record user feedback on recommendation quality"""
        cursor = self.connection.cursor()
        
        query = """
            UPDATE recommendations_feedback 
            SET user_feedback_score = %s, user_feedback_comment = %s
            WHERE recommendation_id = %s
        """
        
        try:
            cursor.execute(query, (feedback_score, feedback_comment, recommendation_id))
            self.connection.commit()
            self.logger.info(f"Recorded user feedback for recommendation {recommendation_id}")
        except Exception as e:
            self.logger.error(f"Failed to record user feedback: {e}")
        finally:
            cursor.close()

    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.logger.info("Database connection closed")
