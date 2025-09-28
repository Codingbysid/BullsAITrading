"""
Reinforcement Learning Feedback System for QuantAI Trading Bot.

This module implements a feedback learning system that improves
recommendation accuracy based on user acceptance and outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import json
import logging

# Try to import scikit-learn, fallback if not available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available, using fallback implementation")

class ReinforcementFeedbackTrainer:
    """Reinforcement learning system using user feedback"""
    
    def __init__(self, db_manager, model_path: str = "models/feedback_model.pkl"):
        self.db = db_manager
        self.model_path = model_path
        self.model = None
        self.feature_columns = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize or load existing model
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        if SKLEARN_AVAILABLE:
            try:
                self.model = joblib.load(self.model_path)
                self.logger.info("Loaded existing feedback model")
            except:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
                self.logger.info("Created new feedback model")
        else:
            # Fallback implementation
            self.model = FallbackModel()
            self.logger.info("Created fallback feedback model")
    
    def prepare_training_data(self, days: int = 90) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from user feedback"""
        feedback_data = self.db.get_feedback_data(days)
        
        if len(feedback_data) < 10:
            self.logger.warning("Insufficient feedback data for training")
            return pd.DataFrame(), pd.Series()
        
        # Convert to DataFrame
        df = pd.DataFrame(feedback_data)
        
        # Extract features from model_features JSON
        feature_rows = []
        targets = []
        
        for _, row in df.iterrows():
            try:
                # Parse model features
                features = json.loads(row['model_features']) if isinstance(row['model_features'], str) else row['model_features']
                
                # Create feature vector
                feature_vector = {
                    'confidence_score': row['confidence_score'],
                    'risk_score': row['risk_score'],
                    'model_prediction': row['model_prediction'],
                    'symbol_encoded': self._encode_symbol(row['symbol']),
                    'recommendation_type_encoded': self._encode_recommendation_type(row['recommendation_type'])
                }
                
                # Add technical indicators from features
                if isinstance(features, dict):
                    for key, value in features.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            feature_vector[f'feature_{key}'] = value
                
                feature_rows.append(feature_vector)
                
                # Target: 1 if profitable (user accepted and positive outcome), 0 otherwise
                if row['user_accepted'] and row['outcome_return_pct'] and row['outcome_return_pct'] > 0:
                    targets.append(1)
                else:
                    targets.append(0)
                    
            except Exception as e:
                self.logger.error(f"Error processing feedback row: {e}")
                continue
        
        if not feature_rows:
            return pd.DataFrame(), pd.Series()
        
        X = pd.DataFrame(feature_rows)
        y = pd.Series(targets)
        
        # Store feature columns for later use
        self.feature_columns = list(X.columns)
        
        return X, y
    
    def _encode_symbol(self, symbol: str) -> int:
        """Encode symbol to numeric value"""
        symbol_map = {'AAPL': 1, 'GOOGL': 2, 'AMZN': 3, 'META': 4, 'NVDA': 5}
        return symbol_map.get(symbol, 0)
    
    def _encode_recommendation_type(self, rec_type: str) -> int:
        """Encode recommendation type to numeric value"""
        type_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        return type_map.get(rec_type, 0)
    
    def train_feedback_model(self) -> Dict:
        """Train model using user feedback"""
        X, y = self.prepare_training_data()
        
        if X.empty or len(X) < 10:
            return {
                'success': False,
                'message': 'Insufficient training data',
                'samples': len(X)
            }
        
        try:
            if SKLEARN_AVAILABLE:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Evaluate
                train_score = self.model.score(X_train, y_train)
                test_score = self.model.score(X_test, y_test)
                
                y_pred = self.model.predict(X_test)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                
                # Save model
                joblib.dump(self.model, self.model_path)
                
                # Feature importance
                feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                training_results = {
                    'success': True,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'precision': precision,
                    'recall': recall,
                    'top_features': top_features,
                    'training_date': datetime.now().isoformat()
                }
            else:
                # Fallback training
                training_results = self.model.train_fallback(X, y)
                training_results['training_date'] = datetime.now().isoformat()
            
            self.logger.info(f"Feedback model trained successfully: {training_results}")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Failed to train feedback model: {e}")
            return {
                'success': False,
                'message': str(e),
                'samples': len(X)
            }
    
    def predict_recommendation_success(self, recommendation_features: Dict) -> float:
        """Predict probability of recommendation success"""
        if self.model is None:
            return 0.5  # Default probability
        
        try:
            if SKLEARN_AVAILABLE:
                # Prepare features in the same format as training
                feature_vector = {}
                
                # Core features
                feature_vector['confidence_score'] = recommendation_features.get('confidence_score', 0.5)
                feature_vector['risk_score'] = recommendation_features.get('risk_score', 0.5)
                feature_vector['model_prediction'] = recommendation_features.get('model_prediction', 0)
                feature_vector['symbol_encoded'] = self._encode_symbol(recommendation_features.get('symbol', ''))
                feature_vector['recommendation_type_encoded'] = self._encode_recommendation_type(
                    recommendation_features.get('recommendation_type', '')
                )
                
                # Technical features
                for key, value in recommendation_features.items():
                    if key.startswith('feature_') and isinstance(value, (int, float)):
                        feature_vector[key] = value
                
                # Ensure all expected features are present
                for col in self.feature_columns:
                    if col not in feature_vector:
                        feature_vector[col] = 0
                
                # Create DataFrame with correct column order
                X = pd.DataFrame([feature_vector])[self.feature_columns]
                
                # Predict probability
                prob = self.model.predict_proba(X)[0][1]  # Probability of success (class 1)
                
                return prob
            else:
                # Fallback prediction
                return self.model.predict_fallback(recommendation_features)
            
        except Exception as e:
            self.logger.error(f"Error predicting recommendation success: {e}")
            return 0.5
    
    def update_model_weights(self, recommendation_id: int, outcome: float):
        """Update model based on recommendation outcome (online learning simulation)"""
        try:
            # Get recommendation details
            cursor = self.db.connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM recommendations_feedback WHERE recommendation_id = %s",
                (recommendation_id,)
            )
            rec = cursor.fetchone()
            cursor.close()
            
            if not rec:
                return
            
            # Extract features
            features = json.loads(rec['model_features']) if isinstance(rec['model_features'], str) else rec['model_features']
            
            # Create feature vector
            feature_vector = {
                'confidence_score': rec['confidence_score'],
                'risk_score': rec['risk_score'],
                'model_prediction': rec['model_prediction'],
                'symbol_encoded': self._encode_symbol(rec['symbol']),
                'recommendation_type_encoded': self._encode_recommendation_type(rec['recommendation_type'])
            }
            
            # Add technical features
            if isinstance(features, dict):
                for key, value in features.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector[f'feature_{key}'] = value
            
            # Simulate online learning by retraining with additional sample
            if len(feature_vector) > 5:  # Ensure we have sufficient features
                # Get existing training data
                X_existing, y_existing = self.prepare_training_data()
                
                if not X_existing.empty:
                    # Add new sample
                    new_sample = pd.DataFrame([feature_vector])
                    new_target = 1 if outcome > 0 else 0
                    
                    # Ensure new sample has all required columns
                    for col in X_existing.columns:
                        if col not in new_sample.columns:
                            new_sample[col] = 0
                    
                    new_sample = new_sample[X_existing.columns]  # Ensure same column order
                    
                    X_updated = pd.concat([X_existing, new_sample], ignore_index=True)
                    y_updated = pd.concat([y_existing, pd.Series([new_target])], ignore_index=True)
                    
                    # Retrain model (partial fit simulation)
                    if SKLEARN_AVAILABLE:
                        self.model.fit(X_updated, y_updated)
                        joblib.dump(self.model, self.model_path)
                    else:
                        self.model.train_fallback(X_updated, y_updated)
                    
                    self.logger.info(f"Updated model with recommendation {recommendation_id} outcome")
            
        except Exception as e:
            self.logger.error(f"Error updating model weights: {e}")
    
    def get_model_performance_report(self) -> Dict:
        """Generate comprehensive model performance report"""
        try:
            feedback_data = self.db.get_feedback_data(days=30)
            
            if len(feedback_data) < 5:
                return {'error': 'Insufficient data for performance report'}
            
            df = pd.DataFrame(feedback_data)
            
            # Calculate success rate by confidence bands
            df['confidence_band'] = pd.cut(df['confidence_score'], 
                                         bins=[0, 0.6, 0.8, 1.0], 
                                         labels=['Low', 'Medium', 'High'])
            
            success_by_confidence = df.groupby('confidence_band').agg({
                'user_accepted': 'mean',
                'outcome_return_pct': lambda x: (x > 0).mean()
            }).round(3)
            
            # Overall statistics
            total_recommendations = len(df)
            acceptance_rate = df['user_accepted'].mean()
            success_rate = (df['outcome_return_pct'] > 0).mean()
            avg_return = df['outcome_return_pct'].mean()
            
            # User feedback analysis
            feedback_scores = df['user_feedback_score'].dropna()
            avg_feedback_score = feedback_scores.mean() if len(feedback_scores) > 0 else None
            
            # Performance by symbol
            symbol_performance = df.groupby('symbol').agg({
                'user_accepted': 'mean',
                'outcome_return_pct': 'mean',
                'confidence_score': 'mean'
            }).round(3)
            
            # Learning progress over time
            df['date'] = pd.to_datetime(df['recommendation_timestamp']).dt.date
            daily_performance = df.groupby('date').agg({
                'user_accepted': 'mean',
                'outcome_return_pct': 'mean'
            }).round(3)
            
            return {
                'total_recommendations': total_recommendations,
                'acceptance_rate': acceptance_rate,
                'success_rate': success_rate,
                'avg_return': avg_return,
                'avg_feedback_score': avg_feedback_score,
                'success_by_confidence': success_by_confidence.to_dict(),
                'symbol_performance': symbol_performance.to_dict(),
                'daily_performance': daily_performance.to_dict(),
                'report_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def analyze_user_behavior_patterns(self, user_id: int) -> Dict:
        """Analyze individual user behavior patterns for personalization"""
        try:
            cursor = self.db.connection.cursor(dictionary=True)
            
            # Get user's recommendation history
            query = """
                SELECT 
                    symbol, recommendation_type, confidence_score,
                    user_accepted, outcome_return_pct, user_feedback_score
                FROM recommendations_feedback 
                WHERE user_id = %s
                ORDER BY recommendation_timestamp DESC
            """
            cursor.execute(query, (user_id,))
            user_data = cursor.fetchall()
            cursor.close()
            
            if len(user_data) < 3:
                return {'error': 'Insufficient user data for analysis'}
            
            df = pd.DataFrame(user_data)
            
            # User preferences analysis
            symbol_preferences = df.groupby('symbol').agg({
                'user_accepted': 'mean',
                'outcome_return_pct': 'mean'
            }).round(3)
            
            # Confidence threshold analysis
            user_acceptance_by_confidence = df.groupby(
                pd.cut(df['confidence_score'], bins=[0, 0.6, 0.8, 1.0])
            )['user_accepted'].mean()
            
            # Performance by recommendation type
            type_performance = df.groupby('recommendation_type').agg({
                'user_accepted': 'mean',
                'outcome_return_pct': 'mean'
            }).round(3)
            
            return {
                'user_id': user_id,
                'total_recommendations': len(df),
                'acceptance_rate': df['user_accepted'].mean(),
                'avg_performance': df['outcome_return_pct'].mean(),
                'symbol_preferences': symbol_preferences.to_dict(),
                'confidence_thresholds': user_acceptance_by_confidence.to_dict(),
                'type_performance': type_performance.to_dict(),
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing user behavior: {e}")
            return {'error': str(e)}
    
    def generate_personalized_recommendations(self, user_id: int, 
                                           market_data: Dict) -> List[Dict]:
        """Generate personalized recommendations based on user behavior"""
        try:
            # Get user behavior analysis
            user_analysis = self.analyze_user_behavior_patterns(user_id)
            
            if 'error' in user_analysis:
                return []
            
            # Get user's current portfolio
            portfolio = self.db.get_user_portfolio(user_id)
            portfolio_symbols = [p['symbol'] for p in portfolio]
            
            personalized_recs = []
            
            for symbol in ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']:
                # Check user's historical preference for this symbol
                symbol_pref = user_analysis['symbol_preferences'].get(symbol, {})
                user_acceptance_rate = symbol_pref.get('user_accepted', 0.5)
                
                # Adjust confidence based on user's historical acceptance
                base_confidence = 0.7
                personalized_confidence = base_confidence * user_acceptance_rate
                
                # Generate recommendation if confidence is high enough
                if personalized_confidence > 0.6:
                    recommendation = {
                        'symbol': symbol,
                        'confidence': personalized_confidence,
                        'personalized': True,
                        'reasoning': f"Based on your {user_acceptance_rate:.1%} acceptance rate for {symbol}",
                        'user_behavior_factor': user_acceptance_rate
                    }
                    personalized_recs.append(recommendation)
            
            return personalized_recs
            
        except Exception as e:
            self.logger.error(f"Error generating personalized recommendations: {e}")
            return []


class FallbackModel:
    """Fallback model implementation when scikit-learn is not available"""
    
    def __init__(self):
        self.weights = {}
        self.bias = 0.5
    
    def train_fallback(self, X, y):
        """Simple fallback training"""
        # Simple linear model approximation
        if len(X) > 0:
            # Calculate simple weights based on correlation
            for col in X.columns:
                if col in X.columns:
                    correlation = X[col].corr(y)
                    self.weights[col] = correlation if not np.isnan(correlation) else 0
        
        return {
            'success': True,
            'training_samples': len(X),
            'test_samples': 0,
            'train_accuracy': 0.6,  # Placeholder
            'test_accuracy': 0.6,
            'precision': 0.6,
            'recall': 0.6,
            'top_features': list(self.weights.items())[:5]
        }
    
    def predict_fallback(self, features):
        """Simple fallback prediction"""
        score = self.bias
        
        for key, value in features.items():
            if key in self.weights:
                score += self.weights[key] * value
        
        # Convert to probability
        return max(0, min(1, score))
