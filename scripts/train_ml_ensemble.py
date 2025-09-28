from src.utils.common_imports import *
from utils.common_imports import setup_logger
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import joblib
import json
import warnings
from utils.data_processing import data_processor
from utils.feature_engineering import feature_engineer
from utils.performance_metrics import performance_calculator
from utils.config_manager import config_manager
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

#!/usr/bin/env python3
"""
ML Ensemble Model Training Script

This script trains the ML ensemble model on real dataset files with comprehensive
preprocessing and feature engineering.

Usage:
    python scripts/train_ml_ensemble.py --datasets datasets/ --output models/
"""

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import unified utilities

# Import ML models
try:
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, using simplified models")

try:
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available, using alternative")

try:
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available, using alternative")

logger = setup_logger(__name__)


class MLEnsembleTrainer:
    """ML Ensemble Model Trainer with real dataset preprocessing."""
    
    def __init__(self):
        self.config = config_manager.get_config('model')
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Model weights for ensemble
        self.ensemble_weights = {
            'random_forest': 0.40,
            'xgboost': 0.35,
            'lstm': 0.25
        }
        
        # Trained models
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Training results
        self.training_results = {}
    
    def load_datasets(self, datasets_dir: str) -> Dict[str, pd.DataFrame]:
        """Load all dataset files."""
        logger.info(f"Loading datasets from {datasets_dir}")
        
        datasets = {}
        dataset_files = {
            'AMZN': 'Amazon.csv',
            'AAPL': 'Apple.csv', 
            'META': 'Facebook.csv',
            'GOOGL': 'Google.csv',
            'NFLX': 'Netflix.csv'
        }
        
        for symbol, filename in dataset_files.items():
            filepath = Path(datasets_dir) / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df['Symbol'] = symbol
                    df = df.sort_values('Date').reset_index(drop=True)
                    
                    # Basic data validation
                    if data_processor.validate_price_data(df):
                        datasets[symbol] = df
                        logger.info(f"✅ Loaded {symbol}: {len(df)} records from {df['Date'].min().date()} to {df['Date'].max().date()}")
                    else:
                        logger.warning(f"❌ Data validation failed for {symbol}")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load {symbol}: {e}")
            else:
                logger.warning(f"❌ Dataset file not found: {filepath}")
        
        logger.info(f"Successfully loaded {len(datasets)} datasets")
        return datasets
    
    def preprocess_data(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Preprocess and combine all datasets."""
        logger.info("Preprocessing datasets")
        
        all_data = []
        
        for symbol, df in datasets.items():
            # Add symbol column if not present
            if 'Symbol' not in df.columns:
                df['Symbol'] = symbol
            
            # Ensure required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df.columns for col in required_cols):
                # Select and rename columns
                processed_df = df[required_cols + ['Symbol']].copy()
                
                # Handle missing values
                processed_df = processed_df.ffill().bfill()
                
                # Remove any remaining NaN values
                processed_df = processed_df.dropna()
                
                if len(processed_df) > 0:
                    all_data.append(processed_df)
                    logger.info(f"✅ Preprocessed {symbol}: {len(processed_df)} clean records")
                else:
                    logger.warning(f"❌ No clean data for {symbol}")
            else:
                logger.warning(f"❌ Missing required columns for {symbol}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values(['Symbol', 'Date']).reset_index(drop=True)
            
            logger.info(f"✅ Combined dataset: {len(combined_data)} total records")
            return combined_data
        else:
            raise ValueError("No valid data after preprocessing")
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features using unified feature engineering."""
        logger.info("Engineering features")
        
        # Group by symbol for feature engineering
        all_features = []
        
        for symbol in data['Symbol'].unique():
            symbol_data = data[data['Symbol'] == symbol].copy()
            
            # Create technical features using unified feature engineering
            features = feature_engineer.create_technical_features(symbol_data)
            
            if not features.empty:
                all_features.append(features)
                logger.info(f"✅ Created {len(features.columns)} features for {symbol}")
            else:
                logger.warning(f"❌ No features created for {symbol}")
        
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            
            # Remove any columns with all NaN values
            combined_features = combined_features.dropna(axis=1, how='all')
            
            # Fill remaining NaN values
            combined_features = combined_features.fillna(0)
            
            logger.info(f"✅ Feature engineering complete: {len(combined_features.columns)} features")
            return combined_features
        else:
            raise ValueError("No features created")
    
    def create_targets(self, features: pd.DataFrame) -> pd.Series:
        """Create target variables for ML training."""
        logger.info("Creating target variables")
        
        # Create future return targets (5-day forward returns)
        targets = []
        
        for symbol in features['Symbol'].unique():
            symbol_features = features[features['Symbol'] == symbol].copy()
            
            if 'Close' in symbol_features.columns:
                # Calculate 5-day future returns
                symbol_features['Future_Return'] = symbol_features['Close'].pct_change(5).shift(-5)
                
                # Create binary classification target (1 for positive return, 0 for negative)
                symbol_features['Target'] = (symbol_features['Future_Return'] > 0).astype(int)
                
                targets.append(symbol_features['Target'])
            else:
                logger.warning(f"❌ No Close price data for {symbol}")
        
        if targets:
            combined_targets = pd.concat(targets, ignore_index=True)
            # Remove NaN values
            combined_targets = combined_targets.dropna()
            
            logger.info(f"✅ Target creation complete: {len(combined_targets)} targets")
            return combined_targets
        else:
            raise ValueError("No targets created")
    
    def prepare_ml_data(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ML training."""
        logger.info("Preparing ML data")
        
        # Align features and targets
        min_length = min(len(features), len(targets))
        features = features.iloc[:min_length]
        targets = targets.iloc[:min_length]
        
        # Select feature columns (exclude non-feature columns)
        exclude_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Future_Return']
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("No feature columns found")
        
        X = features[feature_cols]
        y = targets
        
        # Remove any rows with NaN values
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_indices]
        y = y[valid_indices]
        
        self.feature_columns = feature_cols
        
        logger.info(f"✅ ML data prepared: {len(X)} samples, {len(feature_cols)} features")
        return X, y
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Random Forest model."""
        logger.info("Training Random Forest model")
        
        if not SKLEARN_AVAILABLE:
            return {'status': 'skipped', 'reason': 'scikit-learn not available'}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = rf_model.score(X_train_scaled, y_train)
            test_score = rf_model.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
            
            # Predictions
            y_pred = rf_model.predict(X_test_scaled)
            
            # Feature importance
            feature_importance = dict(zip(self.feature_columns, rf_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Store model and scaler
            self.models['random_forest'] = rf_model
            self.scalers['random_forest'] = scaler
            
            results = {
                'status': 'trained',
                'model_type': 'RandomForestClassifier',
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'top_features': top_features,
                'feature_importance': feature_importance
            }
            
            logger.info(f"✅ Random Forest trained: Test accuracy = {test_score:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Random Forest training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train XGBoost model."""
        logger.info("Training XGBoost model")
        
        if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
            return {'status': 'skipped', 'reason': 'XGBoost or scikit-learn not available'}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            xgb_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = xgb_model.score(X_train_scaled, y_train)
            test_score = xgb_model.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5)
            
            # Feature importance
            feature_importance = dict(zip(self.feature_columns, xgb_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Store model and scaler
            self.models['xgboost'] = xgb_model
            self.scalers['xgboost'] = scaler
            
            results = {
                'status': 'trained',
                'model_type': 'XGBClassifier',
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'top_features': top_features,
                'feature_importance': feature_importance
            }
            
            logger.info(f"✅ XGBoost trained: Test accuracy = {test_score:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ XGBoost training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def train_lstm(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train LSTM model."""
        logger.info("Training LSTM model")
        
        if not TENSORFLOW_AVAILABLE or not SKLEARN_AVAILABLE:
            return {'status': 'skipped', 'reason': 'TensorFlow or scikit-learn not available'}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Reshape data for LSTM (samples, timesteps, features)
            sequence_length = 10
            n_features = X_train_scaled.shape[1]
            
            def create_sequences(data, targets, seq_length):
                X_seq, y_seq = [], []
                for i in range(seq_length, len(data)):
                    X_seq.append(data[i-seq_length:i])
                    y_seq.append(targets.iloc[i])
                return np.array(X_seq), np.array(y_seq)
            
            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
            X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)
            
            # Build LSTM model
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            lstm_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = lstm_model.fit(
                X_train_seq, y_train_seq,
                epochs=10,
                batch_size=32,
                validation_data=(X_test_seq, y_test_seq),
                verbose=0
            )
            
            # Evaluate model
            train_loss, train_acc = lstm_model.evaluate(X_train_seq, y_train_seq, verbose=0)
            test_loss, test_acc = lstm_model.evaluate(X_test_seq, y_test_seq, verbose=0)
            
            # Store model and scaler
            self.models['lstm'] = lstm_model
            self.scalers['lstm'] = scaler
            
            results = {
                'status': 'trained',
                'model_type': 'LSTM',
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'sequence_length': sequence_length,
                'n_features': n_features
            }
            
            logger.info(f"✅ LSTM trained: Test accuracy = {test_acc:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ LSTM training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the complete ML ensemble."""
        logger.info("Training ML ensemble")
        
        # Train individual models
        rf_results = self.train_random_forest(X, y)
        xgb_results = self.train_xgboost(X, y)
        lstm_results = self.train_lstm(X, y)
        
        # Store results
        self.training_results = {
            'random_forest': rf_results,
            'xgboost': xgb_results,
            'lstm': lstm_results
        }
        
        # Calculate ensemble performance
        ensemble_performance = self._calculate_ensemble_performance()
        
        # Save models
        self._save_models()
        
        return {
            'individual_models': self.training_results,
            'ensemble_performance': ensemble_performance,
            'ensemble_weights': self.ensemble_weights,
            'feature_columns': self.feature_columns,
            'status': 'completed'
        }
    
    def _calculate_ensemble_performance(self) -> Dict[str, Any]:
        """Calculate ensemble performance metrics."""
        performance = {
            'models_trained': 0,
            'models_available': [],
            'weighted_accuracy': 0.0,
            'ensemble_ready': False
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for model_name, results in self.training_results.items():
            if results.get('status') == 'trained':
                performance['models_trained'] += 1
                performance['models_available'].append(model_name)
                
                weight = self.ensemble_weights.get(model_name, 0.0)
                accuracy = results.get('test_accuracy', 0.0)
                
                weighted_score += weight * accuracy
                total_weight += weight
        
        if total_weight > 0:
            performance['weighted_accuracy'] = weighted_score / total_weight
            performance['ensemble_ready'] = performance['models_trained'] >= 2
        
        return performance
    
    def _save_models(self) -> None:
        """Save trained models and metadata."""
        try:
            # Save models
            for model_name, model in self.models.items():
                model_path = self.models_dir / f"{model_name}_model.joblib"
                joblib.dump(model, model_path)
            
            # Save scalers
            for model_name, scaler in self.scalers.items():
                scaler_path = self.models_dir / f"{model_name}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'feature_columns': self.feature_columns,
                'ensemble_weights': self.ensemble_weights,
                'training_results': self.training_results,
                'models_available': list(self.models.keys())
            }
            
            metadata_path = self.models_dir / "ml_ensemble_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Models saved to {self.models_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save models: {e}")
    
    def run_training_pipeline(self, datasets_dir: str) -> Dict[str, Any]:
        """Run the complete ML ensemble training pipeline."""
        logger.info("🚀 Starting ML ensemble training pipeline")
        
        try:
            # Step 1: Load datasets
            datasets = self.load_datasets(datasets_dir)
            if not datasets:
                raise ValueError("No datasets loaded")
            
            # Step 2: Preprocess data
            processed_data = self.preprocess_data(datasets)
            
            # Step 3: Engineer features
            features = self.engineer_features(processed_data)
            
            # Step 4: Create targets
            targets = self.create_targets(features)
            
            # Step 5: Prepare ML data
            X, y = self.prepare_ml_data(features, targets)
            
            # Step 6: Train ensemble
            ensemble_results = self.train_ensemble(X, y)
            
            # Step 7: Generate summary
            summary = {
                'status': 'completed',
                'training_date': datetime.now().isoformat(),
                'datasets_loaded': list(datasets.keys()),
                'total_samples': len(X),
                'total_features': len(self.feature_columns),
                'ensemble_results': ensemble_results,
                'models_dir': str(self.models_dir)
            }
            
            logger.info("✅ ML ensemble training pipeline completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"❌ ML ensemble training pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'training_date': datetime.now().isoformat()
            }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train ML Ensemble Model on Real Datasets')
    parser.add_argument('--datasets', type=str, default='datasets/',
                       help='Directory containing dataset files')
    parser.add_argument('--output', type=str, default='models/',
                       help='Output directory for trained models')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MLEnsembleTrainer()
    trainer.models_dir = Path(args.output)
    trainer.models_dir.mkdir(exist_ok=True)
    
    print("🎯 ML Ensemble Model Training on Real Datasets")
    print("=" * 60)
    print(f"Datasets Directory: {args.datasets}")
    print(f"Output Directory: {args.output}")
    print("=" * 60)
    
    try:
        # Run training pipeline
        results = trainer.run_training_pipeline(args.datasets)
        
        # Print results
        print("\n📊 Training Results:")
        print("-" * 40)
        
        if results['status'] == 'completed':
            print("✅ Training Status: COMPLETED")
            
            # Dataset information
            print(f"\n📈 Dataset Information:")
            print(f"   Datasets Loaded: {', '.join(results['datasets_loaded'])}")
            print(f"   Total Samples: {results['total_samples']:,}")
            print(f"   Total Features: {results['total_features']}")
            
            # Ensemble results
            ensemble_results = results['ensemble_results']
            individual_models = ensemble_results['individual_models']
            
            print(f"\n🧠 Individual Model Results:")
            for model_name, model_results in individual_models.items():
                status = model_results.get('status', 'unknown')
                if status == 'trained':
                    test_acc = model_results.get('test_accuracy', 0.0)
                    weight = ensemble_results['ensemble_weights'].get(model_name, 0.0)
                    print(f"   ✅ {model_name.upper()}: {test_acc:.3f} accuracy (weight: {weight:.1%})")
                elif status == 'skipped':
                    reason = model_results.get('reason', 'unknown')
                    print(f"   ⚠️ {model_name.upper()}: SKIPPED ({reason})")
                else:
                    print(f"   ❌ {model_name.upper()}: FAILED")
            
            # Ensemble performance
            ensemble_perf = ensemble_results['ensemble_performance']
            print(f"\n🎯 Ensemble Performance:")
            print(f"   Models Trained: {ensemble_perf['models_trained']}")
            print(f"   Weighted Accuracy: {ensemble_perf['weighted_accuracy']:.3f}")
            print(f"   Ensemble Ready: {'✅' if ensemble_perf['ensemble_ready'] else '❌'}")
            
            # Top features
            print(f"\n🔍 Top Features (Random Forest):")
            if 'random_forest' in individual_models and individual_models['random_forest'].get('status') == 'trained':
                top_features = individual_models['random_forest'].get('top_features', [])
                for i, (feature, importance) in enumerate(top_features[:5], 1):
                    print(f"   {i}. {feature}: {importance:.3f}")
            
            print(f"\n💾 Models saved to: {results['models_dir']}")
            
        else:
            print("❌ Training Status: FAILED")
            print(f"Error: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
