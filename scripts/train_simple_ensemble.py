#!/usr/bin/env python3
"""
Simple ML Ensemble Model Training Script

This script trains a simplified ML ensemble model on real dataset files with
comprehensive preprocessing and feature engineering, using only basic Python libraries.

Usage:
    python scripts/train_simple_ensemble.py --datasets datasets/ --output models/
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import unified utilities
from utils.common_imports import setup_logger
from utils.data_processing import data_processor
from utils.feature_engineering import feature_engineer
from utils.performance_metrics import performance_calculator
from utils.config_manager import config_manager

logger = setup_logger(__name__)


class SimpleMLEnsembleTrainer:
    """Simple ML Ensemble Model Trainer using basic Python libraries."""
    
    def __init__(self):
        self.config = config_manager.get_config('model')
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Model weights for ensemble
        self.ensemble_weights = {
            'linear_model': 0.50,
            'naive_bayes': 0.30,
            'decision_tree': 0.20
        }
        
        # Trained models
        self.models = {}
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
    
    def simple_train_test_split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Simple train-test split without scikit-learn."""
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        
        # Split indices
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        # Split data
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train simple linear model."""
        logger.info("Training linear model")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = self.simple_train_test_split(X, y)
            
            # Simple linear regression using normal equations
            # Add bias term
            X_train_bias = np.column_stack([np.ones(len(X_train)), X_train.values])
            X_test_bias = np.column_stack([np.ones(len(X_test)), X_test.values])
            
            # Calculate weights using normal equations
            try:
                weights = np.linalg.solve(X_train_bias.T @ X_train_bias, X_train_bias.T @ y_train.values)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                weights = np.linalg.pinv(X_train_bias) @ y_train.values
            
            # Make predictions
            y_pred_train = X_train_bias @ weights
            y_pred_test = X_test_bias @ weights
            
            # Convert to binary predictions
            y_pred_train_binary = (y_pred_train > 0.5).astype(int)
            y_pred_test_binary = (y_pred_test > 0.5).astype(int)
            
            # Calculate accuracy
            train_accuracy = np.mean(y_pred_train_binary == y_train.values)
            test_accuracy = np.mean(y_pred_test_binary == y_test.values)
            
            # Store model
            self.models['linear_model'] = {
                'weights': weights,
                'feature_columns': self.feature_columns
            }
            
            results = {
                'status': 'trained',
                'model_type': 'LinearModel',
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'n_features': len(self.feature_columns)
            }
            
            logger.info(f"✅ Linear model trained: Test accuracy = {test_accuracy:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Linear model training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def train_naive_bayes(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train simple Naive Bayes model."""
        logger.info("Training Naive Bayes model")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = self.simple_train_test_split(X, y)
            
            # Simple Naive Bayes implementation
            # Calculate class priors
            class_counts = y_train.value_counts()
            class_priors = class_counts / len(y_train)
            
            # Calculate feature means and stds for each class
            feature_stats = {}
            for class_label in class_counts.index:
                class_data = X_train[y_train == class_label]
                feature_stats[class_label] = {
                    'mean': class_data.mean(),
                    'std': class_data.std() + 1e-8  # Add small value to avoid division by zero
                }
            
            # Make predictions on test set
            y_pred_test = []
            for _, sample in X_test.iterrows():
                class_probs = {}
                for class_label in class_counts.index:
                    # Calculate likelihood for each feature
                    likelihood = 1.0
                    for feature in self.feature_columns:
                        if feature in sample.index:
                            mean = feature_stats[class_label]['mean'][feature]
                            std = feature_stats[class_label]['std'][feature]
                            # Simple Gaussian likelihood
                            likelihood *= np.exp(-0.5 * ((sample[feature] - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                    
                    # Calculate posterior probability
                    class_probs[class_label] = class_priors[class_label] * likelihood
                
                # Predict class with highest probability
                predicted_class = max(class_probs, key=class_probs.get)
                y_pred_test.append(predicted_class)
            
            # Calculate accuracy
            test_accuracy = np.mean(np.array(y_pred_test) == y_test.values)
            
            # Store model
            self.models['naive_bayes'] = {
                'class_priors': class_priors,
                'feature_stats': feature_stats,
                'feature_columns': self.feature_columns
            }
            
            results = {
                'status': 'trained',
                'model_type': 'NaiveBayes',
                'test_accuracy': test_accuracy,
                'n_features': len(self.feature_columns)
            }
            
            logger.info(f"✅ Naive Bayes model trained: Test accuracy = {test_accuracy:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Naive Bayes training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def train_decision_tree(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train simple decision tree model."""
        logger.info("Training decision tree model")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = self.simple_train_test_split(X, y)
            
            # Simple decision tree implementation
            # Find best split based on information gain
            def calculate_entropy(labels):
                if len(labels) == 0:
                    return 0
                counts = np.bincount(labels)
                probabilities = counts / len(labels)
                return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
            
            def find_best_split(X, y):
                best_gain = 0
                best_feature = None
                best_threshold = None
                
                for feature in X.columns:
                    values = X[feature].values
                    unique_values = np.unique(values)
                    
                    for threshold in unique_values:
                        left_mask = values <= threshold
                        right_mask = values > threshold
                        
                        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                            continue
                        
                        # Calculate information gain
                        parent_entropy = calculate_entropy(y.values)
                        left_entropy = calculate_entropy(y[left_mask].values)
                        right_entropy = calculate_entropy(y[right_mask].values)
                        
                        left_weight = np.sum(left_mask) / len(y)
                        right_weight = np.sum(right_mask) / len(y)
                        
                        gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_feature = feature
                            best_threshold = threshold
                
                return best_feature, best_threshold, best_gain
            
            # Build simple decision tree
            def build_tree(X, y, depth=0, max_depth=5):
                if depth >= max_depth or len(np.unique(y)) == 1:
                    return {'prediction': y.mode()[0]}
                
                feature, threshold, gain = find_best_split(X, y)
                
                if feature is None or gain < 0.01:
                    return {'prediction': y.mode()[0]}
                
                left_mask = X[feature] <= threshold
                right_mask = X[feature] > threshold
                
                return {
                    'feature': feature,
                    'threshold': threshold,
                    'left': build_tree(X[left_mask], y[left_mask], depth + 1, max_depth),
                    'right': build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)
                }
            
            # Train tree
            tree = build_tree(X_train, y_train)
            
            # Make predictions
            def predict(tree, X):
                predictions = []
                for _, sample in X.iterrows():
                    node = tree
                    while 'prediction' not in node:
                        if sample[node['feature']] <= node['threshold']:
                            node = node['left']
                        else:
                            node = node['right']
                    predictions.append(node['prediction'])
                return predictions
            
            y_pred_test = predict(tree, X_test)
            
            # Calculate accuracy
            test_accuracy = np.mean(np.array(y_pred_test) == y_test.values)
            
            # Store model
            self.models['decision_tree'] = {
                'tree': tree,
                'feature_columns': self.feature_columns
            }
            
            results = {
                'status': 'trained',
                'model_type': 'DecisionTree',
                'test_accuracy': test_accuracy,
                'n_features': len(self.feature_columns)
            }
            
            logger.info(f"✅ Decision tree model trained: Test accuracy = {test_accuracy:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Decision tree training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the complete simple ML ensemble."""
        logger.info("Training simple ML ensemble")
        
        # Train individual models
        linear_results = self.train_linear_model(X, y)
        naive_bayes_results = self.train_naive_bayes(X, y)
        decision_tree_results = self.train_decision_tree(X, y)
        
        # Store results
        self.training_results = {
            'linear_model': linear_results,
            'naive_bayes': naive_bayes_results,
            'decision_tree': decision_tree_results
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
                model_path = self.models_dir / f"{model_name}_model.json"
                with open(model_path, 'w') as f:
                    json.dump(model, f, indent=2, default=str)
            
            # Save metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'feature_columns': self.feature_columns,
                'ensemble_weights': self.ensemble_weights,
                'training_results': self.training_results,
                'models_available': list(self.models.keys())
            }
            
            metadata_path = self.models_dir / "simple_ensemble_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Models saved to {self.models_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save models: {e}")
    
    def run_training_pipeline(self, datasets_dir: str) -> Dict[str, Any]:
        """Run the complete simple ML ensemble training pipeline."""
        logger.info("🚀 Starting simple ML ensemble training pipeline")
        
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
            
            logger.info("✅ Simple ML ensemble training pipeline completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Simple ML ensemble training pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'training_date': datetime.now().isoformat()
            }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train Simple ML Ensemble Model on Real Datasets')
    parser.add_argument('--datasets', type=str, default='datasets/',
                       help='Directory containing dataset files')
    parser.add_argument('--output', type=str, default='models/',
                       help='Output directory for trained models')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SimpleMLEnsembleTrainer()
    trainer.models_dir = Path(args.output)
    trainer.models_dir.mkdir(exist_ok=True)
    
    print("🎯 Simple ML Ensemble Model Training on Real Datasets")
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
                else:
                    print(f"   ❌ {model_name.upper()}: FAILED")
            
            # Ensemble performance
            ensemble_perf = ensemble_results['ensemble_performance']
            print(f"\n🎯 Ensemble Performance:")
            print(f"   Models Trained: {ensemble_perf['models_trained']}")
            print(f"   Weighted Accuracy: {ensemble_perf['weighted_accuracy']:.3f}")
            print(f"   Ensemble Ready: {'✅' if ensemble_perf['ensemble_ready'] else '❌'}")
            
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
