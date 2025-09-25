"""
Enhanced Portfolio Manager with Complete User Journey Flow.

This is the main application for the QuantAI Portfolio Manager with
enhanced security, user journey management, and reinforcement learning.
"""

import sys
import logging
from datetime import datetime
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from database.db_manager import QuantAIDatabase
from portfolio.portfolio_manager import PortfolioManager
from interface.cli import QuantAITerminalInterface
from interface.user_journey import UserJourneyManager
from security.auth import SecurityManager
from training.feedback_trainer import ReinforcementFeedbackTrainer
from config.settings import settings

class EnhancedPortfolioManager:
    """Enhanced Portfolio Manager with complete user journey"""
    
    def __init__(self):
        self.settings = settings
        self.logger = self._setup_logging()
        
        # Initialize components
        self.db = self._initialize_database()
        self.security_manager = SecurityManager(self.db, self.settings)
        self.portfolio_manager = PortfolioManager(self.db)
        self.feedback_trainer = ReinforcementFeedbackTrainer(self.db)
        self.user_journey = UserJourneyManager(self.db, self.portfolio_manager)
        
        self.logger.info("Enhanced Portfolio Manager initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.settings.logging.level),
            format=self.settings.logging.format,
            handlers=[
                logging.FileHandler(self.settings.logging.file_path),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            db_config = {
                'host': self.settings.database.host,
                'database': self.settings.database.database,
                'user': self.settings.database.user,
                'password': self.settings.database.password
            }
            db = QuantAIDatabase(db_config)
            db.create_tables()
            return db
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def run_cli_mode(self):
        """Run enhanced CLI mode"""
        print("🚀 Enhanced QuantAI Portfolio Manager")
        print("=" * 60)
        print("🎯 Focused on AMZN, META, NVDA, GOOGL, AAPL")
        print("🤖 AI-powered recommendations with reinforcement learning")
        print("📊 Advanced portfolio tracking and analytics")
        print("🛡️ Enhanced security and user management")
        print("🔄 Continuous learning from user feedback")
        print("=" * 60)
        
        try:
            interface = QuantAITerminalInterface()
            interface.start()
        except Exception as e:
            self.logger.error(f"CLI mode failed: {e}")
            print(f"❌ CLI mode failed: {e}")
    
    def run_api_mode(self, port: int):
        """Run enhanced API mode"""
        print(f"🌐 Starting Enhanced API on port {port}...")
        
        try:
            import uvicorn
            from api.portfolio_api import app
            uvicorn.run(app, host=self.settings.api.host, port=port)
        except ImportError as e:
            print(f"❌ Failed to start API mode: {e}")
            print("🔄 Falling back to CLI mode...")
            self.run_cli_mode()
    
    def run_demo_mode(self):
        """Run enhanced demo mode"""
        print("\n🎮 Enhanced Demo Mode")
        print("This mode demonstrates the complete system capabilities.")
        
        try:
            # Demonstrate user journey
            self._demonstrate_user_journey()
            
            # Demonstrate learning system
            self._demonstrate_learning_system()
            
            # Demonstrate security features
            self._demonstrate_security_features()
            
        except Exception as e:
            print(f"❌ Demo mode failed: {e}")
    
    def _demonstrate_user_journey(self):
        """Demonstrate complete user journey"""
        print("\n📋 USER JOURNEY DEMONSTRATION")
        print("=" * 50)
        
        # Step 1: User Registration
        print("1. 👤 User Registration")
        print("   - Secure password hashing (PBKDF2-SHA256)")
        print("   - Risk tolerance profiling")
        print("   - Investment experience tracking")
        
        # Step 2: Portfolio Setup
        print("\n2. 💼 Portfolio Setup")
        print("   - Add positions for 5 focused stocks")
        print("   - Real-time P&L calculation")
        print("   - Risk metrics and alerts")
        
        # Step 3: AI Recommendations
        print("\n3. 🤖 AI Recommendations")
        print("   - Multi-model ensemble predictions")
        print("   - Confidence scoring")
        print("   - Risk-adjusted position sizing")
        
        # Step 4: User Feedback
        print("\n4. 🔄 User Feedback Learning")
        print("   - Accept/reject recommendations")
        print("   - Outcome tracking")
        print("   - Model improvement")
        
        # Step 5: Personalization
        print("\n5. 🎯 Personalization")
        print("   - Behavior pattern analysis")
        print("   - Personalized recommendations")
        print("   - Learning insights")
    
    def _demonstrate_learning_system(self):
        """Demonstrate reinforcement learning system"""
        print("\n🧠 REINFORCEMENT LEARNING DEMONSTRATION")
        print("=" * 50)
        
        print("📊 Learning Components:")
        print("   - User feedback collection")
        print("   - Outcome evaluation")
        print("   - Model retraining")
        print("   - Performance improvement")
        
        print("\n📈 Learning Metrics:")
        print("   - Recommendation accuracy: >65%")
        print("   - User acceptance rate: >40%")
        print("   - Learning improvement: 30 days")
        print("   - Personalization: Individual user patterns")
    
    def _demonstrate_security_features(self):
        """Demonstrate security features"""
        print("\n🛡️ SECURITY FEATURES DEMONSTRATION")
        print("=" * 50)
        
        print("🔐 Authentication:")
        print("   - PBKDF2 password hashing")
        print("   - Session management")
        print("   - Rate limiting")
        print("   - Login attempt tracking")
        
        print("\n🛡️ Authorization:")
        print("   - User role management")
        print("   - Portfolio access control")
        print("   - API endpoint protection")
        
        print("\n📊 Security Monitoring:")
        print("   - Failed login tracking")
        print("   - Session management")
        print("   - Security alerts")
    
    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            # Database status
            db_status = "Connected" if self.db.connection else "Disconnected"
            
            # Security status
            security_summary = self.security_manager.get_security_summary()
            
            # Learning status
            learning_report = self.feedback_trainer.get_model_performance_report()
            
            # System metrics
            system_metrics = {
                'database_status': db_status,
                'active_sessions': security_summary['active_sessions'],
                'security_features': security_summary['security_features'],
                'learning_performance': learning_report,
                'supported_symbols': self.settings.trading.supported_symbols,
                'feature_flags': self.settings.get_feature_flags(),
                'performance_targets': self.settings.get_performance_targets()
            }
            
            return system_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

def main():
    """Main entry point for Enhanced Portfolio Manager"""
    parser = argparse.ArgumentParser(description='Enhanced QuantAI Portfolio Manager')
    parser.add_argument('--mode', choices=['cli', 'api', 'demo', 'status'], 
                       default='cli', help='Run mode')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize enhanced portfolio manager
        manager = EnhancedPortfolioManager()
        
        if args.mode == 'cli':
            manager.run_cli_mode()
        elif args.mode == 'api':
            manager.run_api_mode(args.port)
        elif args.mode == 'demo':
            manager.run_demo_mode()
        elif args.mode == 'status':
            status = manager.get_system_status()
            print("🔧 ENHANCED SYSTEM STATUS")
            print("=" * 40)
            for key, value in status.items():
                print(f"{key}: {value}")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
