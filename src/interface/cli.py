"""
Terminal Interface for QuantAI Portfolio Manager & Trade Suggestion Bot.

This module provides a command-line interface for users to interact with
the portfolio management system, view recommendations, and manage positions.
"""

import getpass
from typing import Optional, Dict
import logging

# Import database and portfolio managers
try:
    from ..database.db_manager import QuantAIDatabase
    from ..portfolio.portfolio_manager import PortfolioManager
    from ..data.data_sources import DataManager
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    logging.warning("Some modules not available, using fallback implementations")

class QuantAITerminalInterface:
    """Terminal-based interface for QuantAI Portfolio Manager"""
    
    def __init__(self):
        # Database configuration (adjust as needed)
        db_config = {
            'host': 'localhost',
            'database': 'quantai_portfolio',
            'user': 'root',
            'password': ''
        }
        
        try:
            self.db = QuantAIDatabase(db_config)
            self.data_manager = DataManager() if MODULES_AVAILABLE else None
            self.portfolio_manager = PortfolioManager(self.db, self.data_manager)
            self.current_user_id = None
            self.logger = logging.getLogger(__name__)
            
            # Initialize database
            self.db.create_tables()
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            print("🔄 Using fallback mode...")
            self._create_fallback_system()
    
    def _create_fallback_system(self):
        """Create fallback system for development"""
        class FallbackDB:
            def create_tables(self): pass
            def create_user(self, *args): return 1
            def authenticate_user(self, *args): return 1
            def get_user_portfolio(self, user_id): return []
            def add_portfolio_position(self, *args): return 1
            def update_portfolio_prices(self, *args): pass
            def save_recommendation(self, *args): return 1
            def record_user_response(self, *args): pass
        
        self.db = FallbackDB()
        self.portfolio_manager = PortfolioManager(self.db)
        self.current_user_id = 1
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the terminal interface"""
        print("🚀 Welcome to QuantAI Portfolio Manager & Trading Bot")
        print("=" * 60)
        print("🎯 Focused on AMZN, META, NVDA, GOOGL, AAPL")
        print("🤖 AI-powered recommendations with risk management")
        print("📊 Portfolio tracking and performance analytics")
        print("=" * 60)
        
        while True:
            if self.current_user_id is None:
                self._show_login_menu()
            else:
                self._show_main_menu()
    
    def _show_login_menu(self):
        """Show login/registration menu"""
        print("\n📋 LOGIN / REGISTRATION")
        print("1. Login")
        print("2. Create New Account")
        print("3. Demo Mode (No Database)")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            self._login()
        elif choice == '2':
            self._register()
        elif choice == '3':
            self._demo_mode()
        elif choice == '4':
            print("👋 Goodbye!")
            exit()
        else:
            print("❌ Invalid option. Please try again.")
    
    def _demo_mode(self):
        """Enter demo mode without database"""
        print("\n🎮 DEMO MODE")
        print("Using fallback system for demonstration...")
        self._create_fallback_system()
        print("✅ Demo mode activated!")
    
    def _login(self):
        """User login"""
        print("\n🔐 USER LOGIN")
        username = input("Username: ").strip()
        password = getpass.getpass("Password: ")
        
        user_id = self.db.authenticate_user(username, password)
        if user_id:
            self.current_user_id = user_id
            print(f"✅ Welcome back, {username}!")
        else:
            print("❌ Invalid credentials. Please try again.")
    
    def _register(self):
        """User registration"""
        print("\n📝 CREATE NEW ACCOUNT")
        username = input("Choose username: ").strip()
        password = getpass.getpass("Choose password: ")
        email = input("Email (optional): ").strip() or None
        
        print("\nRisk Tolerance:")
        print("1. Conservative")
        print("2. Moderate") 
        print("3. Aggressive")
        risk_choice = input("Select (1-3): ").strip()
        risk_map = {'1': 'Conservative', '2': 'Moderate', '3': 'Aggressive'}
        risk_tolerance = risk_map.get(risk_choice, 'Moderate')
        
        print("\nInvestment Experience:")
        print("1. Beginner")
        print("2. Intermediate")
        print("3. Advanced")
        exp_choice = input("Select (1-3): ").strip()
        exp_map = {'1': 'Beginner', '2': 'Intermediate', '3': 'Advanced'}
        experience = exp_map.get(exp_choice, 'Beginner')
        
        try:
            user_id = self.db.create_user(username, password, email, risk_tolerance, experience)
            self.current_user_id = user_id
            print(f"✅ Account created successfully! Welcome, {username}!")
        except Exception as e:
            print(f"❌ Failed to create account: {e}")
    
    def _show_main_menu(self):
        """Show main application menu"""
        print(f"\n📊 QUANTAI PORTFOLIO MANAGER")
        print("1. View Portfolio")
        print("2. Get AI Recommendations") 
        print("3. Add Stock Position")
        print("4. Performance Analytics")
        print("5. Update Prices")
        print("6. System Status")
        print("7. Logout")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            self._view_portfolio()
        elif choice == '2':
            self._get_recommendations()
        elif choice == '3':
            self._add_position()
        elif choice == '4':
            self._show_analytics()
        elif choice == '5':
            self._update_prices()
        elif choice == '6':
            self._show_system_status()
        elif choice == '7':
            self.current_user_id = None
            print("👋 Logged out successfully!")
        else:
            print("❌ Invalid option. Please try again.")
    
    def _view_portfolio(self):
        """Display user portfolio"""
        print("\n📊 YOUR PORTFOLIO")
        print("=" * 80)
        
        summary = self.portfolio_manager.get_user_portfolio_summary(self.current_user_id)
        
        if not summary['positions']:
            print("📭 Your portfolio is empty. Add some positions to get started!")
            print("\n💡 Available stocks: AMZN, META, NVDA, GOOGL, AAPL")
            return
        
        # Portfolio summary
        print(f"💰 Total Value: ${summary['total_value']:,.2f}")
        print(f"📈 Total P&L: ${summary['total_pnl']:,.2f} ({summary['total_return_pct']:.2f}%)")
        print(f"🎯 Risk Level: {summary['risk_metrics'].get('risk_level', 'N/A')}")
        print(f"📊 Diversification Score: {summary['risk_metrics'].get('diversification_score', 0):.2f}")
        print()
        
        # Individual positions
        print(f"{'Symbol':<8} {'Shares':<12} {'Avg Price':<12} "
              f"{'Current':<12} {'Value':<15} {'P&L':<15} {'Return %':<10}")
        print("-" * 90)
        
        for pos in summary['positions']:
            symbol = pos['symbol']
            shares = pos['shares']
            avg_price = pos['avg_purchase_price']
            current = pos['current_price'] or 0
            value = pos['market_value'] or 0
            pnl = pos['unrealized_pnl'] or 0
            return_pct = (pnl / (shares * avg_price) * 100) if shares * avg_price > 0 else 0
            
            pnl_color = "📈" if pnl >= 0 else "📉"
            
            print(f"{symbol:<8} {shares:<12.2f} ${avg_price:<11.2f} "
                  f"${current:<11.2f} ${value:<14.2f} {pnl_color}${pnl:<13.2f} "
                  f"{return_pct:<9.2f}%")
    
    def _get_recommendations(self):
        """Get and display AI recommendations"""
        print("\n🤖 AI TRADING RECOMMENDATIONS")
        print("=" * 50)
        print("⏳ Generating recommendations...")
        
        # Update prices first
        self.portfolio_manager.update_all_portfolios()
        
        recommendations = self.portfolio_manager.generate_recommendations(self.current_user_id)
        
        if not recommendations:
            print("😴 No recommendations at this time. The market conditions don't favor any trades right now.")
            return
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n📋 Recommendation {i}")
            print(f"🏷️  Symbol: {rec['symbol']}")
            print(f"🎯 Action: {rec['type']}")
            print(f"📊 Shares: {rec['shares']:.2f}")
            print(f"💲 Target Price: ${rec['target_price']:.2f}")
            print(f"🎪 Confidence: {rec['confidence']:.1%}")
            print(f"⚠️  Risk Score: {rec['risk_score']:.2f}/1.0")
            print(f"💡 Reasoning: {rec.get('reasoning', 'Technical analysis indicates favorable conditions')}")
            
            # Ask user for decision
            while True:
                decision = input(f"\n🤔 Accept this recommendation? (y/n/skip): ").lower().strip()
                if decision in ['y', 'yes']:
                    self._execute_recommendation(rec, True)
                    break
                elif decision in ['n', 'no']:
                    self._execute_recommendation(rec, False)
                    print("❌ Recommendation declined.")
                    break
                elif decision == 'skip':
                    print("⏭️  Skipped recommendation.")
                    break
                else:
                    print("Please enter 'y' for yes, 'n' for no, or 'skip' to skip.")
    
    def _execute_recommendation(self, rec: Dict, accept: bool):
        """Execute recommendation based on user decision"""
        if accept:
            print(f"\n✅ Executing {rec['type']} order for {rec['symbol']}")
            
            # Get actual execution details
            actual_shares = float(input(f"Shares to trade (recommended: {rec['shares']:.2f}): ") or rec['shares'])
            actual_price = float(input(f"Execution price (target: ${rec['target_price']:.2f}): ") or rec['target_price'])
            
            result = self.portfolio_manager.execute_recommendation(
                rec['recommendation_id'], self.current_user_id, True, actual_shares, actual_price
            )
            
            if result['success']:
                print(f"🎉 {result['message']}")
            else:
                print(f"❌ {result['message']}")
        else:
            # Record rejection
            self.portfolio_manager.execute_recommendation(
                rec['recommendation_id'], self.current_user_id, False
            )
    
    def _add_position(self):
        """Add new stock position"""
        print("\n📈 ADD STOCK POSITION")
        print("Available symbols: AMZN, META, NVDA, GOOGL, AAPL")
        
        symbol = input("Stock symbol: ").upper().strip()
        if symbol not in ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']:
            print("❌ Invalid symbol. Only FAANG+ stocks are supported.")
            return
        
        try:
            shares = float(input("Number of shares: "))
            price = float(input("Purchase price per share: $"))
            
            portfolio_id = self.db.add_portfolio_position(self.current_user_id, symbol, shares, price)
            print(f"✅ Added {shares} shares of {symbol} at ${price:.2f} per share")
            
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"❌ Failed to add position: {e}")
    
    def _show_analytics(self):
        """Display performance analytics"""
        print("\n📊 PERFORMANCE ANALYTICS (Last 30 Days)")
        print("=" * 50)
        
        analytics = self.portfolio_manager.get_performance_analytics(self.current_user_id)
        
        if not analytics:
            print("📭 No analytics data available yet.")
            return
        
        print(f"🎯 Total Recommendations: {analytics.get('total_recommendations', 0)}")
        print(f"✅ Acceptance Rate: {analytics.get('acceptance_rate', 0):.1f}%")
        print(f"🏆 Win Rate: {analytics.get('win_rate', 0):.1f}%")
        print(f"📈 Average Return: {analytics.get('avg_return', 0):.2f}%")
        print(f"💰 Portfolio Value: ${analytics.get('portfolio_value', 0):,.2f}")
        print(f"📊 Portfolio Return: {analytics.get('portfolio_return', 0):.2f}%")
        print(f"🎪 Average Confidence: {analytics.get('avg_confidence', 0):.1%}")
    
    def _update_prices(self):
        """Update current stock prices"""
        print("\n🔄 UPDATING PRICES...")
        try:
            self.portfolio_manager.update_all_portfolios()
            print("✅ Prices updated successfully!")
        except Exception as e:
            print(f"❌ Failed to update prices: {e}")
    
    def _show_system_status(self):
        """Display system status"""
        print("\n🔧 SYSTEM STATUS")
        print("=" * 40)
        
        print(f"👤 Current User ID: {self.current_user_id}")
        print(f"🗄️  Database: {'Connected' if hasattr(self.db, 'connection') else 'Fallback'}")
        print(f"📊 Data Manager: {'Available' if MODULES_AVAILABLE else 'Fallback'}")
        print(f"🎯 Supported Symbols: AMZN, META, NVDA, GOOGL, AAPL")
        print(f"🤖 AI Models: {'Available' if MODULES_AVAILABLE else 'Fallback'}")
        print(f"🛡️  Risk Management: {'Available' if MODULES_AVAILABLE else 'Fallback'}")

def main():
    """Main entry point"""
    interface = QuantAITerminalInterface()
    interface.start()

if __name__ == "__main__":
    main()
