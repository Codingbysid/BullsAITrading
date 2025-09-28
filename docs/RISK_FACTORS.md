# 🎯 QuantAI Risk Factor Analysis Documentation

## 📊 **Comprehensive Risk Factor Integration**

The QuantAI Trading Platform now features **comprehensive risk factor analysis** integrated into the four-model decision engine, providing intelligent, risk-aware trading recommendations.

## 🧠 **Four-Model Decision Engine with Risk Factors**

### **Enhanced Architecture**
```
┌─────────────────────────────────────────────────────────────────┐
│                Four-Model Decision Engine                       │
│                    with Risk Factor Analysis                    │
├─────────────────────────────────────────────────────────────────┤
│  📊 Data Collection Layer                                      │
│  ├── Real-time Market Data (Price, Volume, Technical Indicators)│
│  ├── News & Sentiment Data (News API, Gemini AI, Social Media) │
│  ├── Fundamental Data (Financial Ratios, Earnings, Growth)     │
│  └── Historical Patterns (5+ Years of Market Data)            │
├─────────────────────────────────────────────────────────────────┤
│  🧠 Four-Model Processing Layer                               │
│  ├── Sentiment Analysis Model (25% input weight)              │
│  │   ├── News API Integration with risk sentiment analysis    │
│  │   ├── Gemini AI Processing with volatility awareness       │
│  │   ├── Social Media Monitoring with correlation risk        │
│  │   └── Multi-Source Aggregation with risk weighting         │
│  ├── Quantitative Risk Model (25% input weight)               │
│  │   ├── Sharpe Ratio, MAR Ratio, Alpha, Beta analysis      │
│  │   ├── Sortino Ratio, Calmar Ratio, Information Ratio      │
│  │   ├── Volatility Analysis with real-time calculation      │
│  │   ├── Drawdown Protection with maximum drawdown analysis   │
│  │   └── VaR Calculations with 95% and 99% confidence        │
│  ├── ML Ensemble Model (35% input weight)                     │
│  │   ├── Random Forest with risk-aware feature selection     │
│  │   ├── XGBoost with volatility-based boosting              │
│  │   ├── LSTM Networks with risk-adjusted sequences          │
│  │   └── Ensemble Voting with risk-weighted consensus        │
│  └── RL Decider Agent (Final decision maker)                  │
│      ├── Deep Q-Network with risk-adjusted Q-values          │
│      ├── Experience Replay with risk factor learning          │
│      ├── Epsilon-Greedy with risk-aware exploration          │
│      └── Position Sizing with Kelly Criterion optimization   │
├─────────────────────────────────────────────────────────────────┤
│  🎯 Risk Factor Analysis Layer                                │
│  ├── Volatility Analysis (Real-time volatility calculation)  │
│  ├── Maximum Drawdown Analysis (Historical and current)      │
│  ├── VaR Calculations (95% and 99% confidence levels)        │
│  ├── Correlation Risk Assessment (Portfolio correlation)      │
│  ├── Risk Score Calculation (0-100% risk assessment)          │
│  ├── Sharpe Ratio Analysis (Risk-adjusted returns)          │
│  └── Position Sizing Optimization (Kelly Criterion)          │
├─────────────────────────────────────────────────────────────────┤
│  🔧 Risk Adjustment Layer                                     │
│  ├── Volatility Adjustments (High vol = reduced signals)     │
│  ├── Drawdown Adjustments (High drawdown = reduced size)     │
│  ├── Correlation Adjustments (High correlation = reduced size)│
│  ├── VaR Adjustments (High VaR = reduced exposure)           │
│  ├── Risk Score Thresholds (Risk-based BUY/SELL/HOLD)        │
│  └── Position Size Limits (Risk-adjusted position sizing)   │
├─────────────────────────────────────────────────────────────────┤
│  🎯 Final Decision Output                                      │
│  ├── Action: BUY/SELL/HOLD with risk-adjusted reasoning      │
│  ├── Confidence: Risk-adjusted confidence scoring             │
│  ├── Position Size: Kelly Criterion with risk adjustments     │
│  ├── Risk Metrics: Comprehensive risk factor analysis         │
│  └── Reasoning: Detailed risk-aware decision explanation      │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 **Risk Factor Components**

### **1. Volatility Analysis**
- **Real-time Volatility**: Current market volatility calculation
- **Historical Volatility**: Long-term volatility trends
- **Volatility Thresholds**: High/medium/low volatility classification
- **Volatility Adjustments**: High volatility reduces signal strength

### **2. Maximum Drawdown Analysis**
- **Historical Drawdown**: Worst historical loss analysis
- **Current Drawdown**: Real-time drawdown monitoring
- **Drawdown Protection**: Maximum drawdown limits
- **Drawdown Adjustments**: High drawdown reduces position size

### **3. Value at Risk (VaR) Calculations**
- **VaR 95%**: 95% confidence level risk assessment
- **VaR 99%**: 99% confidence level risk assessment
- **VaR Adjustments**: High VaR reduces exposure
- **VaR Monitoring**: Real-time VaR tracking

### **4. Correlation Risk Assessment**
- **Portfolio Correlation**: Cross-asset correlation analysis
- **Correlation Limits**: Maximum correlation thresholds
- **Correlation Adjustments**: High correlation reduces position size
- **Diversification Analysis**: Portfolio diversification assessment

### **5. Risk Score Calculation**
- **Risk Score (0-100%)**: Comprehensive risk assessment
- **Risk Score Thresholds**: Risk-based decision thresholds
- **Risk Score Adjustments**: High risk score reduces signals
- **Risk Score Monitoring**: Real-time risk score tracking

### **6. Sharpe Ratio Analysis**
- **Sharpe Ratio**: Risk-adjusted return calculation
- **Sharpe Adjustments**: Low Sharpe reduces position size
- **Sharpe Monitoring**: Real-time Sharpe ratio tracking
- **Sharpe Optimization**: Sharpe ratio-based decisions

## 🔧 **Risk Adjustment Mechanisms**

### **Volatility Adjustments**
```python
# High volatility reduces signal strength
if volatility > 0.05:  # Very high volatility
    adjustment = 0.5
elif volatility > 0.03:  # High volatility
    adjustment = 0.7
elif volatility < 0.01:  # Very low volatility
    adjustment = 1.2
else:  # Normal volatility
    adjustment = 1.0
```

### **Drawdown Adjustments**
```python
# High drawdown reduces position size
drawdown_adjustment = 1.0 - (max_drawdown * 2)
```

### **Correlation Adjustments**
```python
# High correlation reduces position size
correlation_adjustment = 1.0 - (correlation_risk * 0.5)
```

### **VaR Adjustments**
```python
# High VaR reduces exposure
var_adjustment = 1.0 - (abs(var_95) * 5)
```

### **Risk Score Thresholds**
```python
# Risk-based BUY/SELL/HOLD decisions
if final_signal > 0.3 and risk_score < 0.7:
    action = "BUY"
elif final_signal < -0.3 and risk_score < 0.7:
    action = "SELL"
else:
    action = "HOLD"
```

## 📊 **Position Sizing with Kelly Criterion**

### **Kelly Criterion Implementation**
```python
# Kelly Criterion position sizing
win_rate = 0.6  # Assumed win rate
avg_win = 0.1   # Average win
avg_loss = 0.05  # Average loss

kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
```

### **Risk-Adjusted Position Sizing**
```python
# Adjust for risk factors
risk_adjustment = 1.0 - (risk_score * 0.5)
vol_adjustment = 1.0 - (volatility * 10)

# Final position size
position_size = kelly_fraction * risk_adjustment * vol_adjustment * abs(signal)
position_size = max(0.01, min(0.2, position_size))  # Clamp between 1% and 20%
```

## 🎯 **Risk Configuration**

### **Risk Factor Settings**
```json
{
  "risk_tolerance": 0.05,
  "risk_factors": {
    "portfolio_risk": 0.05,
    "max_drawdown": 0.15,
    "volatility_threshold": 0.03,
    "correlation_limit": 0.7,
    "var_confidence": 0.95,
    "kelly_criterion": true,
    "position_sizing": "kelly"
  },
  "trading_settings": {
    "max_position_size": 0.3,
    "max_portfolio_risk": 0.15,
    "stop_loss": 0.1,
    "take_profit": 0.2
  }
}
```

### **Risk Adjustment Presets**

#### **Conservative (Low Risk)**
- Risk Tolerance: 2%
- Max Drawdown: 8%
- Volatility Threshold: 2%
- Max Position Size: 15%
- Stop Loss: 5%
- Take Profit: 15%

#### **Moderate (Medium Risk)**
- Risk Tolerance: 5%
- Max Drawdown: 15%
- Volatility Threshold: 3%
- Max Position Size: 30%
- Stop Loss: 10%
- Take Profit: 20%

#### **Aggressive (High Risk)**
- Risk Tolerance: 10%
- Max Drawdown: 25%
- Volatility Threshold: 5%
- Max Position Size: 50%
- Stop Loss: 15%
- Take Profit: 30%

## 🚀 **Usage Examples**

### **Run Enhanced Risk Analysis**
```bash
# Run four-model recommendations with risk factors
python run_with_risk_factors.py
```

### **Adjust Risk Factors**
```bash
# Interactive risk factor adjustment
python adjust_risk_factors.py
```

### **Run Portfolio Manager with Risk**
```bash
# Set up local environment
python run_portfolio.py --setup

# Run portfolio manager with AI recommendations
python run_portfolio.py --run
```

## 📊 **Example Output with Risk Factors**

```
📊 AAPL Analysis:
🎯 Action: HOLD
📈 Confidence: 70.0%
💭 Reasoning: RL Decision: Sentiment=1.00(80.0%), Quant=0.00(60.0%), 
              ML=0.00(50.0%), Final=0.15, Risk=0.0%, Vol=1.7%, 
              Corr=30.0%, PosSize=2.7%

⚠️  RISK FACTOR ANALYSIS:
   📊 Volatility: 1.73%
   📉 Max Drawdown: 12.30%
   💰 VaR (95%): -3.08%
   🔗 Correlation Risk: 30.00%
   ⚖️  Risk Score: 19.09%
   📈 Sharpe Ratio: -0.11
   💼 Position Size: 2.7%

🔧 RISK ADJUSTMENTS:
   Risk Adjustment: 0.69
   Volatility Adjustment: 1.00
   Correlation Adjustment: 0.85
   Cash Adjustment: 1.25
```

## 🎯 **Key Benefits**

✅ **Risk-Aware Decisions**: All recommendations consider comprehensive risk factors
✅ **Volatility Protection**: High volatility reduces signal strength and position size
✅ **Drawdown Protection**: Maximum drawdown limits prevent excessive losses
✅ **VaR Monitoring**: Value at Risk calculations provide risk quantification
✅ **Correlation Analysis**: Portfolio correlation risk assessment
✅ **Position Sizing**: Kelly Criterion with risk adjustments
✅ **Real-time Risk Monitoring**: Continuous risk factor analysis
✅ **Risk Score Thresholds**: Risk-based BUY/SELL/HOLD decisions

## 🔧 **Risk Factor Tools**

### **adjust_risk_factors.py**
Interactive tool for adjusting risk factors with presets:
- Conservative (Low Risk)
- Moderate (Medium Risk)
- Aggressive (High Risk)
- Custom Settings
- Reset to Default

### **run_with_risk_factors.py**
Demonstrates the four-model decision engine with comprehensive risk factor analysis.

### **Configuration Files**
- `config/local/portfolio_config.json`: Risk factor configuration
- Real-time risk factor updates
- Risk adjustment presets

The QuantAI Trading Platform now provides **intelligent, risk-aware trading recommendations** with comprehensive risk factor analysis, ensuring that all decisions consider volatility, drawdown, VaR, correlation risk, and position sizing optimization!
