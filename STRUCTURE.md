# Karma Agent Project Structure

This document outlines the directory structure of the Karma Agent, an advanced cryptocurrency market analysis and social influence platform.

```
karma/
├── .env                        # Environment variables (API keys, credentials)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation and overview
├── STRUCTURE.md                # This file - project structure documentation
├── __init__.py                 # Makes the directory a Python package
├── requirements.txt            # Python dependencies
├── architecture.txt            # System architecture description
├── tech_test.py                # Technical testing script
│
├── data/                       # Data storage directory
│   ├── crypto_history.db       # SQLite database for crypto historical data
│   └── backup/                 # Database backups
│       └── crypto_history.db.bak
│
├── logs/                       # Logging directory with organized subsystems
│   ├── thc.log                 # General application logs
│   ├── claude.log              # Claude AI integration logs
│   ├── claude_api.log          # Claude API interaction logs
│   ├── coingecko.log           # CoinGecko data logs
│   ├── coingecko_api.log       # CoinGecko API interaction logs
│   ├── google_sheets_api.log   # Google Sheets API interaction logs
│   ├── ETHBTCCorrelation.log   # Ethereum-Bitcoin correlation logs
│   ├── eth_btc_correlation.log # ETH-BTC correlation analysis logs
│   ├── trading_system_*.log    # Daily trading system logs
│   ├── wealth_system_*.log     # Wealth generation system logs
│   │
│   ├── analysis/               # Analysis-specific logs
│   │   └── market_analysis.log
│   │
│   ├── billion_dollar_system/  # Wealth generation subsystem logs
│   │   └── wealth_generation_*.log
│   │
│   └── technical/              # Technical analysis logs
│       └── m4_foundation.log
│
└── src/                        # Source code directory
    ├── __init__.py             # Package initialization
    │
    ├── bot.py                  # Main bot implementation and orchestration
    ├── config.py               # Configuration settings
    ├── conf.py                 # Additional configuration
    ├── database.py             # Database interaction module
    ├── mood_config.py          # Market mood/sentiment configuration
    │
    ├── coingecko_handler.py    # CoinGecko API handler
    ├── gecko.py                # Additional CoinGecko utilities
    │
    ├── llm_provider.py         # Multi-LLM integration (Claude, GPT, Mistral, Groq)
    ├── content_analyzer.py     # Content analysis for social media
    ├── reply_handler.py        # Social media reply intelligence system
    ├── timeline_scraper.py     # Social media timeline monitoring
    ├── meme_phrases.py         # Meme phrases for bot responses
    │
    ├── prediction_engine.py    # Machine learning prediction engine
    ├── technical_core.py       # Core technical analysis functions
    ├── technical_foundation.py # Foundation layer for technical analysis
    ├── technical_indicators.py # Technical indicator calculations
    ├── technical_signals.py    # Trading signal generation
    ├── technical_calculations.py # Advanced technical calculations
    ├── technical_portfolio.py  # Portfolio analysis and optimization
    ├── technical_integration.py # Integration of technical systems
    ├── technical_system.py     # Complete technical analysis system
    │
    ├── calculations.py         # General mathematical calculations
    ├── integration.py          # System integration utilities
    ├── data_aggregation_system.py # Data aggregation from multiple sources
    ├── data_validation_layer.py # Data validation and quality assurance
    │
    ├── numba_thread_manager.py # Thread management with Numba optimization
    ├── cpu.py                  # CPU optimization utilities
    ├── thread.py               # Threading utilities
    │
    ├── endpoint.py             # API endpoint management
    ├── network_test.py         # Network connectivity testing
    ├── datetime_utils.py       # Date and time utilities
    │
    ├── fou.py                  # Foundation utilities
    ├── found.py                # Foundation components
    ├── lizard.py               # Additional utilities
    ├── tech_int.py             # Technical integration helpers
    │
    ├── test_browser.py         # Browser automation testing
    │
    └── utils/                  # Utility modules
        ├── __init__.py         # Package initialization
        ├── logger.py           # Logging utilities and configuration
        ├── browser.py          # Web browser automation utilities
        └── sheets_handler.py   # Google Sheets API integration
```

## Key Components & Subsystems

### 🎯 Core Bot System
- **bot.py**: Main orchestrator for the Karma Agent
- **config.py / conf.py**: Environment configuration and settings management
- **database.py**: SQLite database operations for historical data persistence

### 📊 Data Acquisition & Processing
- **coingecko_handler.py / gecko.py**: CoinGecko API integration for 1000+ cryptocurrencies
- **data_aggregation_system.py**: Multi-source data aggregation and normalization
- **data_validation_layer.py**: Data quality assurance and validation
- **Database**: SQLite for historical price data, technical indicators, and analysis results

### 🤖 AI & Social Intelligence
- **llm_provider.py**: Multi-LLM integration (Anthropic Claude, OpenAI GPT, Mistral AI, Groq)
- **content_analyzer.py**: Social media content analysis and sentiment extraction
- **reply_handler.py**: Intelligent reply generation with context awareness
- **timeline_scraper.py**: Social media timeline monitoring and trend detection
- **meme_phrases.py**: Cultural crypto knowledge for authentic engagement

### 📈 Technical Analysis Engine
- **technical_core.py**: Core technical analysis functions
- **technical_foundation.py**: Foundation layer with base indicator calculations
- **technical_indicators.py**: RSI, MACD, Bollinger Bands, VWAP, and custom indicators
- **technical_signals.py**: Trading signal generation from technical indicators
- **technical_calculations.py**: Advanced calculations with Numba optimization
- **technical_portfolio.py**: Portfolio-level analysis and cross-asset correlations
- **technical_system.py**: Complete technical analysis system integration

### 🧠 Machine Learning & Predictions
- **prediction_engine.py**: ML model ensemble (LSTM, ARIMA, Random Forest, Gradient Boosting)
- **mood_config.py**: Market psychology and sentiment classification system
- **calculations.py**: Statistical and mathematical calculations

### ⚡ Performance Optimization
- **numba_thread_manager.py**: Thread management optimized for M4 MacBook
- **cpu.py**: CPU-specific optimizations using Polars and Numba
- **thread.py**: Thread-safe operations and concurrency management

### 🔧 Infrastructure & Utilities
- **endpoint.py**: API endpoint management and routing
- **network_test.py**: Network connectivity and API health checks
- **datetime_utils.py**: Timezone handling and timestamp utilities
- **utils/logger.py**: Comprehensive logging system with file rotation
- **utils/browser.py**: Selenium-based browser automation for web scraping
- **utils/sheets_handler.py**: Google Sheets API for data export and analysis

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │  CoinGecko    │  │ Social Media  │  │  Database     │    │
│  │  API          │  │  Timelines    │  │  Historical   │    │
│  └───────────────┘  └───────────────┘  └───────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              DATA AGGREGATION & VALIDATION                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ data_aggregation_system.py  +  data_validation_layer.py│ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              TECHNICAL ANALYSIS ENGINE                      │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ Technical      │  │ Prediction     │  │ Mood          │  │
│  │ System         │  │ Engine         │  │ Analysis      │  │
│  └────────────────┘  └────────────────┘  └───────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              AI INTELLIGENCE LAYER                          │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ LLM Provider   │  │ Content        │  │ Reply         │  │
│  │ (Multi-model)  │  │ Analyzer       │  │ Handler       │  │
│  └────────────────┘  └────────────────┘  └───────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              BOT ORCHESTRATION & OUTPUT                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              bot.py (Main Controller)                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Performance Features

### M4 MacBook Optimization
- **Polars DataFrames**: Lightning-fast data processing
- **Numba JIT Compilation**: Optimized numerical computations
- **Thread-Safe Architecture**: Optimal multi-core utilization
- **Memory Management**: Efficient resource allocation

### Reliability & Scalability
- **Circuit Breakers**: API failure protection
- **Rate Limiting**: Intelligent request management
- **Error Recovery**: Comprehensive fallback mechanisms
- **Logging System**: Multi-level logging with automatic rotation

## Development Workflow

### Setup Instructions
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
# - CoinGecko API key
# - Claude/OpenAI/Mistral API keys
# - Social media credentials
# - Google Sheets credentials

# Initialize database
python src/database.py

# Run the bot
python src/bot.py
```

### Testing
```bash
# Test browser automation
python src/test_browser.py

# Test network connectivity
python src/network_test.py

# Run technical tests
python tech_test.py
```

### Logging
All logs are organized in the `logs/` directory:
- **System logs**: General application activity
- **API logs**: External API interactions and rate limiting
- **Analysis logs**: Technical analysis and predictions
- **Trading logs**: Trading system operations (if enabled)

## API Integrations

### Current Active
- **CoinGecko**: Primary data source for 1000+ cryptocurrencies
- **Claude AI**: Advanced language model for content generation
- **Google Sheets**: Data export and visualization

### Ready to Deploy
- **CoinMarketCap**: Enhanced market data coverage
- **OpenAI GPT**: Additional LLM provider
- **Mistral AI**: Open-source LLM alternative
- **Groq**: Ultra-fast inference for real-time analysis

## Security Considerations

- **Environment Variables**: All sensitive data in `.env` (never commit)
- **API Key Rotation**: Support for multiple API providers
- **Rate Limiting**: Intelligent request throttling
- **Data Validation**: Multi-layer input verification
- **Error Handling**: Comprehensive exception management

## Future Expansion

The modular architecture supports easy addition of:
- New data sources (CoinMarketCap, Moralis, Alchemy)
- Additional LLM providers
- New social media platforms
- Advanced ML models
- Custom technical indicators
- Portfolio management tools

---

**Note**: This structure is optimized for institutional-grade cryptocurrency analysis with AI-powered social engagement capabilities.
