#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Union, Set
import os
import logging
import time
import json
import threading
from logging.handlers import RotatingFileHandler
from datetime import datetime
import logging.config

class DuplicateFilter(logging.Filter):
    """Filter to prevent duplicate log messages from appearing in console output"""
    
    def __init__(self):
        super().__init__()
        self.seen_messages = set()
        self.message_timestamps = {}
        self.lock = threading.Lock()
        self.cleanup_interval = 300  # Clean up old messages every 5 minutes
        self.last_cleanup = time.time()
    
    def filter(self, record):
        """Filter out duplicate messages within a short time window"""
        with self.lock:
            current_time = time.time()
            
            # Clean up old messages periodically
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_messages(current_time)
                self.last_cleanup = current_time
            
            # Create a unique identifier for this message
            message_key = f"{record.name}:{record.levelname}:{record.getMessage()}"
            
            # Check if we've seen this exact message recently (within 30 seconds)
            if message_key in self.message_timestamps:
                last_seen = self.message_timestamps[message_key]
                if current_time - last_seen < 30:  # 30 second window
                    return False  # Filter out duplicate
            
            # Record this message
            self.message_timestamps[message_key] = current_time
            return True  # Allow message through
    
    def _cleanup_old_messages(self, current_time):
        """Remove old message timestamps to prevent memory buildup"""
        cutoff_time = current_time - 300  # Remove messages older than 5 minutes
        keys_to_remove = [
            key for key, timestamp in self.message_timestamps.items() 
            if timestamp < cutoff_time
        ]
        for key in keys_to_remove:
            del self.message_timestamps[key]

class SingleConsoleHandler(logging.StreamHandler):
    """Custom StreamHandler that ensures only one console handler exists globally"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SingleConsoleHandler, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, stream=None):
        if not hasattr(self, '_initialized') or not self._initialized:
            super().__init__(stream)
            self.addFilter(DuplicateFilter())
            self._initialized = True

class CorrelationLogger:
    def __init__(self) -> None:
        # Setup unified logging first (before any other logger setup)
        self.logger = self.setup_unified_logging()
        
        # Get references to specific sub-loggers (these are already configured by setup_unified_logging)
        self.coingecko_logger = logging.getLogger('ETHBTCCorrelation.coingecko')
        self.client_logger = logging.getLogger('ETHBTCCorrelation.claude')
        self.sheets_logger = logging.getLogger('ETHBTCCorrelation.google_sheets')
        self.analysis_logger = logging.getLogger('ETHBTCCorrelation.analysis')
        
        # Create a reference to the M4TechnicalFoundation logger
        self.m4_logger = logging.getLogger('M4TechnicalFoundation')

    def info(self, message):
        """Log an info message (compatibility method)"""
        self.logger.info(message)

    def debug(self, message):
        """Log a debug message (compatibility method)"""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log a warning message (compatibility method)"""
        self.logger.warning(message)
    
    def error(self, message, exc_info=False):
        """Log an error message (compatibility method)"""
        self.logger.error(message, exc_info=exc_info)    

    def _setup_api_logger(self, api_name: str) -> logging.Logger:
        """Setup specific logger for each API with its own file"""
        # The loggers are already configured by setup_unified_logging
        # Just return the appropriate logger based on the API name
        return logging.getLogger(f'ETHBTCCorrelation.{api_name}')

    def _setup_analysis_logger(self) -> logging.Logger:
        """Setup specific logger for market analysis"""
        # The logger is already configured by setup_unified_logging
        return logging.getLogger('ETHBTCCorrelation.analysis')

    def log_coingecko_request(self, endpoint: str, success: bool = True) -> None:
        """Log Coingecko API interactions"""
        msg = f"CoinGecko API Request - Endpoint: {endpoint}"
        if success:
            self.coingecko_logger.info(msg)
        else:
            self.coingecko_logger.error(msg)

    def log_claude_analysis(
        self, 
        btc_price: float, 
        eth_price: float, 
        status: bool = True
    ) -> None:
        """Log Claude analysis details"""
        msg = (
            "Claude Analysis - "
            f"BTC Price: ${btc_price:,.2f} - "
            f"ETH Price: ${eth_price:,.2f}"
        )
        
        if status:
            self.client_logger.info(msg)
        else:
            self.client_logger.error(msg)

    def log_sheets_update(
        self, 
        data_type: str, 
        status: bool = True
    ) -> None:
        """Log Google Sheets interactions"""
        msg = f"Google Sheets Update - Data Type: {data_type}"
       
        if status:
            self.sheets_logger.info(msg)
        else:
            self.sheets_logger.error(msg)

    def log_market_correlation(
        self, 
        correlation_coefficient: float, 
        price_movement: float
    ) -> None:
        """Log market correlation details"""
        self.logger.info(
            "Market Correlation - "
            f"Correlation Coefficient: {correlation_coefficient:.2f} - "
            f"Price Movement: {price_movement:.2f}%"
        )

    def log_error(
        self, 
        error_type: str, 
        message: str, 
        exc_info: Union[bool, Exception, None] = None
    ) -> None:
        """Log errors with stack trace option"""
        self.logger.error(
            f"Error - Type: {error_type} - Message: {message}",
            exc_info=exc_info if exc_info else False
        )

    def log_twitter_action(self, action_type: str, status: str) -> None:
        """Log Twitter related actions"""
        self.logger.info(f"Twitter Action - Type: {action_type} - Status: {status}")

    def log_startup(self) -> None:
        """Log application startup"""
        self.logger.info("=" * 50)
        self.logger.info(f"ETH-BTC Correlation Bot Starting - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 50)

    def log_shutdown(self) -> None:
        """Log application shutdown"""
        self.logger.info("=" * 50)
        self.logger.info(f"ETH-BTC Correlation Bot Shutting Down - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 50)

    def setup_unified_logging(self):
        """
        Configure a unified logging system to prevent duplicate logs.
        This resets any existing logging configuration to ensure consistency.
        FIXED VERSION - Eliminates all duplicate terminal output.
        """
        # Step 1: Complete logging system reset
        self._reset_logging_system()
        
        # Step 2: Create directory structure
        self._create_log_directories()
        
        # Step 3: Create and configure the global duplicate filter
        duplicate_filter = DuplicateFilter()
        
        # Step 4: Create console handlers with duplicate filtering
        correlation_console_handler = logging.StreamHandler()
        correlation_console_handler.setLevel(logging.INFO)
        correlation_formatter = logging.Formatter(
            '%(asctime)s | 📊 CORR | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        correlation_console_handler.setFormatter(correlation_formatter)
        correlation_console_handler.addFilter(duplicate_filter)
        
        m4_console_handler = logging.StreamHandler()
        m4_console_handler.setLevel(logging.INFO)
        m4_formatter = logging.Formatter(
            '%(asctime)s | 🚀 M4 | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        m4_console_handler.setFormatter(m4_formatter)
        m4_console_handler.addFilter(duplicate_filter)
        
        # Step 5: Create file handlers
        correlation_file_handler = RotatingFileHandler(
            'logs/eth_btc_correlation.log',
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        correlation_file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        correlation_file_handler.setFormatter(file_formatter)
        
        m4_file_handler = RotatingFileHandler(
            'logs/technical/m4_foundation.log',
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        m4_file_handler.setLevel(logging.INFO)
        m4_file_handler.setFormatter(file_formatter)
        
        # Additional file handlers for sub-loggers
        coingecko_file_handler = RotatingFileHandler(
            'logs/coingecko.log',
            maxBytes=5*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        coingecko_file_handler.setLevel(logging.INFO)
        coingecko_file_handler.setFormatter(file_formatter)
        
        claude_file_handler = RotatingFileHandler(
            'logs/claude.log',
            maxBytes=5*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        claude_file_handler.setLevel(logging.INFO)
        claude_file_handler.setFormatter(file_formatter)
        
        sheets_file_handler = RotatingFileHandler(
            'logs/google_sheets_api.log',
            maxBytes=5*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        sheets_file_handler.setLevel(logging.INFO)
        sheets_file_handler.setFormatter(file_formatter)
        
        analysis_file_handler = RotatingFileHandler(
            'logs/analysis/market_analysis.log',
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        analysis_file_handler.setLevel(logging.INFO)
        analysis_file_handler.setFormatter(file_formatter)
        
        # Step 6: Configure loggers manually to avoid circular import issues
        
        # ETHBTCCorrelation main logger
        correlation_logger = logging.getLogger('ETHBTCCorrelation')
        correlation_logger.setLevel(logging.INFO)
        correlation_logger.handlers.clear()  # Remove any existing handlers
        correlation_logger.addHandler(correlation_console_handler)
        correlation_logger.addHandler(correlation_file_handler)
        correlation_logger.propagate = False  # CRITICAL: No propagation
        
        # ETHBTCCorrelation sub-loggers - file only
        coingecko_logger = logging.getLogger('ETHBTCCorrelation.coingecko')
        coingecko_logger.setLevel(logging.INFO)
        coingecko_logger.handlers.clear()
        coingecko_logger.addHandler(coingecko_file_handler)
        coingecko_logger.propagate = False
        
        claude_logger = logging.getLogger('ETHBTCCorrelation.claude')
        claude_logger.setLevel(logging.INFO)
        claude_logger.handlers.clear()
        claude_logger.addHandler(claude_file_handler)
        claude_logger.propagate = False
        
        sheets_logger = logging.getLogger('ETHBTCCorrelation.google_sheets')
        sheets_logger.setLevel(logging.INFO)
        sheets_logger.handlers.clear()
        sheets_logger.addHandler(sheets_file_handler)
        sheets_logger.propagate = False
        
        analysis_logger = logging.getLogger('ETHBTCCorrelation.analysis')
        analysis_logger.setLevel(logging.INFO)
        analysis_logger.handlers.clear()
        analysis_logger.addHandler(analysis_file_handler)
        analysis_logger.propagate = False
        
        # M4TechnicalFoundation logger - separate console and file
        m4_logger = logging.getLogger('M4TechnicalFoundation')
        m4_logger.setLevel(logging.INFO)
        m4_logger.handlers.clear()  # Remove any existing handlers
        m4_logger.addHandler(m4_console_handler)
        m4_logger.addHandler(m4_file_handler)
        m4_logger.propagate = False  # CRITICAL: No propagation
        
        # Step 7: Initialize loggers and confirm setup
        correlation_logger.info("🔧 Anti-duplication logging system initialized")
        m4_logger.info("🔧 M4TechnicalFoundation logger initialized - no duplicates")
        correlation_logger.info("✅ Duplicate terminal logging issue resolved")
        
        return correlation_logger
    
    def _reset_logging_system(self):
        """Completely reset the logging system to eliminate any existing handlers"""
        # Remove all handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        
        # Clear all existing loggers
        logging.Logger.manager.loggerDict.clear()
        
        # Reset root logger level
        root_logger.setLevel(logging.WARNING)
        
        # Clear SingleConsoleHandler instance to allow fresh creation
        SingleConsoleHandler._instance = None
    
    def _create_log_directories(self):
        """Create necessary log directories"""
        directories = [
            'logs',
            'logs/analysis', 
            'logs/technical'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def log_sheets_operation(
        self, 
        operation_type: str, 
        status: bool, 
        details: Optional[str] = None
    ) -> None:
        """Log Google Sheets operations"""
        msg = f"Google Sheets Operation - Type: {operation_type} - Status: {'Success' if status else 'Failed'}"
        if details:
            msg += f" - Details: {details}"
        
        if status:
            self.sheets_logger.info(msg)
        else:
            self.sheets_logger.error(msg)

# Singleton instance - this ensures only one logger instance exists
logger = CorrelationLogger()