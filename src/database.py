import sqlite3
import threading
import traceback
from datetime import datetime, timedelta, timezone
import json
from typing import Dict, List, Optional, Union, Any, Tuple, Optional, cast, Callable
from dataclasses import asdict
import os
from utils.logger import logger
from data_aggregation_system import create_enhanced_data_system
import sqlite3
import time
import uuid
from enum import Enum
from functools import wraps


def serialize_datetime_objects(obj):
    """
    Recursively convert datetime objects to ISO format strings for JSON serialization
    
    Args:
        obj: Object that may contain datetime objects (dict, list, datetime, or other)
        
    Returns:
        Object with datetime objects converted to ISO strings
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_datetime_objects(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime_objects(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_datetime_objects(item) for item in obj)
    else:
        return obj

def enterprise_operation(table_name: str, record_key_func: Optional[Callable] = None):
        """
        Decorator to wrap database operations with enterprise features
        
        Args:
            table_name: Name of the table being operated on
            record_key_func: Function to extract record key from arguments
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                # Only apply enterprise features if enterprise mode is enabled
                if not getattr(self, 'enterprise_mode', False):
                    return func(self, *args, **kwargs)
                
                # Initialize operation manager if not already done
                if not hasattr(self, 'operation_manager') or self.operation_manager is None:
                    self.operation_manager = DatabaseOperationManager(self)
                
                # Extract record key for audit trail
                if record_key_func:
                    try:
                        record_key = record_key_func(*args, **kwargs)
                    except:
                        record_key = str(args[0]) if args else "unknown"
                else:
                    record_key = str(args[0]) if args else "unknown"
                
                # Execute with enterprise features
                return self.operation_manager.execute_with_audit(
                    func, table_name, record_key, self, *args, **kwargs
                )
            
            return wrapper
        return decorator  

def wrap_enterprise_methods():
    """Wrap existing database methods with enterprise operation management"""
    
    # Define record key extraction functions
    def extract_post_id(*args, **kwargs):
        return args[0] if args else kwargs.get('post_id', 'unknown')
    
    def extract_token(*args, **kwargs):
        return args[0] if args else kwargs.get('token', 'unknown')
    
    def extract_chain(*args, **kwargs):
        return args[0] if args else kwargs.get('chain', 'unknown')
    
    # Apply enterprise wrapper to key methods
    CryptoDatabase.store_content_analysis = enterprise_operation(
        'content_analysis', extract_post_id
    )(CryptoDatabase.store_content_analysis)
    
    CryptoDatabase.store_reply = enterprise_operation(
        'replied_posts', extract_post_id  
    )(CryptoDatabase.store_reply)
    
    CryptoDatabase.mark_post_as_replied = enterprise_operation(
        'replied_posts', extract_post_id
    )(CryptoDatabase.mark_post_as_replied)
    
    CryptoDatabase.store_reply_restriction = enterprise_operation(
        'reply_restrictions', extract_post_id
    )(CryptoDatabase.store_reply_restriction)
    
    CryptoDatabase.store_market_data = enterprise_operation(
        'market_data', extract_chain
    )(CryptoDatabase.store_market_data)
    
    CryptoDatabase.store_posted_content = enterprise_operation(
        'posted_content', lambda *args, **kwargs: args[0][:50] if args else 'unknown'
    )(CryptoDatabase.store_posted_content)
    
    logger.logger.info("✅ Enterprise operation management enabled for core database methods")

class DatabaseManager:
    """
    Enterprise Database Manager - Singleton Pattern Implementation
    
    Ensures only one CryptoDatabase instance exists across the entire application.
    Follows enterprise patterns with comprehensive logging, error handling,
    and thread-safe operations.
    
    Key Features:
    - Thread-safe singleton implementation
    - Enterprise audit logging
    - Health monitoring integration
    - Graceful error handling with fallback
    - Configuration management integration
    """
    
    _instance = None
    _lock = threading.Lock()
    _database_instance = None
    _initialization_status = {
        'initialized': False,
        'initialization_time': None,
        'initialization_errors': [],
        'instance_count': 0,
        'last_access_time': None
    }
    
    def __new__(cls, *args, **kwargs):
        """Thread-safe singleton implementation"""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the DatabaseManager (only runs once due to singleton)"""
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._creation_time = datetime.now()
        self.enhanced_data_system = create_enhanced_data_system(self)
        
        logger.logger.info("🏢 Enterprise DatabaseManager singleton initializing...")
        logger.logger.debug(f"DatabaseManager created at: {self._creation_time.isoformat()}")
    
    def get_database(self, db_path: Optional[str] = None, enterprise_mode: bool = True, 
                    force_reinit: bool = False) -> 'CryptoDatabase':
        """
        Get the singleton database instance
        
        Args:
            db_path: Database path (optional, uses config default if None)
            enterprise_mode: Enable enterprise features (default: True)
            force_reinit: Force recreation of database instance (default: False)
            
        Returns:
            CryptoDatabase instance (singleton)
            
        Raises:
            RuntimeError: If database initialization fails critically
        """
        with self._lock:
            self._initialization_status['instance_count'] += 1
            self._initialization_status['last_access_time'] = datetime.now()
            
            # Return existing instance if available and not forcing reinit
            if self._database_instance is not None and not force_reinit:
                logger.logger.debug(f"📊 Returning existing database instance (access #{self._initialization_status['instance_count']})")
                return self._database_instance
            
            # Create new database instance
            try:
                logger.logger.info("🔨 Creating new CryptoDatabase instance via DatabaseManager...")
                
                # Import here to avoid circular imports
                from database import CryptoDatabase
                
                # Create the database instance
                initialization_start = datetime.now()
                self._database_instance = CryptoDatabase(db_path=db_path, enterprise_mode=enterprise_mode)
                initialization_time = datetime.now() - initialization_start
                
                # Update status
                self._initialization_status['initialized'] = True
                self._initialization_status['initialization_time'] = initialization_time.total_seconds()
                
                # Log success
                logger.logger.info(f"✅ Database instance created successfully via DatabaseManager")
                logger.logger.info(f"⏱️ Initialization time: {initialization_time.total_seconds():.3f}s")
                logger.logger.info(f"📍 Database path: {self._database_instance.db_path}")
                logger.logger.info(f"🏢 Enterprise mode: {'ENABLED' if self._database_instance.enterprise_mode else 'DISABLED'}")
                
                return self._database_instance
                
            except Exception as e:
                # Record error
                error_details = {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'db_path': db_path,
                    'enterprise_mode': enterprise_mode
                }
                self._initialization_status['initialization_errors'].append(error_details)
                
                # Log error
                logger.logger.error(f"❌ CRITICAL: DatabaseManager failed to create database instance: {e}")
                logger.logger.debug(f"Database initialization error details: {error_details}")
                
                # Clean up failed instance
                self._database_instance = None
                
                # Re-raise as RuntimeError for enterprise error handling
                raise RuntimeError(f"Enterprise DatabaseManager initialization failure: {e}") from e
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current DatabaseManager status for health monitoring
        
        Returns:
            Dictionary with comprehensive status information
        """
        with self._lock:
            status = {
                'singleton_status': {
                    'instance_exists': self._database_instance is not None,
                    'initialization_status': self._initialization_status['initialized'],
                    'instance_access_count': self._initialization_status['instance_count'],
                    'last_access_time': self._initialization_status['last_access_time'].isoformat() if self._initialization_status['last_access_time'] else None,
                    'initialization_time_seconds': self._initialization_status['initialization_time'],
                    'initialization_errors_count': len(self._initialization_status['initialization_errors'])
                },
                'database_status': {},
                'manager_info': {
                    'manager_creation_time': self._creation_time.isoformat(),
                    'manager_class': self.__class__.__name__,
                    'thread_safe': True,
                    'pattern': 'singleton'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Add database-specific status if instance exists
            if self._database_instance is not None:
                try:
                    status['database_status'] = {
                        'db_path': self._database_instance.db_path,
                        'enterprise_mode': self._database_instance.enterprise_mode,
                        'database_file_exists': os.path.exists(self._database_instance.db_path),
                        'database_file_size_bytes': os.path.getsize(self._database_instance.db_path) if os.path.exists(self._database_instance.db_path) else 0
                    }
                    
                    # Add enterprise status if available
                    if hasattr(self._database_instance, 'get_enterprise_status'):
                        try:
                            enterprise_status = self._database_instance.get_enterprise_status()
                            status['database_status']['enterprise_status'] = enterprise_status
                        except Exception as e:
                            status['database_status']['enterprise_status_error'] = str(e)
                            
                except Exception as e:
                    status['database_status']['error'] = str(e)
            
            return status
    
    def reset_instance(self, force: bool = False) -> Dict[str, Any]:
        """
        Reset the database instance (enterprise operation)
        
        Args:
            force: Force reset even if operations are in progress
            
        Returns:
            Dictionary with reset operation results
        """
        if not force:
            logger.logger.warning("⚠️ Database instance reset requested - this will affect all active connections")
        
        with self._lock:
            old_instance_info = {
                'existed': self._database_instance is not None,
                'db_path': getattr(self._database_instance, 'db_path', None) if self._database_instance else None,
                'enterprise_mode': getattr(self._database_instance, 'enterprise_mode', None) if self._database_instance else None
            }
            
            # Clean up existing instance
            try:
                if self._database_instance is not None:
                    # Simply clear the instance reference - SQLite handles cleanup automatically
                    self._database_instance = None
                    logger.logger.info("🔄 Database instance reset successfully")
                
            except Exception as e:
                logger.logger.warning(f"⚠️ Error during database instance cleanup: {e}")
            
            # Reset initialization status
            self._initialization_status['initialized'] = False
            self._initialization_status['initialization_time'] = None
            
            reset_info = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'old_instance': old_instance_info,
                'force_reset': force
            }
            
            logger.logger.info("✅ DatabaseManager instance reset completed")
            return reset_info
    
    def get_initialization_errors(self) -> list:
        """Get list of initialization errors for troubleshooting"""
        with self._lock:
            return self._initialization_status['initialization_errors'].copy()
    
    @classmethod
    def get_instance(cls) -> 'DatabaseManager':
        """
        Class method to get the singleton instance
        Alternative access pattern for enterprise applications
        """
        return cls()


# Enterprise convenience functions for backward compatibility
def get_database_instance(db_path: Optional[str] = None, enterprise_mode: bool = True) -> 'CryptoDatabase':
    """
    Enterprise function to get database instance via DatabaseManager
    Provides backward compatibility while using singleton pattern
    """
    manager = DatabaseManager()
    return manager.get_database(db_path=db_path, enterprise_mode=enterprise_mode)

def get_database_manager_status() -> Dict[str, Any]:
    """Get comprehensive DatabaseManager status for monitoring"""
    manager = DatabaseManager()
    return manager.get_status()

class CryptoDatabase:
    """Database handler for cryptocurrency market data, analysis and predictions"""
    
    def __init__(self, db_path: Optional[str] = None, enterprise_mode: bool = True):
        """
        Initialize database connection and create tables if they don't exist
        
        Args:
            db_path: Path to database file (optional)
            enterprise_mode: Enable enterprise features (default: True)
        """
        
        # 🆕 ENTERPRISE: Store enterprise mode setting FIRST
        self.enterprise_mode = enterprise_mode
        
        # Use centralized path resolver if no path provided
        if db_path is None:
            try:
                from config import Config
                db_path = Config.get_database_path()
                logger.logger.debug(f"Using centralized database path: {db_path}")
            except ImportError:
                # Fallback to relative path if config import fails
                db_path = "data/crypto_history.db"
                logger.logger.warning("Config import failed, using fallback relative path")
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.local = threading.local()  # Thread-local storage
        
        # Add comprehensive path validation logging
        logger.logger.info(f"CryptoDatabase initializing with path: {self.db_path}")
        logger.logger.debug(f"Enterprise mode: {'ENABLED' if self.enterprise_mode else 'DISABLED'}")
        logger.logger.debug(f"Database file exists: {os.path.exists(self.db_path)}")
        if os.path.exists(self.db_path):
            logger.logger.debug(f"Database file size: {os.path.getsize(self.db_path)} bytes")
            logger.logger.debug(f"Database file permissions: readable={os.access(self.db_path, os.R_OK)}, writable={os.access(self.db_path, os.W_OK)}")
        
        # Initialize core database structure
        self._initialize_database()
        self.add_ichimoku_column()
        self.add_missing_columns()
        self.add_replied_posts_table()
        self.add_price_history_table()
        self._ensure_reply_restrictions_table_exists()
        
        # 🆕 ENTERPRISE: Apply enterprise migrations if enabled
        if self.enterprise_mode:
            logger.logger.info("🚀 Applying enterprise schema migrations...")
            try:
                migration_results = self._apply_enterprise_migrations() if not self._is_migration_applied("enterprise_phase_2_1") else {'skipped': True, 'reason': 'Migration already applied'}
                
                # Log migration results summary
                tables_created = len(migration_results.get('tables_created', []))
                columns_added = len(migration_results.get('columns_added', []))
                indexes_created = len(migration_results.get('indexes_created', []))
                config_inserted = len(migration_results.get('config_inserted', []))
                errors = len(migration_results.get('errors', []))
                
                if tables_created > 0:
                    logger.logger.info(f"✅ Enterprise tables created: {', '.join(migration_results['tables_created'])}")
                
                if columns_added > 0:
                    logger.logger.info(f"🔧 Enterprise columns added: {columns_added}")
                    # Log specific columns for debugging
                    for column in migration_results['columns_added'][:5]:  # Show first 5
                        logger.logger.debug(f"   + {column}")
                    if columns_added > 5:
                        logger.logger.debug(f"   + ... and {columns_added - 5} more columns")
                
                if indexes_created > 0:
                    logger.logger.info(f"🔍 Performance indexes created: {indexes_created}")
                
                if config_inserted > 0:
                    logger.logger.info(f"⚙️ Configurations inserted: {config_inserted}")
                
                if errors > 0:
                    logger.logger.warning(f"⚠️ Migration completed with {errors} errors")
                    # Log first few errors for debugging
                    for error in migration_results['errors'][:3]:
                        logger.logger.debug(f"   - {error}")
                    if errors > 3:
                        logger.logger.debug(f"   - ... and {errors - 3} more errors")
                else:
                    logger.logger.info("🎉 Enterprise migration completed successfully!")
                
                # Verify enterprise status
                try:
                    enterprise_status = self._check_enterprise_schema_status()
                    overall_status = enterprise_status.get('overall_status', 'unknown')
                    
                    if overall_status == 'ready':
                        logger.logger.info("✅ Enterprise features fully operational")
                    elif overall_status == 'partial':
                        logger.logger.warning("⚠️ Enterprise features partially operational")
                        missing = enterprise_status.get('missing_components', [])
                        if missing:
                            logger.logger.debug(f"Missing components: {missing[:3]}")
                    else:
                        logger.logger.warning(f"⚠️ Enterprise status: {overall_status}")
                        
                except Exception as status_error:
                    logger.logger.debug(f"Could not verify enterprise status: {status_error}")
                    
            except Exception as e:
                logger.logger.error(f"❌ Enterprise migration failed: {e}")
                logger.logger.warning("🔄 Falling back to basic database mode")
                self.enterprise_mode = False
                
                # Log the specific error for debugging
                logger.logger.debug(f"Migration error details: {str(e)}")
                import traceback
                logger.logger.debug(f"Migration traceback: {traceback.format_exc()}")
        else:
            logger.logger.info("📊 Enterprise mode disabled - using basic database features")
        
        # Log final initialization status
        if self.enterprise_mode:
            logger.logger.info(f"✅ CryptoDatabase initialization complete: {self.db_path} (Enterprise Mode)")
        else:
            logger.logger.info(f"✅ CryptoDatabase initialization complete: {self.db_path} (Basic Mode)")
        
        # Run database migrations for TokenMappingManager compatibility
        try:
            logger.logger.info("🔍 Checking for required database migrations...")
            migration_success = self._run_market_data_migration_safely()
            if migration_success:
                logger.logger.info("✅ Database migrations completed successfully")
            else:
                logger.logger.warning("⚠️ Database migration skipped - system continues normally")
        except Exception as migration_error:
            logger.logger.error(f"🚨 Migration error (non-fatal): {migration_error}")
            logger.logger.warning("⚠️ System continuing with existing database schema")
        
        # 🆕 ENTERPRISE: Initialize enterprise components if enabled
        if self.enterprise_mode:
            try:
                # Initialize operation manager (Phase 3)
                from database import DatabaseOperationManager  # Import here to avoid circular imports
                self.operation_manager = DatabaseOperationManager(self)
                from database import DatabaseHealthMonitor
                self.health_monitor = DatabaseHealthMonitor(self)
                
                logger.logger.info("✅ Enterprise operation manager and health monitor initialized")
                
            except Exception as e:
                logger.logger.warning(f"Enterprise operation manager initialization warning: {e}")
                self.operation_manager = None
                self.health_monitor = None
        
        # Final validation and summary
        try:
            # Quick database connectivity test
            conn, cursor = self._get_connection()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            if self.enterprise_mode:
                # Test enterprise table access
                try:
                    cursor.execute("SELECT COUNT(*) FROM system_config")
                    config_count = cursor.fetchone()[0]
                    logger.logger.debug(f"Enterprise validation: {config_count} configuration entries")
                except Exception:
                    logger.logger.debug("Enterprise tables not yet fully accessible")
            
            logger.logger.debug("✅ Database connectivity validated")
            
        except Exception as validation_error:
            logger.logger.warning(f"Database validation warning: {validation_error}")
        
        # 🆕 ENTERPRISE: Log system capabilities summary
        capabilities = []
        capabilities.append("✅ Core database operations")
        capabilities.append("✅ Content analysis storage")
        capabilities.append("✅ Reply tracking")
        capabilities.append("✅ Market data storage")
        capabilities.append("✅ Technical indicators")
        
        if self.enterprise_mode:
            capabilities.append("✅ Enterprise audit trail")
            capabilities.append("✅ Conflict resolution")
            capabilities.append("✅ Health monitoring")
            capabilities.append("✅ Configuration management")
            capabilities.append("✅ Retry policies")
        else:
            capabilities.append("⚪ Enterprise features (disabled)")
        
        logger.logger.info("🚀 Database capabilities:")
        for capability in capabilities:
            logger.logger.info(f"   {capability}")
        
        # Final success message
        mode_description = "Enterprise" if self.enterprise_mode else "Basic"
        logger.logger.info(f"🎉 {mode_description} CryptoDatabase ready for operations!")

        # Initialize enhanced data system for multi-source data aggregation
        self.enhanced_data_system = create_enhanced_data_system(self)
        logger.logger.info("✅ Enhanced multi-source data system initialized")

    def _run_market_data_migration_safely(self) -> bool:
        """
        Safely run the market_data migration with comprehensive error handling
        Returns True if migration succeeded or was already completed
        Returns False if migration failed but system should continue
        """
        conn = None
        try:
            # Import TokenMappingManager with fallback
            try:
                from config import TokenMappingManager
                logger.logger.debug("✅ TokenMappingManager imported for migration")
            except ImportError as import_error:
                logger.logger.warning(f"⚠️ TokenMappingManager not available: {import_error}")
                logger.logger.info("📝 Migration skipped - TokenMappingManager required")
                return False
            
            # Check if migration is needed
            conn, cursor = self._get_connection()
            cursor.execute("PRAGMA table_info(market_data)")
            existing_columns = [col[1] for col in cursor.fetchall()]
            
            symbol_exists = 'symbol' in existing_columns
            coin_id_exists = 'coin_id' in existing_columns
            
            if symbol_exists and coin_id_exists:
                logger.logger.debug("✅ Migration not needed - columns already exist")
                return True
            
            # Check if we have any data to migrate
            cursor.execute("SELECT COUNT(*) as count FROM market_data")
            record_count = cursor.fetchone()['count']
            
            if record_count == 0:
                logger.logger.info("📝 No existing data - adding columns for future use")
                
                # Just add the columns for future use
                if not symbol_exists:
                    cursor.execute("ALTER TABLE market_data ADD COLUMN symbol TEXT")
                    logger.logger.info("➕ Added symbol column to market_data table")
                
                if not coin_id_exists:
                    cursor.execute("ALTER TABLE market_data ADD COLUMN coin_id TEXT")
                    logger.logger.info("➕ Added coin_id column to market_data table")
                
                # Add indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_coin_id ON market_data(coin_id)")
                
                conn.commit()
                logger.logger.info("✅ Empty table migration completed")
                return True
            
            # We have data and need migration - run the full migration
            logger.logger.info(f"🔄 Running migration for {record_count} existing records...")
            
            # Call the full migration method
            return self.migrate_market_data_add_symbol_columns()
            
        except Exception as e:
            # Log the error but don't crash the system
            logger.logger.error(f"🚨 Migration safety check failed: {str(e)}")
            logger.logger.debug(f"Migration error details: {traceback.format_exc()}")
            
            # Try to rollback any partial changes (only if conn was successfully created)
            if conn is not None:
                try:
                    conn.rollback()
                    logger.logger.info("🔄 Rolled back partial migration changes")
                except Exception as rollback_error:
                    logger.logger.warning(f"⚠️ Rollback also failed: {rollback_error}")
            
            return False
        
    def _apply_enterprise_migrations(self):
        """
        Apply enterprise database schema migrations - Phase 2.1 Implementation
        
        This method will:
        1. Create 5 new enterprise tables
        2. Add enterprise columns to existing tables
        3. Create performance indexes
        4. Insert default configuration
        5. Handle all errors gracefully with rollback
        
        Safe to run multiple times - uses IF NOT EXISTS patterns
        """
        
        # Check if enterprise migration already applied
        if self._is_migration_applied("enterprise_phase_2_1"):
            logger.logger.info("⏭️ Enterprise Phase 2.1 migration already applied, skipping...")
            return {'skipped': True, 'reason': 'Migration already applied'}

        migration_start_time = time.time()
        migration_results = {
            'tables_created': [],
            'columns_added': [],
            'indexes_created': [],
            'config_inserted': [],
            'errors': [],
            'warnings': []
        }
        
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            logger.logger.info("🚀 Starting enterprise schema migration (Phase 2.1)...")
            
            # ========================================================================
            # STEP 1: Create Enterprise Tables
            # ========================================================================
            
            logger.logger.info("📋 Step 1: Creating enterprise tables...")
            
            # 1.1 Operation Audit Table
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS operation_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_id TEXT NOT NULL UNIQUE,
                        table_name TEXT NOT NULL,
                        operation_type TEXT NOT NULL,
                        record_key TEXT NOT NULL,
                        attempt_count INTEGER DEFAULT 1,
                        status TEXT NOT NULL,
                        error_details TEXT,
                        conflict_resolution_strategy TEXT,
                        execution_time_ms INTEGER,
                        created_at DATETIME NOT NULL,
                        completed_at DATETIME,
                        metadata JSON
                    )
                """)
                
                # Check if table was actually created
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='operation_audit'")
                if cursor.fetchone():
                    migration_results['tables_created'].append('operation_audit')
                    logger.logger.info("✅ Created operation_audit table")
                
            except Exception as e:
                migration_results['errors'].append(f"operation_audit table: {str(e)}")
                logger.logger.error(f"❌ Failed to create operation_audit table: {e}")
            
            # 1.2 Database Health Metrics Table
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS db_health_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        table_name TEXT NOT NULL,
                        successful_operations INTEGER DEFAULT 0,
                        failed_operations INTEGER DEFAULT 0,
                        constraint_violations INTEGER DEFAULT 0,
                        conflict_resolutions INTEGER DEFAULT 0,
                        avg_operation_time_ms REAL DEFAULT 0.0,
                        max_operation_time_ms INTEGER DEFAULT 0,
                        min_operation_time_ms INTEGER DEFAULT 0,
                        lock_conflicts INTEGER DEFAULT 0,
                        connection_failures INTEGER DEFAULT 0,
                        retry_operations INTEGER DEFAULT 0,
                        table_size_bytes INTEGER DEFAULT 0,
                        index_size_bytes INTEGER DEFAULT 0,
                        metadata JSON
                    )
                """)
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='db_health_metrics'")
                if cursor.fetchone():
                    migration_results['tables_created'].append('db_health_metrics')
                    logger.logger.info("✅ Created db_health_metrics table")
                
            except Exception as e:
                migration_results['errors'].append(f"db_health_metrics table: {str(e)}")
                logger.logger.error(f"❌ Failed to create db_health_metrics table: {e}")
            
            # 1.3 Conflict Resolution Log Table
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conflict_resolution_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        operation_id TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        record_id TEXT NOT NULL,
                        conflict_type TEXT NOT NULL,
                        constraint_name TEXT,
                        resolution_strategy TEXT NOT NULL,
                        resolution_successful BOOLEAN NOT NULL,
                        resolution_time_ms INTEGER,
                        old_values JSON,
                        new_values JSON,
                        final_values JSON
                    )
                """)
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conflict_resolution_log'")
                if cursor.fetchone():
                    migration_results['tables_created'].append('conflict_resolution_log')
                    logger.logger.info("✅ Created conflict_resolution_log table")
                
            except Exception as e:
                migration_results['errors'].append(f"conflict_resolution_log table: {str(e)}")
                logger.logger.error(f"❌ Failed to create conflict_resolution_log table: {e}")
            
            # 1.4 System Configuration Table
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_config (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        config_key TEXT UNIQUE NOT NULL,
                        config_value TEXT NOT NULL,
                        config_type TEXT NOT NULL,
                        description TEXT,
                        is_sensitive BOOLEAN DEFAULT FALSE,
                        updated_at DATETIME NOT NULL,
                        updated_by TEXT DEFAULT 'system'
                    )
                """)
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_config'")
                if cursor.fetchone():
                    migration_results['tables_created'].append('system_config')
                    logger.logger.info("✅ Created system_config table")
                
            except Exception as e:
                migration_results['errors'].append(f"system_config table: {str(e)}")
                logger.logger.error(f"❌ Failed to create system_config table: {e}")
            
            # 1.5 Retry Policies Table
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS retry_policies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        policy_name TEXT UNIQUE NOT NULL,
                        table_name TEXT NOT NULL,
                        max_retries INTEGER NOT NULL DEFAULT 3,
                        backoff_strategy TEXT NOT NULL DEFAULT 'EXPONENTIAL',
                        base_delay_ms INTEGER NOT NULL DEFAULT 100,
                        max_delay_ms INTEGER NOT NULL DEFAULT 5000,
                        retry_on_constraint_violation BOOLEAN DEFAULT TRUE,
                        retry_on_lock_timeout BOOLEAN DEFAULT TRUE,
                        retry_on_connection_error BOOLEAN DEFAULT TRUE,
                        enabled BOOLEAN DEFAULT TRUE,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME
                    )
                """)
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='retry_policies'")
                if cursor.fetchone():
                    migration_results['tables_created'].append('retry_policies')
                    logger.logger.info("✅ Created retry_policies table")
                
            except Exception as e:
                migration_results['errors'].append(f"retry_policies table: {str(e)}")
                logger.logger.error(f"❌ Failed to create retry_policies table: {e}")
            
            # ========================================================================
            # STEP 2: Create Performance Indexes
            # ========================================================================
            
            logger.logger.info("🔍 Step 2: Creating performance indexes...")
            
            enterprise_indexes = [
                # Operation Audit Indexes
                ("idx_operation_audit_operation_id", "operation_audit", "operation_id"),
                ("idx_operation_audit_table_status", "operation_audit", "table_name, status"),
                ("idx_operation_audit_created_at", "operation_audit", "created_at"),
                ("idx_operation_audit_record_key", "operation_audit", "record_key"),
                
                # Health Metrics Indexes
                ("idx_db_health_timestamp", "db_health_metrics", "timestamp"),
                ("idx_db_health_table", "db_health_metrics", "table_name"),
                ("idx_db_health_table_timestamp", "db_health_metrics", "table_name, timestamp"),
                
                # Conflict Resolution Indexes
                ("idx_conflict_log_timestamp", "conflict_resolution_log", "timestamp"),
                ("idx_conflict_log_operation_id", "conflict_resolution_log", "operation_id"),
                ("idx_conflict_log_table_record", "conflict_resolution_log", "table_name, record_id"),
                
                # System Config Indexes
                ("idx_system_config_key", "system_config", "config_key"),
                
                # Retry Policies Indexes
                ("idx_retry_policies_table", "retry_policies", "table_name"),
                ("idx_retry_policies_enabled", "retry_policies", "enabled")
            ]
            
            for index_name, table_name, columns in enterprise_indexes:
                try:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({columns})")
                    migration_results['indexes_created'].append(index_name)
                    logger.logger.debug(f"✅ Created index {index_name}")
                except Exception as e:
                    migration_results['errors'].append(f"index {index_name}: {str(e)}")
                    logger.logger.warning(f"⚠️ Failed to create index {index_name}: {e}")
            
            logger.logger.info(f"✅ Created {len(migration_results['indexes_created'])} performance indexes")
            
            # ========================================================================
            # STEP 3: Add Enterprise Columns to Existing Tables
            # ========================================================================
            
            logger.logger.info("🔧 Step 3: Adding enterprise columns to existing tables...")
            
            # 3.1 Enhance content_analysis table
            try:
                cursor.execute("PRAGMA table_info(content_analysis)")
                existing_columns = {col[1] for col in cursor.fetchall()}
                
                enterprise_columns_content_analysis = [
                    ('version', 'INTEGER DEFAULT 1'),
                    ('created_by', 'TEXT DEFAULT "system"'),
                    ('updated_at', 'DATETIME'),
                    ('operation_id', 'TEXT'),
                    ('conflict_resolution_applied', 'BOOLEAN DEFAULT FALSE')
                ]
                
                for column_name, column_def in enterprise_columns_content_analysis:
                    if column_name not in existing_columns:
                        try:
                            cursor.execute(f"ALTER TABLE content_analysis ADD COLUMN {column_name} {column_def}")
                            migration_results['columns_added'].append(f"content_analysis.{column_name}")
                            logger.logger.info(f"✅ Added column '{column_name}' to content_analysis table")
                        except Exception as e:
                            migration_results['errors'].append(f"content_analysis.{column_name}: {str(e)}")
                            logger.logger.warning(f"⚠️ Failed to add column {column_name} to content_analysis: {e}")
                
                # Create composite index for content_analysis
                try:
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_content_analysis_composite 
                        ON content_analysis(post_id, version, updated_at)
                    """)
                    migration_results['indexes_created'].append('idx_content_analysis_composite')
                    logger.logger.debug("✅ Created composite index for content_analysis")
                except Exception as e:
                    migration_results['warnings'].append(f"content_analysis composite index: {str(e)}")
                    
            except Exception as e:
                migration_results['errors'].append(f"content_analysis table enhancement: {str(e)}")
                logger.logger.error(f"❌ Failed to enhance content_analysis table: {e}")
            
            # 3.2 Enhance replied_posts table
            try:
                cursor.execute("PRAGMA table_info(replied_posts)")
                existing_columns = {col[1] for col in cursor.fetchall()}
                
                enterprise_columns_replied_posts = [
                    ('operation_id', 'TEXT'),
                    ('retry_count', 'INTEGER DEFAULT 0'),
                    ('last_attempt_at', 'DATETIME'),
                    ('status', 'TEXT DEFAULT "active"')
                ]
                
                for column_name, column_def in enterprise_columns_replied_posts:
                    if column_name not in existing_columns:
                        try:
                            cursor.execute(f"ALTER TABLE replied_posts ADD COLUMN {column_name} {column_def}")
                            migration_results['columns_added'].append(f"replied_posts.{column_name}")
                            logger.logger.info(f"✅ Added column '{column_name}' to replied_posts table")
                        except Exception as e:
                            migration_results['errors'].append(f"replied_posts.{column_name}: {str(e)}")
                            logger.logger.warning(f"⚠️ Failed to add column {column_name} to replied_posts: {e}")
                
                # Create performance indexes for replied_posts
                enterprise_replied_posts_indexes = [
                    ("idx_replied_posts_operation_id", "operation_id"),
                    ("idx_replied_posts_status", "status"),
                    ("idx_replied_posts_last_attempt", "last_attempt_at")
                ]
                
                for index_name, column in enterprise_replied_posts_indexes:
                    try:
                        cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON replied_posts({column})")
                        migration_results['indexes_created'].append(index_name)
                        logger.logger.debug(f"✅ Created index {index_name}")
                    except Exception as e:
                        migration_results['warnings'].append(f"replied_posts index {index_name}: {str(e)}")
                        
            except Exception as e:
                migration_results['errors'].append(f"replied_posts table enhancement: {str(e)}")
                logger.logger.error(f"❌ Failed to enhance replied_posts table: {e}")
            
            # ========================================================================
            # STEP 4: Insert Default Configuration
            # ========================================================================
            
            logger.logger.info("⚙️ Step 4: Inserting default system configuration...")
            
            default_configs = [
                ('enterprise_mode_enabled', 'true', 'BOOLEAN', 'Enable enterprise database features'),
                ('audit_trail_enabled', 'true', 'BOOLEAN', 'Enable operation audit logging'),
                ('health_monitoring_enabled', 'true', 'BOOLEAN', 'Enable database health monitoring'),
                ('conflict_resolution_strategy', 'UPSERT', 'STRING', 'Default conflict resolution: UPSERT, IGNORE, RETRY'),
                ('max_operation_retries', '3', 'INTEGER', 'Maximum retry attempts for failed operations'),
                ('operation_timeout_ms', '30000', 'INTEGER', 'Operation timeout in milliseconds'),
                ('health_metrics_interval_minutes', '60', 'INTEGER', 'How often to collect health metrics'),
                ('audit_retention_days', '90', 'INTEGER', 'How long to keep audit records'),
                ('conflict_resolution_content_analysis', 'UPSERT', 'STRING', 'Conflict resolution for content_analysis table'),
                ('conflict_resolution_replied_posts', 'UPSERT', 'STRING', 'Conflict resolution for replied_posts table'),
                ('conflict_resolution_posted_content', 'RETRY', 'STRING', 'Conflict resolution for posted_content table'),
                ('conflict_resolution_market_data', 'IGNORE', 'STRING', 'Conflict resolution for market_data table')
            ]
            
            for config_key, config_value, config_type, description in default_configs:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO system_config 
                        (config_key, config_value, config_type, description, updated_at) 
                        VALUES (?, ?, ?, ?, ?)
                    """, (config_key, config_value, config_type, description, datetime.now()))
                    
                    if cursor.rowcount > 0:
                        migration_results['config_inserted'].append(config_key)
                        logger.logger.debug(f"✅ Inserted config: {config_key}")
                        
                except Exception as e:
                    migration_results['errors'].append(f"config {config_key}: {str(e)}")
                    logger.logger.warning(f"⚠️ Failed to insert config {config_key}: {e}")
            
            logger.logger.info(f"✅ Inserted {len(migration_results['config_inserted'])} configuration entries")
            
            # ========================================================================
            # STEP 5: Insert Default Retry Policies
            # ========================================================================
            
            logger.logger.info("🔄 Step 5: Inserting default retry policies...")
            
            default_policies = [
                ('content_analysis_default', 'content_analysis', 3, 'EXPONENTIAL', 100, 2000),
                ('replied_posts_default', 'replied_posts', 2, 'LINEAR', 200, 1000),
                ('posted_content_default', 'posted_content', 3, 'EXPONENTIAL', 150, 3000),
                ('market_data_default', 'market_data', 5, 'FIXED', 50, 500),
                ('reply_restrictions_default', 'reply_restrictions', 2, 'LINEAR', 100, 1000)
            ]
            
            retry_policies_inserted = 0
            for policy_name, table_name, max_retries, strategy, base_delay, max_delay in default_policies:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO retry_policies 
                        (policy_name, table_name, max_retries, backoff_strategy, 
                        base_delay_ms, max_delay_ms, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (policy_name, table_name, max_retries, strategy, base_delay, max_delay, datetime.now()))
                    
                    if cursor.rowcount > 0:
                        retry_policies_inserted += 1
                        logger.logger.debug(f"✅ Inserted retry policy: {policy_name}")
                        
                except Exception as e:
                    migration_results['errors'].append(f"retry policy {policy_name}: {str(e)}")
                    logger.logger.warning(f"⚠️ Failed to insert retry policy {policy_name}: {e}")
            
            logger.logger.info(f"✅ Inserted {retry_policies_inserted} retry policies")
            
            # ========================================================================
            # STEP 6: Commit All Changes
            # ========================================================================
            
            conn.commit()
            
            migration_time = time.time() - migration_start_time
            
            # ========================================================================
            # STEP 7: Migration Summary
            # ========================================================================
            
            logger.logger.info("📊 Enterprise schema migration completed!")
            logger.logger.info(f"⏱️ Migration time: {migration_time:.2f} seconds")
            logger.logger.info(f"📋 Tables created: {len(migration_results['tables_created'])}")
            logger.logger.info(f"🔧 Columns added: {len(migration_results['columns_added'])}")
            logger.logger.info(f"🔍 Indexes created: {len(migration_results['indexes_created'])}")
            logger.logger.info(f"⚙️ Configs inserted: {len(migration_results['config_inserted'])}")
            
            if migration_results['errors']:
                logger.logger.warning(f"⚠️ Migration completed with {len(migration_results['errors'])} errors")
                for error in migration_results['errors'][:5]:  # Show first 5 errors
                    logger.logger.warning(f"   - {error}")
            
            if migration_results['warnings']:
                logger.logger.info(f"ℹ️ Migration completed with {len(migration_results['warnings'])} warnings")
            
            # Set enterprise mode flag
            if not hasattr(self, 'enterprise_mode'):
                self.enterprise_mode = True
                logger.logger.info("✅ Enterprise mode enabled")
            
            return migration_results
            
        except Exception as e:
            # Critical migration failure - rollback everything
            logger.log_error("Enterprise Migration Critical Failure", str(e))
            logger.logger.error(f"❌ CRITICAL: Enterprise migration failed: {e}")
            
            if conn:
                try:
                    conn.rollback()
                    logger.logger.warning("🔄 Database changes rolled back due to critical failure")
                except Exception as rollback_error:
                    logger.logger.error(f"❌ CRITICAL: Rollback failed: {rollback_error}")
            
            migration_results['errors'].append(f"CRITICAL FAILURE: {str(e)}")
            return migration_results

    def _check_enterprise_schema_status(self) -> Dict[str, Any]:
        """
        Check the current status of enterprise schema implementation
        
        Returns:
            Dictionary with detailed status of enterprise features
        """
        
        status = {
            'enterprise_tables': {},
            'enterprise_columns': {},
            'enterprise_indexes': {},
            'configuration_status': {},
            'overall_status': 'unknown',
            'missing_components': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            conn, cursor = self._get_connection()
            
            # Check enterprise tables
            required_tables = [
                'operation_audit', 'db_health_metrics', 'conflict_resolution_log',
                'system_config', 'retry_policies'
            ]
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ({})
            """.format(','.join('?' * len(required_tables))), required_tables)
            
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            for table in required_tables:
                status['enterprise_tables'][table] = table in existing_tables
            
            # Check enterprise columns in existing tables
            tables_to_check = [
                ('content_analysis', ['version', 'created_by', 'updated_at', 'operation_id', 'conflict_resolution_applied']),
                ('replied_posts', ['operation_id', 'retry_count', 'last_attempt_at', 'status'])
            ]
            
            for table_name, required_columns in tables_to_check:
                try:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    existing_columns = {col[1] for col in cursor.fetchall()}
                    
                    status['enterprise_columns'][table_name] = {}
                    for column in required_columns:
                        status['enterprise_columns'][table_name][column] = column in existing_columns
                        
                except Exception as e:
                    status['enterprise_columns'][table_name] = f"Error: {str(e)}"
            
            # Check configuration status
            if 'system_config' in existing_tables:
                try:
                    cursor.execute("SELECT config_key, config_value FROM system_config")
                    configs = dict(cursor.fetchall())
                    status['configuration_status'] = {
                        'enterprise_mode_enabled': configs.get('enterprise_mode_enabled', 'missing'),
                        'audit_trail_enabled': configs.get('audit_trail_enabled', 'missing'),
                        'health_monitoring_enabled': configs.get('health_monitoring_enabled', 'missing'),
                        'total_configs': len(configs)
                    }
                except Exception as e:
                    status['configuration_status'] = f"Error: {str(e)}"
            
            # Determine overall status
            tables_ready = all(status['enterprise_tables'].values())
            columns_ready = all(
                all(cols.values()) if isinstance(cols, dict) else False 
                for cols in status['enterprise_columns'].values()
            )
            config_ready = isinstance(status['configuration_status'], dict) and \
                        status['configuration_status'].get('enterprise_mode_enabled') == 'true'
            
            if tables_ready and columns_ready and config_ready:
                status['overall_status'] = 'ready'
            elif any(status['enterprise_tables'].values()):
                status['overall_status'] = 'partial'
            else:
                status['overall_status'] = 'not_implemented'
            
            # Identify missing components
            for table, exists in status['enterprise_tables'].items():
                if not exists:
                    status['missing_components'].append(f"table: {table}")
            
            for table, columns in status['enterprise_columns'].items():
                if isinstance(columns, dict):
                    for column, exists in columns.items():
                        if not exists:
                            status['missing_components'].append(f"column: {table}.{column}")
            
            return status
            
        except Exception as e:
            status['overall_status'] = 'error'
            status['error'] = str(e)
            return status  
        
    def _get_schema_version(self) -> str:
        """
        Get current schema version from system_config table
        
        Returns:
            Current schema version string (e.g., "2.1.0") or "0.0.0" if not set
        """
        try:
            conn, cursor = self._get_connection()
            
            cursor.execute("""
                SELECT config_value FROM system_config 
                WHERE config_key = 'schema_version'
            """)
            
            result = cursor.fetchone()
            if result:
                return result['config_value']
            else:
                # Schema version not set - this is a new installation
                return "0.0.0"
                
        except Exception as e:
            # If system_config table doesn't exist, this is a pre-enterprise installation
            logger.logger.debug(f"Schema version check failed (expected for new installs): {e}")
            return "0.0.0"

    def _set_schema_version(self, version: str, migration_id: str) -> bool:
        """
        Set schema version after successful migration
        
        Args:
            version: Schema version string (e.g., "2.1.0")
            migration_id: Migration identifier for audit trail
            
        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            # Update or insert schema version
            cursor.execute("""
                INSERT OR REPLACE INTO system_config 
                (config_key, config_value, config_type, description, updated_at) 
                VALUES (?, ?, 'STRING', 'Current database schema version', ?)
            """, ('schema_version', version, datetime.now()))
            
            # Record last migration ID
            cursor.execute("""
                INSERT OR REPLACE INTO system_config 
                (config_key, config_value, config_type, description, updated_at) 
                VALUES (?, ?, 'STRING', 'Last applied migration identifier', ?)
            """, ('last_migration_id', migration_id, datetime.now()))
            
            # Record migration timestamp
            cursor.execute("""
                INSERT OR REPLACE INTO system_config 
                (config_key, config_value, config_type, description, updated_at) 
                VALUES (?, ?, 'DATETIME', 'Timestamp of last migration', ?)
            """, ('last_migration_timestamp', datetime.now().isoformat(), datetime.now()))
            
            conn.commit()
            logger.logger.info(f"✅ Schema version updated to {version} (migration: {migration_id})")
            return True
            
        except Exception as e:
            logger.logger.error(f"❌ Failed to set schema version {version}: {e}")
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False

    def _is_migration_applied(self, migration_id: str) -> bool:
        """
        Check if specific migration has already been applied
        
        Args:
            migration_id: Migration identifier to check
            
        Returns:
            True if migration already applied, False if needs to be run
        """
        try:
            conn, cursor = self._get_connection()
            
            # Check if this specific migration was already applied
            cursor.execute("""
                SELECT config_value FROM system_config 
                WHERE config_key = 'last_migration_id'
            """)
            
            result = cursor.fetchone()
            if result and result['config_value'] == migration_id:
                logger.logger.debug(f"Migration {migration_id} already applied")
                return True
                
            # Also check schema version for additional validation
            current_version = self._get_schema_version()
            
            # Enterprise Phase 2.1 corresponds to schema version 2.1.0+
            if migration_id == "enterprise_phase_2_1":
                if current_version and current_version != "0.0.0":
                    # Parse version to check if >= 2.1.0
                    try:
                        version_parts = [int(x) for x in current_version.split('.')]
                        if len(version_parts) >= 3:
                            major, minor, patch = version_parts[0], version_parts[1], version_parts[2]
                            if (major > 2) or (major == 2 and minor > 1) or (major == 2 and minor == 1 and patch >= 0):
                                logger.logger.debug(f"Migration {migration_id} already applied (version {current_version})")
                                return True
                    except ValueError:
                        logger.logger.debug(f"Could not parse version {current_version}, assuming migration needed")
            
            logger.logger.debug(f"Migration {migration_id} needs to be applied")
            return False
            
        except Exception as e:
            logger.logger.debug(f"Migration status check failed for {migration_id}: {e}")
            # If we can't determine status, assume migration is needed (safer)
            return False    

    def get_enterprise_status(self) -> Dict[str, Any]:
        """
        Get current status of enterprise features
        
        Returns:
            Dictionary with enterprise feature status
        """
        
        if not self.enterprise_mode:
            return {
                'enterprise_mode': False,
                'status': 'disabled',
                'message': 'Enterprise mode is disabled'
            }
        
        return self._check_enterprise_schema_status()
    
    def enable_enterprise_mode(self) -> Dict[str, Any]:
        """
        Enable enterprise mode and apply migrations
        
        Returns:
            Dictionary with migration results
        """
        
        if self.enterprise_mode:
            return {
                'success': True,
                'message': 'Enterprise mode already enabled',
                'status': self.get_enterprise_status()
            }
        
        logger.logger.info("🔄 Enabling enterprise mode...")
        
        self.enterprise_mode = True
        migration_results = self._apply_enterprise_migrations() if not self._is_migration_applied("enterprise_phase_2_1") else {'skipped': True, 'reason': 'Migration already applied'}
        
        return {
            'success': len(migration_results['errors']) == 0,
            'migration_results': migration_results,
            'status': self.get_enterprise_status()
        }
    
    def disable_enterprise_mode(self) -> Dict[str, Any]:
        """
        Disable enterprise mode (does not remove tables)
        
        Returns:
            Dictionary with status
        """
        
        self.enterprise_mode = False
        logger.logger.info("❌ Enterprise mode disabled")
        
        return {
            'success': True,
            'message': 'Enterprise mode disabled (enterprise tables preserved)',
            'enterprise_mode': False
        }
    
    def get_enterprise_config(self, config_key: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Get enterprise configuration value(s)
        
        Args:
            config_key: Specific config key to retrieve (optional)
            
        Returns:
            Config value if key specified, otherwise all configs
            Returns empty string if config key not found
            Returns error dict if enterprise mode disabled or error occurs
        """
        
        if not self.enterprise_mode:
            return {'error': 'Enterprise mode not enabled'}
        
        try:
            conn, cursor = self._get_connection()
            
            if config_key:
                cursor.execute("""
                    SELECT config_value FROM system_config 
                    WHERE config_key = ?
                """, (config_key,))
                
                result = cursor.fetchone()
                # Fix: Return empty string instead of None to match return type
                return result[0] if result else ""
            else:
                cursor.execute("""
                    SELECT config_key, config_value, config_type, description 
                    FROM system_config 
                    ORDER BY config_key
                """)
                
                configs = {}
                for row in cursor.fetchall():
                    configs[row[0]] = {
                        'value': row[1],
                        'type': row[2],
                        'description': row[3]
                    }
                
                return configs
                
        except Exception as e:
            logger.log_error("Get Enterprise Config", str(e))
            return {'error': str(e)}
    
    def set_enterprise_config(self, config_key: str, config_value: str, 
                            config_type: str = 'STRING', description: str = '') -> bool:
        """
        Set enterprise configuration value
        
        Args:
            config_key: Configuration key
            config_value: Configuration value
            config_type: Value type (STRING, INTEGER, BOOLEAN, JSON)
            description: Description of the config
            
        Returns:
            True if successful, False otherwise
        """
        
        if not self.enterprise_mode:
            logger.logger.warning("Cannot set enterprise config - enterprise mode not enabled")
            return False
        
        try:
            conn, cursor = self._get_connection()
            
            cursor.execute("""
                INSERT OR REPLACE INTO system_config 
                (config_key, config_value, config_type, description, updated_at, updated_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (config_key, config_value, config_type, description, datetime.now(), 'user'))
            
            conn.commit()
            logger.logger.info(f"✅ Updated enterprise config: {config_key} = {config_value}")
            return True
            
        except Exception as e:
            logger.log_error("Set Enterprise Config", str(e))
            return False
    
    def cleanup_enterprise_data(self, days_to_keep: int = 90) -> Dict[str, Union[int, str]]:
        """
        Clean up old enterprise audit data
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Dictionary with count of deleted records by table (int values)
            Or error information (str values) if enterprise mode disabled or error occurs
        """
        
        if not self.enterprise_mode:
            return {'error': 'Enterprise mode not enabled'}
        
        deleted_counts: Dict[str, Union[int, str]] = {}
        conn = None  # Fix: Initialize conn to None to avoid unbound variable warning
        
        try:
            conn, cursor = self._get_connection()
            
            # Clean up old operation audit records
            cursor.execute("""
                DELETE FROM operation_audit 
                WHERE created_at < datetime('now', '-' || ? || ' days')
            """, (days_to_keep,))
            deleted_counts['operation_audit'] = cursor.rowcount
            
            # Clean up old health metrics (keep longer than audit data)
            cursor.execute("""
                DELETE FROM db_health_metrics 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days_to_keep * 2,))
            deleted_counts['db_health_metrics'] = cursor.rowcount
            
            # Clean up old conflict resolution logs
            cursor.execute("""
                DELETE FROM conflict_resolution_log 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days_to_keep,))
            deleted_counts['conflict_resolution_log'] = cursor.rowcount
            
            conn.commit()
            
            # Calculate total only from int values
            total_deleted = sum(v for v in deleted_counts.values() if isinstance(v, int))
            logger.logger.info(f"🧹 Enterprise cleanup completed: {total_deleted} records deleted")
            logger.logger.debug(f"Cleanup details: {deleted_counts}")
            
            return deleted_counts
            
        except Exception as e:
            logger.log_error("Enterprise Data Cleanup", str(e))
            if conn is not None:  # Fix: Use 'is not None' for better type checking
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.logger.debug(f"Rollback error: {rollback_error}")
            return {'error': str(e)}
        
    def _enhance_existing_tables_for_enterprise(self):
        """
        Phase 2.2: Add enterprise columns to existing tables
        
        This method adds enterprise tracking columns to existing tables
        without modifying any business logic or methods.
        """
        
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            logger.logger.info("🔧 Phase 2.2: Enhancing existing tables with enterprise columns...")
            
            # Enhance content_analysis table
            cursor.execute("PRAGMA table_info(content_analysis)")
            existing_columns = {col[1] for col in cursor.fetchall()}
            
            enterprise_columns_content_analysis = [
                ('version', 'INTEGER DEFAULT 1'),
                ('created_by', 'TEXT DEFAULT "system"'),
                ('updated_at', 'DATETIME'),
                ('operation_id', 'TEXT'),
                ('conflict_resolution_applied', 'BOOLEAN DEFAULT FALSE')
            ]
            
            for column_name, column_def in enterprise_columns_content_analysis:
                if column_name not in existing_columns:
                    cursor.execute(f"ALTER TABLE content_analysis ADD COLUMN {column_name} {column_def}")
                    logger.logger.info(f"✅ Added column '{column_name}' to content_analysis table")
            
            # Enhance replied_posts table
            cursor.execute("PRAGMA table_info(replied_posts)")
            existing_columns = {col[1] for col in cursor.fetchall()}
            
            enterprise_columns_replied_posts = [
                ('operation_id', 'TEXT'),
                ('retry_count', 'INTEGER DEFAULT 0'),
                ('last_attempt_at', 'DATETIME'),
                ('status', 'TEXT DEFAULT "active"')
            ]
            
            for column_name, column_def in enterprise_columns_replied_posts:
                if column_name not in existing_columns:
                    cursor.execute(f"ALTER TABLE replied_posts ADD COLUMN {column_name} {column_def}")
                    logger.logger.info(f"✅ Added column '{column_name}' to replied_posts table")
            
            # Create performance indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_analysis_composite ON content_analysis(post_id, version, updated_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_operation_id ON replied_posts(operation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_status ON replied_posts(status)")
            
            conn.commit()
            logger.logger.info("✅ Phase 2.2: Existing table enhancements completed")
            
            return True
            
        except Exception as e:
            logger.log_error("Enhance Existing Tables", str(e))
            if conn:
                conn.rollback()
            return False    
    
    def get_enterprise_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive enterprise database statistics
        
        Returns:
            Dictionary with enterprise statistics
        """
        
        if not self.enterprise_mode:
            return {'error': 'Enterprise mode not enabled'}
        
        try:
            conn, cursor = self._get_connection()
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'enterprise_mode': True,
                'tables': {},
                'operations': {},
                'health': {},
                'configuration': {}
            }
            
            # Table statistics
            enterprise_tables = [
                'operation_audit', 'db_health_metrics', 'conflict_resolution_log',
                'system_config', 'retry_policies'
            ]
            
            for table in enterprise_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    result = cursor.fetchone()
                    stats['tables'][table] = result[0] if result else 0
                except Exception as e:
                    stats['tables'][table] = f"Error: {str(e)}"
            
            # Operation statistics (last 24 hours)
            try:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_operations,
                        SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                        SUM(CASE WHEN status = 'CONFLICT_RESOLVED' THEN 1 ELSE 0 END) as conflicts_resolved,
                        AVG(execution_time_ms) as avg_execution_time
                    FROM operation_audit 
                    WHERE created_at >= datetime('now', '-24 hours')
                """)
                
                result = cursor.fetchone()
                if result:
                    stats['operations'] = {
                        'total_operations_24h': result[0] or 0,
                        'successful_operations_24h': result[1] or 0,
                        'failed_operations_24h': result[2] or 0,
                        'conflicts_resolved_24h': result[3] or 0,
                        'avg_execution_time_ms': round(result[4] or 0, 2),
                        'success_rate_24h': round((result[1] or 0) / max(result[0] or 1, 1) * 100, 2)
                    }
                
            except Exception as e:
                stats['operations'] = {'error': str(e)}
            
            # Health metrics summary
            try:
                cursor.execute("""
                    SELECT 
                        table_name,
                        SUM(successful_operations) as total_success,
                        SUM(failed_operations) as total_failed,
                        SUM(constraint_violations) as total_violations,
                        AVG(avg_operation_time_ms) as avg_time
                    FROM db_health_metrics 
                    WHERE timestamp >= datetime('now', '-24 hours')
                    GROUP BY table_name
                    ORDER BY total_success + total_failed DESC
                """)
                
                health_by_table = {}
                for row in cursor.fetchall():
                    health_by_table[row[0]] = {
                        'successful_operations': row[1] or 0,
                        'failed_operations': row[2] or 0,
                        'constraint_violations': row[3] or 0,
                        'avg_operation_time_ms': round(row[4] or 0, 2)
                    }
                
                stats['health']['by_table'] = health_by_table
                
            except Exception as e:
                stats['health'] = {'error': str(e)}
            
            # Configuration summary
            try:
                cursor.execute("SELECT COUNT(*) FROM system_config")
                result = cursor.fetchone()
                config_count = result[0] if result else 0
                
                cursor.execute("""
                    SELECT config_key, config_value 
                    FROM system_config 
                    WHERE config_key IN ('enterprise_mode_enabled', 'audit_trail_enabled', 'health_monitoring_enabled')
                """)
                
                key_configs = dict(cursor.fetchall())
                
                stats['configuration'] = {
                    'total_configs': config_count,
                    'enterprise_enabled': key_configs.get('enterprise_mode_enabled', 'unknown'),
                    'audit_enabled': key_configs.get('audit_trail_enabled', 'unknown'),
                    'health_monitoring_enabled': key_configs.get('health_monitoring_enabled', 'unknown')
                }
                
            except Exception as e:
                stats['configuration'] = {'error': str(e)}
            
            return stats
            
        except Exception as e:
            logger.log_error("Get Enterprise Statistics", str(e))
            return {'error': str(e)}      

    def get_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get database health summary for the specified time period
        
        Args:
            hours: Number of hours of data to analyze (default: 24)
            
        Returns:
            Dictionary with health summary data
        """
        if not self.enterprise_mode:
            return {'error': 'Enterprise mode not enabled'}
        
        try:
            # Initialize health monitor if not exists
            if not hasattr(self, 'health_monitor') or self.health_monitor is None:
                self.health_monitor = DatabaseHealthMonitor(self)
            
            # Collect metrics
            metrics = self.health_monitor.collect_metrics(hours)
            
            # Detect patterns
            patterns = self.health_monitor.detect_patterns(hours)
            
            # Generate alerts
            alerts = self.health_monitor.generate_alerts(patterns)
            
            # Build summary
            summary = {
                'time_period_hours': hours,
                'generated_at': datetime.now().isoformat(),
                'enterprise_mode': True,
                'metrics': metrics,
                'patterns_detected': len(patterns),
                'alerts_generated': len(alerts),
                'patterns': patterns,
                'alerts': alerts,
                'health_status': 'HEALTHY'
            }
            
            # Determine overall health status
            if any(alert['alert_type'] == 'CRITICAL' for alert in alerts):
                summary['health_status'] = 'CRITICAL'
            elif any(alert['alert_type'] == 'WARNING' for alert in alerts):
                summary['health_status'] = 'WARNING'
            elif patterns:
                summary['health_status'] = 'MONITORING'
            
            return summary
            
        except Exception as e:
            logger.log_error("Get Health Summary", str(e))
            return {
                'error': str(e),
                'time_period_hours': hours,
                'generated_at': datetime.now().isoformat(),
                'enterprise_mode': True,
                'health_status': 'ERROR'
            }

    def get_operation_audit(self, hours: int = 24, table_name: Optional[str] = None,
                        status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent database operation audit records
        
        Args:
            hours: Number of hours of audit data to retrieve (default: 24)
            table_name: Filter by specific table name (optional)
            status: Filter by operation status (optional)
            limit: Maximum number of records to return (default: 100)
            
        Returns:
            List of audit records
        """
        if not self.enterprise_mode:
            return [{'error': 'Enterprise mode not enabled'}]
        
        try:
            conn, cursor = self._get_connection()
            
            # Build query with filters
            where_conditions = ["created_at >= datetime('now', '-' || ? || ' hours')"]
            params: List[Union[int, str]] = [hours]  # Fix: Explicit type annotation
            
            if table_name:
                where_conditions.append("table_name = ?")
                params.append(table_name)
            
            if status:
                where_conditions.append("status = ?")
                params.append(status)
            
            query = f"""
                SELECT 
                    operation_id, table_name, operation_type, record_key,
                    attempt_count, status, error_details, conflict_resolution_strategy,
                    execution_time_ms, created_at, completed_at, metadata
                FROM operation_audit 
                WHERE {' AND '.join(where_conditions)}
                ORDER BY created_at DESC 
                LIMIT ?
            """
            params.append(limit)
            
            cursor.execute(query, params)
            
            audit_records = []
            for row in cursor.fetchall():
                # Parse metadata JSON
                metadata = None
                if row[11]:  # metadata column
                    try:
                        metadata = json.loads(row[11])
                    except json.JSONDecodeError:
                        metadata = {'raw': row[11]}
                
                audit_records.append({
                    'operation_id': row[0],
                    'table_name': row[1],
                    'operation_type': row[2],
                    'record_key': row[3],
                    'attempt_count': row[4],
                    'status': row[5],
                    'error_details': row[6],
                    'conflict_resolution_strategy': row[7],
                    'execution_time_ms': row[8],
                    'created_at': row[9],
                    'completed_at': row[10],
                    'metadata': metadata
                })
            
            return audit_records
            
        except Exception as e:
            logger.log_error("Get Operation Audit", str(e))
            return [{'error': str(e)}]

    def get_conflict_log(self, hours: int = 24, table_name: Optional[str] = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent conflict resolution history
        
        Args:
            hours: Number of hours of conflict data to retrieve (default: 24)
            table_name: Filter by specific table name (optional)
            limit: Maximum number of records to return (default: 50)
            
        Returns:
            List of conflict resolution records
        """
        if not self.enterprise_mode:
            return [{'error': 'Enterprise mode not enabled'}]
        
        try:
            conn, cursor = self._get_connection()
            
            # Build query with filters
            where_conditions = ["timestamp >= datetime('now', '-' || ? || ' hours')"]
            params: List[Union[int, str]] = [hours]  # Fix: Explicit type annotation
            
            if table_name:
                where_conditions.append("table_name = ?")
                params.append(table_name)
            
            query = f"""
                SELECT 
                    timestamp, operation_id, table_name, record_id,
                    conflict_type, constraint_name, resolution_strategy,
                    resolution_successful, resolution_time_ms,
                    old_values, new_values, final_values
                FROM conflict_resolution_log 
                WHERE {' AND '.join(where_conditions)}
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            params.append(limit)
            
            cursor.execute(query, params)
            
            conflict_records = []
            for row in cursor.fetchall():
                # Parse JSON values
                old_values = None
                new_values = None
                final_values = None
                
                try:
                    if row[9]:  # old_values
                        old_values = json.loads(row[9])
                    if row[10]:  # new_values
                        new_values = json.loads(row[10])
                    if row[11]:  # final_values
                        final_values = json.loads(row[11])
                except json.JSONDecodeError as e:
                    logger.logger.debug(f"Could not parse conflict resolution JSON: {e}")
                
                conflict_records.append({
                    'timestamp': row[0],
                    'operation_id': row[1],
                    'table_name': row[2],
                    'record_id': row[3],
                    'conflict_type': row[4],
                    'constraint_name': row[5],
                    'resolution_strategy': row[6],
                    'resolution_successful': bool(row[7]),
                    'resolution_time_ms': row[8],
                    'old_values': old_values,
                    'new_values': new_values,
                    'final_values': final_values
                })
            
            return conflict_records
            
        except Exception as e:
            logger.log_error("Get Conflict Log", str(e))
            return [{'error': str(e)}]

    def strip_timezone(self, dt):
        """
        Ensure datetime is timezone-naive by converting to UTC and removing tzinfo

        Args:
            dt: Datetime object that might have timezone info

        Returns:
            Timezone-naive datetime object in UTC
        
        Raises:
            ValueError: If dt cannot be converted to datetime
            TypeError: If dt is not a datetime or string
        """
        from datetime import datetime, timezone

        # If it's not a datetime, try to convert
        if not isinstance(dt, datetime):
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                except ValueError:
                    formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']
                    for fmt in formats:
                        try:
                            dt = datetime.strptime(dt, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError(f"Unable to parse datetime string: {dt}")
            else:
                raise TypeError(f"Expected datetime or string, got {type(dt)}")

        # Handle timezone - dt is guaranteed to be datetime here
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

        return dt

    def standardize_timestamp_for_storage(self, timestamp):
        """
        Prepare timestamp for consistent database storage
        - Converts to UTC if timezone-aware
        - Strips timezone info 
        - Ensures consistent ISO format
    
        Args:
            timestamp: Datetime object or string to standardize
        
        Returns:
            Timestamp string formatted for storage
        """
        # First ensure we have a standardized datetime object
        timestamp = self.strip_timezone(timestamp)
    
        # Format with microsecond precision for accurate ordering
        return timestamp.strftime('%Y-%m-%d %H:%M:%S.%f') if isinstance(timestamp, datetime) else str(timestamp)

    def standardize_timestamp_for_query(self, target_time):
        """
        Prepare timestamp for database queries
        - Handles various input formats
        - Ensures consistent output format for SQLite
    
        Args:
            target_time: Datetime object or string to standardize
        
        Returns:
            Timestamp in format suitable for SQLite queries
        """
        # Use our comprehensive standardization method
        target_time = self.strip_timezone(target_time)
    
        # Return formatted for SQLite query
        return target_time.strftime('%Y-%m-%d %H:%M:%S.%f') if isinstance(target_time, datetime) else str(target_time)
    

    def _standardize_timestamp(self, timestamp):
        """
        Ensure timestamp is in a consistent format for database operations
    
        Args:
            timestamp: Datetime object or string
    
        Returns:
            Standardized datetime object (timezone-naive)
        """
        # Use our strip_timezone method which handles all the conversion logic
        return self.strip_timezone(timestamp)

    def _get_closest_historical_price(self, token, target_time, max_time_difference_hours=None):
        """
        Get the closest historical price to the target time

        Args:
            token: Token symbol
            target_time: Target datetime
            max_time_difference_hours: Maximum allowed time difference in hours (optional)
    
        Returns:
            dict with price, timestamp and time_difference or None if no suitable record found
        """
        from datetime import datetime

        conn, cursor = self._get_connection()

        try:
            # Standardize target time for querying
            target_time_std = self.standardize_timestamp_for_query(target_time)
        
            # Convert max_time_difference to seconds if provided
            max_time_difference = None
            if max_time_difference_hours is not None:
                max_time_difference = max_time_difference_hours * 3600
    
            # First try to find a price point BEFORE the target time
            cursor.execute("""
                SELECT price, timestamp
                FROM price_history
                WHERE token = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (token, target_time_std))
    
            before_result = cursor.fetchone()
    
            # Then try to find a price point AFTER the target time
            cursor.execute("""
                SELECT price, timestamp
                FROM price_history
                WHERE token = ? AND timestamp > ?
                ORDER BY timestamp ASC
                LIMIT 1
            """, (token, target_time_std))
    
            after_result = cursor.fetchone()
    
            # No data available at all
            if not before_result and not after_result:
                return None
        
            # Calculate time differences and find the closest point
            best_result = None
            smallest_diff = float('inf')
    
            # Parse target_time to datetime object for comparison
            target_time_dt = self.strip_timezone(target_time)

            # Initialize variables to avoid unbound errors
            before_time = after_time = None
            best_result = None
            smallest_diff = float('inf')

            if before_result and isinstance(target_time_dt, datetime):
                before_time = self.strip_timezone(before_result['timestamp'])
                if isinstance(before_time, datetime):
                    before_diff = abs((target_time_dt - before_time).total_seconds())

                    if max_time_difference is None or before_diff <= max_time_difference:
                        best_result = before_result
                        smallest_diff = before_diff

            if after_result and isinstance(target_time_dt, datetime):
                after_time = self.strip_timezone(after_result['timestamp'])
                if isinstance(after_time, datetime):
                    after_diff = abs((target_time_dt - after_time).total_seconds())

                    if (max_time_difference is None or after_diff <= max_time_difference) and after_diff < smallest_diff:
                        best_result = after_result
                        smallest_diff = after_diff

            if best_result:
                actual_time = self.strip_timezone(best_result['timestamp'])
                time_diff_hours = smallest_diff / 3600

                return {
                    'price': best_result['price'],
                    'timestamp': actual_time,
                    'time_difference_hours': time_diff_hours
                }
            else:
                return None
    
        except Exception as e:
            # Add appropriate logging
            return None
        
    def migrate_market_data_add_symbol_columns(self):
        """
        Migration: Add symbol and coin_id columns to market_data table
        Populates new columns from existing data using TokenMappingManager
        """
        conn, cursor = self._get_connection()
        
        try:
            logger.logger.info("🔄 Starting market_data table migration: adding symbol and coin_id columns")
            
            # Debug: Check actual table structure
            cursor.execute("PRAGMA table_info(market_data)")
            table_info = cursor.fetchall()
            existing_columns = [col[1] for col in table_info]
            
            logger.logger.debug(f"🔍 Current market_data table columns: {existing_columns}")
            
            symbol_exists = 'symbol' in existing_columns
            coin_id_exists = 'coin_id' in existing_columns
            
            if symbol_exists and coin_id_exists:
                logger.logger.info("✅ Migration already completed - symbol and coin_id columns exist")
                return True
            
            # Import TokenMappingManager for conversions
            try:
                from config import TokenMappingManager
                token_mapper = TokenMappingManager()
                logger.logger.info("✅ TokenMappingManager loaded successfully")
            except Exception as import_error:
                logger.logger.error(f"❌ Failed to import TokenMappingManager: {import_error}")
                return False
            
            # Step 1: Add missing columns
            if not symbol_exists:
                logger.logger.info("➕ Adding symbol column to market_data table")
                cursor.execute("ALTER TABLE market_data ADD COLUMN symbol TEXT")
            
            if not coin_id_exists:
                logger.logger.info("➕ Adding coin_id column to market_data table")
                cursor.execute("ALTER TABLE market_data ADD COLUMN coin_id TEXT")
            
            # Step 2: Get all existing records that need population
            cursor.execute("""
                SELECT id, token, chain 
                FROM market_data 
                WHERE symbol IS NULL OR coin_id IS NULL
                ORDER BY timestamp DESC
            """)
            
            records_to_update = cursor.fetchall()
            total_records = len(records_to_update)
            
            if total_records == 0:
                logger.logger.info("✅ No records need updating - migration complete")
                
                # Create indexes even if no data to update
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_coin_id ON market_data(coin_id)")
                
                conn.commit()
                return True
            
            logger.logger.info(f"📊 Found {total_records} records to update")
            
            # Step 3: Process records in batches
            successful_updates = 0
            mapping_stats = {
                'token_to_symbol': 0,
                'symbol_to_coin_id': 0,
                'chain_to_symbol': 0,
                'fallback_used': 0,
                'failed_mappings': 0
            }
            
            for i, record in enumerate(records_to_update):
                record_id = record['id']
                token_value = record['token']  # This should exist in your table
                chain_value = record['chain']  # This should exist in your table
                
                try:
                    # Determine symbol value
                    symbol = None
                    coin_id = None
                    
                    # Strategy 1: Use token value directly if it looks like a symbol
                    if token_value and len(token_value) <= 10 and token_value.replace('-', '').replace('_', '').isalnum():
                        # Clean token value and use as symbol
                        symbol = token_value.upper().replace('-', '').replace('_', '')
                        mapping_stats['token_to_symbol'] += 1
                        logger.logger.debug(f"✅ Direct token mapping: {token_value} -> {symbol}")
                    
                    # Strategy 2: Try to map chain value using TokenMappingManager
                    elif chain_value:
                        try:
                            mapped_symbol = token_mapper.database_name_to_symbol(chain_value)
                            if mapped_symbol and mapped_symbol != chain_value.upper():
                                symbol = mapped_symbol
                                mapping_stats['chain_to_symbol'] += 1
                                logger.logger.debug(f"✅ Chain mapping: {chain_value} -> {symbol}")
                            else:
                                # Fallback: use chain value as symbol
                                symbol = chain_value.upper()
                                mapping_stats['fallback_used'] += 1
                                logger.logger.debug(f"⚠️ Fallback: {chain_value} -> {symbol}")
                        except Exception as mapping_error:
                            # Final fallback: use chain value
                            symbol = chain_value.upper() if chain_value else 'UNKNOWN'
                            mapping_stats['fallback_used'] += 1
                            logger.logger.warning(f"⚠️ Mapping failed for {chain_value}, using fallback: {symbol}")
                    
                    # Strategy 3: Last resort fallback
                    if not symbol:
                        symbol = (token_value or chain_value or 'UNKNOWN').upper()
                        mapping_stats['fallback_used'] += 1
                        logger.logger.warning(f"⚠️ Last resort fallback: {symbol}")
                    
                    # Strategy 4: Get coin_id from symbol
                    if symbol and symbol != 'UNKNOWN':
                        try:
                            coin_id = token_mapper.symbol_to_coingecko_id(symbol)
                            if coin_id:
                                mapping_stats['symbol_to_coin_id'] += 1
                                logger.logger.debug(f"✅ Symbol to coin_id: {symbol} -> {coin_id}")
                        except Exception as coin_id_error:
                            logger.logger.debug(f"⚠️ Could not map {symbol} to coin_id: {coin_id_error}")
                    
                    # Update the record
                    cursor.execute("""
                        UPDATE market_data 
                        SET symbol = ?, coin_id = ?
                        WHERE id = ?
                    """, (symbol, coin_id, record_id))
                    
                    successful_updates += 1
                    
                    # Log progress every 100 records
                    if (i + 1) % 100 == 0:
                        logger.logger.info(f"📈 Progress: {i + 1}/{total_records} records processed")
                    
                except Exception as record_error:
                    mapping_stats['failed_mappings'] += 1
                    logger.logger.error(f"❌ Failed to process record {record_id}: {record_error}")
                    continue
            
            # Step 4: Create indexes for performance
            logger.logger.info("🔧 Creating indexes on new columns")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_coin_id ON market_data(coin_id)")
            
            # Step 5: Commit all changes
            conn.commit()
            
            # Step 6: Generate migration report
            logger.logger.info("📊 Migration completed successfully!")
            logger.logger.info(f"✅ Records updated: {successful_updates}/{total_records}")
            logger.logger.info(f"📈 Mapping statistics:")
            for stat_name, count in mapping_stats.items():
                logger.logger.info(f"   - {stat_name}: {count}")
            
            # Step 7: Verify migration results
            cursor.execute("SELECT COUNT(*) as total FROM market_data")
            total_count = cursor.fetchone()['total']
            
            cursor.execute("SELECT COUNT(*) as with_symbol FROM market_data WHERE symbol IS NOT NULL")
            with_symbol = cursor.fetchone()['with_symbol']
            
            cursor.execute("SELECT COUNT(*) as with_coin_id FROM market_data WHERE coin_id IS NOT NULL")
            with_coin_id = cursor.fetchone()['with_coin_id']
            
            logger.logger.info(f"🔍 Verification results:")
            logger.logger.info(f"   - Total records: {total_count}")
            logger.logger.info(f"   - Records with symbol: {with_symbol} ({(with_symbol/total_count)*100:.1f}%)")
            logger.logger.info(f"   - Records with coin_id: {with_coin_id} ({(with_coin_id/total_count)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.logger.error(f"❌ Migration failed: {str(e)}")
            logger.logger.debug(f"Migration error traceback: {traceback.format_exc()}")
            conn.rollback()
            return False    
        
    def _ensure_reply_restrictions_table_exists(self):
        """
        Ensure the reply_restrictions table exists in the database - Enterprise version
        
        ENHANCED FEATURES:
        - Improved error handling with specific error types
        - Enhanced logging for table creation process
        - Better transaction management
        - Maintains full backward compatibility
        """
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            # Check if table already exists for logging
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reply_restrictions'")
            table_exists = cursor.fetchone() is not None
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reply_restrictions (
                    post_id TEXT PRIMARY KEY,
                    post_url TEXT,
                    author_handle TEXT,
                    restriction_reason TEXT,
                    detected_at DATETIME NOT NULL,
                    UNIQUE(post_id)
                )
            """)
            
            # Create index for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reply_restrictions_post_id ON reply_restrictions(post_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reply_restrictions_author ON reply_restrictions(author_handle)")
            
            conn.commit()
            
            # 🆕 ENTERPRISE: Enhanced logging based on operation
            if table_exists:
                logger.logger.debug("✅ Reply restrictions table verified (already exists)")
            else:
                logger.logger.info("✅ Reply restrictions table created successfully")
            
            return True
            
        except sqlite3.OperationalError as e:
            # 🆕 ENTERPRISE: Handle database lock/operational errors
            logger.log_error("Reply Restrictions Table - Database Lock", str(e))
            logger.logger.error(f"❌ Database operation failed creating reply restrictions table: {e}")
            if conn:
                conn.rollback()
            return False
            
        except sqlite3.DatabaseError as e:
            # 🆕 ENTERPRISE: Handle database-specific errors
            logger.log_error("Reply Restrictions Table - Database Error", str(e))
            logger.logger.error(f"❌ Database error creating reply restrictions table: {e}")
            if conn:
                conn.rollback()
            return False
            
        except Exception as e:
            # General error handling
            logger.log_error("Ensure Reply Restrictions Table", str(e))
            logger.logger.error(f"❌ Failed to create/verify reply restrictions table: {e}")
            if conn:
                conn.rollback()
            return False

    def store_reply_restriction(self, post_id: str, post_url: Optional[str] = None, 
                            author_handle: Optional[str] = None,
                            restriction_reason: Optional[str] = None, 
                            timestamp: Optional[datetime] = None):
        """
        Store a reply restriction for a post to avoid future attempts - Enterprise version
        
        ENHANCED FEATURES:
        - Resolves UNIQUE constraint violations using UPSERT pattern
        - Handles duplicate restrictions gracefully (updates existing restrictions)
        - Enhanced logging for conflict detection
        - Improved error classification and handling
        - Maintains full backward compatibility
        
        Args:
            post_id: The ID of the post with reply restrictions
            post_url: URL of the post (optional)
            author_handle: Author of the post (optional)
            restriction_reason: Reason for the restriction (optional)
            timestamp: When restriction was detected (defaults to current time)

        Returns:
            bool: True if stored successfully, False otherwise
        """
        conn = None
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            conn, cursor = self._get_connection()

            # First check if we need to create the table
            self._ensure_reply_restrictions_table_exists()

            # 🆕 ENTERPRISE: Check if restriction already exists for logging
            cursor.execute("SELECT COUNT(*) FROM reply_restrictions WHERE post_id = ?", (post_id,))
            restriction_exists = cursor.fetchone()[0] > 0

            # 🆕 ENTERPRISE: Use INSERT OR REPLACE to handle duplicates gracefully
            cursor.execute("""
                INSERT OR REPLACE INTO reply_restrictions (
                    post_id, post_url, author_handle, restriction_reason, detected_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                post_id,
                post_url,
                author_handle,
                restriction_reason,
                timestamp
            ))

            conn.commit()
            
            # 🆕 ENTERPRISE: Enhanced logging to track updates vs new restrictions
            if restriction_exists:
                logger.logger.info(f"🔄 Updated existing reply restriction for post {post_id} (reason: {restriction_reason})")
            else:
                logger.logger.debug(f"✅ Stored new reply restriction for post {post_id} (reason: {restriction_reason})")
            
            return True

        except sqlite3.IntegrityError as e:
            # 🆕 ENTERPRISE: Specific handling for integrity constraint errors
            if conn:
                conn.rollback()
            logger.log_error("Reply Restriction Storage Integrity Error", str(e))
            logger.logger.error(f"❌ Integrity constraint failed storing restriction for post {post_id}: {e}")
            return False

        except sqlite3.OperationalError as e:
            # 🆕 ENTERPRISE: Handle database lock/operational errors
            if conn:
                conn.rollback()
            logger.log_error("Reply Restriction Storage Database Lock", str(e))
            logger.logger.error(f"❌ Database operation failed storing restriction for post {post_id}: {e}")
            return False

        except Exception as e:
            # Existing general error handling
            if conn:
                conn.rollback()
            logger.log_error("Store Reply Restriction", str(e))
            logger.logger.error(f"❌ Failed to store reply restriction for post {post_id}: {e}")
            return False

    def check_reply_restriction(self, post_id: str) -> bool:
        """
        Check if a post has known reply restrictions - Enterprise version
        
        ENHANCED FEATURES:
        - Improved error handling with specific error types
        - Enhanced logging for debugging
        - Better performance with optimized query
        - Maintains full backward compatibility
        
        Args:
            post_id: The ID of the post to check
            
        Returns:
            bool: True if post has reply restrictions, False otherwise
        """
        try:
            conn, cursor = self._get_connection()
            
            # 🆕 ENTERPRISE: Optimized query - just check existence, don't fetch data
            cursor.execute("""
                SELECT 1 FROM reply_restrictions 
                WHERE post_id = ? 
                LIMIT 1
            """, (post_id,))
            
            result = cursor.fetchone()
            has_restriction = result is not None
            
            # 🆕 ENTERPRISE: Enhanced logging for debugging
            if has_restriction:
                logger.logger.debug(f"🚫 Post {post_id} has reply restrictions")
            else:
                logger.logger.debug(f"✅ Post {post_id} has no reply restrictions")
            
            return has_restriction
            
        except sqlite3.OperationalError as e:
            # 🆕 ENTERPRISE: Handle database lock/operational errors
            logger.log_error("Check Reply Restriction Database Lock", str(e))
            logger.logger.error(f"❌ Database operation failed checking restriction for post {post_id}: {e}")
            return False
            
        except sqlite3.DatabaseError as e:
            # 🆕 ENTERPRISE: Handle database-specific errors
            logger.log_error("Check Reply Restriction Database Error", str(e))
            logger.logger.error(f"❌ Database error checking restriction for post {post_id}: {e}")
            return False
            
        except Exception as e:
            # General error handling
            logger.log_error("Check Reply Restriction", str(e))
            logger.logger.error(f"❌ Failed to check reply restriction for post {post_id}: {e}")
            return False

    def get_reply_restrictions_stats(self) -> Dict[str, Any]:
        """
        Get statistics about reply restrictions - Enterprise version
        
        ENHANCED FEATURES:
        - Fixed bug in total_restrictions calculation
        - Improved error handling with specific error types
        - Enhanced data validation and error recovery
        - Better performance with optimized queries
        - More comprehensive statistics
        - Maintains full backward compatibility
        
        Returns:
            Dictionary with restriction statistics
        """
        try:
            conn, cursor = self._get_connection()
            
            # 🆕 ENTERPRISE: Fixed bug - proper fetchone() handling
            cursor.execute("SELECT COUNT(*) FROM reply_restrictions")
            result = cursor.fetchone()
            total_restrictions = result[0] if result else 0
            
            # 🆕 ENTERPRISE: Enhanced error handling for individual queries
            restrictions_by_reason = {}
            try:
                cursor.execute("""
                    SELECT restriction_reason, COUNT(*) 
                    FROM reply_restrictions 
                    WHERE restriction_reason IS NOT NULL
                    GROUP BY restriction_reason
                    ORDER BY COUNT(*) DESC
                """)
                restrictions_by_reason = dict(cursor.fetchall())
            except Exception as e:
                logger.logger.warning(f"Could not fetch restrictions by reason: {e}")
                restrictions_by_reason = {}
            
            # Recent restrictions (last 7 days) with proper error handling
            recent_restrictions = 0
            try:
                week_ago = datetime.now() - timedelta(days=7)
                cursor.execute("""
                    SELECT COUNT(*) FROM reply_restrictions 
                    WHERE detected_at > ?
                """, (week_ago,))
                result = cursor.fetchone()
                recent_restrictions = result[0] if result else 0
            except Exception as e:
                logger.logger.warning(f"Could not fetch recent restrictions: {e}")
                recent_restrictions = 0
            
            # 🆕 ENTERPRISE: Additional useful statistics
            restrictions_by_author = {}
            try:
                cursor.execute("""
                    SELECT author_handle, COUNT(*) 
                    FROM reply_restrictions 
                    WHERE author_handle IS NOT NULL
                    GROUP BY author_handle
                    ORDER BY COUNT(*) DESC
                    LIMIT 10
                """)
                restrictions_by_author = dict(cursor.fetchall())
            except Exception as e:
                logger.logger.warning(f"Could not fetch restrictions by author: {e}")
                restrictions_by_author = {}
            
            # 🆕 ENTERPRISE: Daily restrictions trend (last 30 days)
            daily_restrictions = {}
            try:
                thirty_days_ago = datetime.now() - timedelta(days=30)
                cursor.execute("""
                    SELECT DATE(detected_at) as restriction_date, COUNT(*) 
                    FROM reply_restrictions 
                    WHERE detected_at > ?
                    GROUP BY DATE(detected_at)
                    ORDER BY restriction_date DESC
                    LIMIT 30
                """, (thirty_days_ago,))
                daily_restrictions = dict(cursor.fetchall())
            except Exception as e:
                logger.logger.warning(f"Could not fetch daily restrictions trend: {e}")
                daily_restrictions = {}
            
            # 🆕 ENTERPRISE: Build comprehensive statistics response
            stats = {
                'total_restrictions': total_restrictions,
                'restrictions_by_reason': restrictions_by_reason,
                'recent_restrictions': recent_restrictions,
                'restrictions_by_author': restrictions_by_author,
                'daily_restrictions_last_30_days': daily_restrictions,
                'timestamp': datetime.now().isoformat(),
                'data_quality': {
                    'has_reasons': len(restrictions_by_reason) > 0,
                    'has_authors': len(restrictions_by_author) > 0,
                    'has_recent_data': recent_restrictions > 0
                }
            }
            
            # 🆕 ENTERPRISE: Log statistics summary
            logger.logger.debug(f"📊 Reply restrictions stats: {total_restrictions} total, {recent_restrictions} recent")
            
            return stats
            
        except sqlite3.OperationalError as e:
            # 🆕 ENTERPRISE: Handle database lock/operational errors
            logger.log_error("Reply Restrictions Stats Database Lock", str(e))
            logger.logger.error(f"❌ Database operation failed getting reply restrictions stats: {e}")
            return {
                'error': f'Database operation failed: {str(e)}',
                'error_type': 'OperationalError',
                'timestamp': datetime.now().isoformat()
            }
            
        except sqlite3.DatabaseError as e:
            # 🆕 ENTERPRISE: Handle database-specific errors
            logger.log_error("Reply Restrictions Stats Database Error", str(e))
            logger.logger.error(f"❌ Database error getting reply restrictions stats: {e}")
            return {
                'error': f'Database error: {str(e)}',
                'error_type': 'DatabaseError',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            # General error handling
            logger.log_error("Get Reply Restrictions Stats", str(e))
            logger.logger.error(f"❌ Failed to get reply restrictions stats: {e}")
            return {
                'error': str(e),
                'error_type': 'GeneralError',
                'timestamp': datetime.now().isoformat()
            }   

    def add_enterprise_operation_methods(self, cls):
        """
        Add enterprise operation methods to CryptoDatabase class
        This should be called after the class definition
        """
        
        def get_operation_audit(self, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
            """Get recent operation audit logs"""
            if not self.enterprise_mode:
                return []
                
            try:
                conn, cursor = self._get_connection()
                
                cursor.execute("""
                    SELECT * FROM operation_audit
                    WHERE created_at >= datetime('now', '-' || ? || ' hours')
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (hours, limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
            except Exception as e:
                logger.log_error("Get Operation Audit", str(e))
                return []
        
        def get_conflict_log(self, hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
            """Get recent conflict resolution logs"""
            if not self.enterprise_mode:
                return []
                
            try:
                conn, cursor = self._get_connection()
                
                cursor.execute("""
                    SELECT * FROM conflict_resolution_log
                    WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (hours, limit))
                
                results = [dict(row) for row in cursor.fetchall()]
                
                # Parse JSON fields
                for result in results:
                    if result.get('old_values'):
                        result['old_values'] = json.loads(result['old_values'])
                    if result.get('new_values'):
                        result['new_values'] = json.loads(result['new_values'])
                    if result.get('final_values'):
                        result['final_values'] = json.loads(result['final_values'])
                
                return results
                
            except Exception as e:
                logger.log_error("Get Conflict Log", str(e))
                return []
        
        def get_operation_statistics(self, hours: int = 24) -> Dict[str, Any]:
            """Get operation statistics for the specified time period"""
            if not self.enterprise_mode:
                return {'error': 'Enterprise mode not enabled'}
                
            try:
                conn, cursor = self._get_connection()
                
                # Overall statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_operations,
                        SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                        SUM(CASE WHEN status = 'CONFLICT_RESOLVED' THEN 1 ELSE 0 END) as conflicts_resolved,
                        AVG(execution_time_ms) as avg_execution_time,
                        MAX(execution_time_ms) as max_execution_time
                    FROM operation_audit
                    WHERE created_at >= datetime('now', '-' || ? || ' hours')
                """, (hours,))
                
                stats = cursor.fetchone()
                
                if not stats:
                    return {'error': 'No operation data found'}
                
                result = dict(stats)
                
                # Calculate success rate
                total = result['total_operations'] or 0
                successful = result['successful'] or 0
                result['success_rate'] = (successful / total * 100) if total > 0 else 0
                
                # Statistics by table
                cursor.execute("""
                    SELECT 
                        table_name,
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                        AVG(execution_time_ms) as avg_time
                    FROM operation_audit
                    WHERE created_at >= datetime('now', '-' || ? || ' hours')
                    GROUP BY table_name
                    ORDER BY total DESC
                """, (hours,))
                
                by_table = []
                for row in cursor.fetchall():
                    table_stats = dict(row)
                    table_total = table_stats['total'] or 0
                    table_successful = table_stats['successful'] or 0
                    table_stats['success_rate'] = (table_successful / table_total * 100) if table_total > 0 else 0
                    by_table.append(table_stats)
                
                result['by_table'] = by_table
                
                return result
                
            except Exception as e:
                logger.log_error("Get Operation Statistics", str(e))
                return {'error': str(e)}
        
        # Add methods to the class
        cls.get_operation_audit = get_operation_audit
        cls.get_conflict_log = get_conflict_log  
        cls.get_operation_statistics = get_operation_statistics
        
        return cls     
        
    def create_billionaire_tracking_tables(self):
        """
        Create tables for billionaire wealth tracking system
    
        This method adds the necessary tables to track:
        - Portfolio performance and milestones
        - Trade history for wealth generation
        - Risk management and position sizing
        - Performance metrics and analytics
        """
        conn, cursor = self._get_connection()
    
        try:
            # Billionaire Portfolio Tracking Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_portfolio_value REAL NOT NULL,
                    initial_capital REAL NOT NULL,
                    total_return_pct REAL,
                    daily_return_pct REAL,
                    max_drawdown_pct REAL,
                    positions_count INTEGER,
                    risk_level TEXT,
                    wealth_milestone TEXT,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
            # Billionaire Trade History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    action TEXT NOT NULL,  -- 'BUY', 'SELL', 'HOLD'
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    total_value REAL NOT NULL,
                    portfolio_allocation_pct REAL,
                    position_size_pct REAL,
                    risk_score REAL,
                    profit_loss REAL,
                    profit_loss_pct REAL,
                    trade_reason TEXT,
                    technical_signals TEXT,  -- JSON string of signals
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
            # Billionaire Milestones Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_milestones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    milestone_type TEXT NOT NULL,  -- 'first_million', 'ten_million', etc.
                    milestone_value REAL NOT NULL,
                    portfolio_value REAL NOT NULL,
                    time_to_achieve INTEGER,  -- days from start
                    strategy_used TEXT,
                    performance_metrics TEXT,  -- JSON string
                    celebration_notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
            # Billionaire Performance Metrics Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    portfolio_value REAL NOT NULL,
                    daily_return REAL,
                    cumulative_return REAL,
                    volatility REAL,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    max_consecutive_wins INTEGER,
                    max_consecutive_losses INTEGER,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    avg_win REAL,
                    avg_loss REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
        
            # Billionaire Risk Management Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_risk_management (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    portfolio_value REAL NOT NULL,
                    total_risk_exposure REAL,
                    max_position_size_pct REAL,
                    risk_per_trade_pct REAL,
                    correlation_risk REAL,
                    leverage_used REAL,
                    var_95 REAL,  -- Value at Risk 95%
                    expected_shortfall REAL,
                    risk_adjusted_return REAL,
                    risk_level TEXT,  -- 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
            # Billionaire Wealth Targets Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billionaire_wealth_targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_name TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    current_progress REAL,
                    progress_pct REAL,
                    estimated_time_to_achieve INTEGER,  -- days
                    strategy_focus TEXT,
                    priority_level INTEGER,
                    is_achieved BOOLEAN DEFAULT FALSE,
                    achieved_date DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
            # Insert default wealth targets if table is empty
            cursor.execute("SELECT COUNT(*) FROM billionaire_wealth_targets")
            if cursor.fetchone()[0] == 0:
                default_targets = [
                    ('First Million', 1_000_000, 1),
                    ('Ten Million', 10_000_000, 2),
                    ('Hundred Million', 100_000_000, 3),
                    ('Quarter Billion', 250_000_000, 4),
                    ('Half Billion', 500_000_000, 5),
                    ('First Billion', 1_000_000_000, 6),
                    ('Five Billion', 5_000_000_000, 7),
                    ('Ten Billion', 10_000_000_000, 8),
                    ('Ultimate Target', 50_000_000_000, 9)
                ]
            
                for target_name, target_value, priority in default_targets:
                    cursor.execute("""
                        INSERT INTO billionaire_wealth_targets 
                        (target_name, target_value, current_progress, progress_pct, priority_level)
                        VALUES (?, ?, 0, 0, ?)
                    """, (target_name, target_value, priority))
        
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_billionaire_portfolio_timestamp 
                ON billionaire_portfolio(timestamp)
            """)
        
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_billionaire_trades_timestamp_token 
                ON billionaire_trades(timestamp, token)
            """)
        
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_billionaire_performance_date 
                ON billionaire_performance(date)
            """)
        
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_billionaire_milestones_type 
                ON billionaire_milestones(milestone_type)
            """)
        
            conn.commit()
            logger.logger.info("✅ Billionaire tracking tables created successfully")
        
            # Log the tables created
            tables_created = [
                "billionaire_portfolio",
                "billionaire_trades", 
                "billionaire_milestones",
                "billionaire_performance",
                "billionaire_risk_management",
                "billionaire_wealth_targets"
            ]
        
            logger.logger.info(f"💰 Created {len(tables_created)} billionaire tracking tables:")
            for table in tables_created:
                logger.logger.info(f"   📊 {table}")
        
            return True
        
        except Exception as e:
            logger.log_error("Create Billionaire Tracking Tables", str(e))
            conn.rollback()
            return False


    # Additional helper methods for the billionaire system

    def store_billionaire_trade(self, trade_data: Dict[str, Any]) -> Optional[int]:
        """Store a billionaire trade in the database"""
        conn, cursor = self._get_connection()
    
        try:
            cursor.execute("""
                INSERT INTO billionaire_trades (
                    timestamp, token, action, quantity, price, total_value,
                    portfolio_allocation_pct, position_size_pct, risk_score,
                    profit_loss, profit_loss_pct, trade_reason, technical_signals
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get('timestamp', datetime.now()),
                trade_data.get('token'),
                trade_data.get('action'),
                trade_data.get('quantity'),
                trade_data.get('price'),
                trade_data.get('total_value'),
                trade_data.get('portfolio_allocation_pct'),
                trade_data.get('position_size_pct'),
                trade_data.get('risk_score'),
                trade_data.get('profit_loss'),
                trade_data.get('profit_loss_pct'),
                trade_data.get('trade_reason'),
                json.dumps(trade_data.get('technical_signals', {}))
            ))
        
            trade_id = cursor.lastrowid
            conn.commit()
            logger.logger.info(f"💰 Billionaire trade stored: {trade_data.get('action')} {trade_data.get('token')}")
            return trade_id
        
        except Exception as e:
            logger.log_error("Store Billionaire Trade", str(e))
            conn.rollback()
            return None


    def update_billionaire_portfolio(self, portfolio_data: Dict[str, Any]) -> bool:
        """Update billionaire portfolio tracking"""
        conn, cursor = self._get_connection()
    
        try:
            cursor.execute("""
                INSERT INTO billionaire_portfolio (
                    timestamp, total_portfolio_value, initial_capital,
                    total_return_pct, daily_return_pct, max_drawdown_pct,
                    positions_count, risk_level, wealth_milestone, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                portfolio_data.get('timestamp', datetime.now()),
                portfolio_data.get('total_portfolio_value'),
                portfolio_data.get('initial_capital'),
                portfolio_data.get('total_return_pct'),
                portfolio_data.get('daily_return_pct'),
                portfolio_data.get('max_drawdown_pct'),
                portfolio_data.get('positions_count'),
                portfolio_data.get('risk_level'),
                portfolio_data.get('wealth_milestone'),
                portfolio_data.get('notes')
            ))
        
            conn.commit()
            logger.logger.info(f"💰 Portfolio updated: ${portfolio_data.get('total_portfolio_value', 0):,.2f}")
            return True
        
        except Exception as e:
            logger.log_error("Update Billionaire Portfolio", str(e))
            conn.rollback()
            return False


    def record_billionaire_milestone(self, milestone_data: Dict[str, Any]) -> bool:
        """Record a billionaire wealth milestone achievement"""
        conn, cursor = self._get_connection()
    
        try:
            cursor.execute("""
                INSERT INTO billionaire_milestones (
                    timestamp, milestone_type, milestone_value, portfolio_value,
                    time_to_achieve, strategy_used, performance_metrics, celebration_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                milestone_data.get('timestamp', datetime.now()),
                milestone_data.get('milestone_type'),
                milestone_data.get('milestone_value'),
                milestone_data.get('portfolio_value'),
                milestone_data.get('time_to_achieve'),
                milestone_data.get('strategy_used'),
                json.dumps(milestone_data.get('performance_metrics', {})),
                milestone_data.get('celebration_notes')
            ))
        
            # Update the wealth targets table
            cursor.execute("""
                UPDATE billionaire_wealth_targets 
                SET is_achieved = TRUE, achieved_date = ?, current_progress = ?, progress_pct = 100
                WHERE target_name = ?
            """, (
                datetime.now(),
                milestone_data.get('milestone_value'),
                milestone_data.get('milestone_type')
            ))
        
            conn.commit()
            logger.logger.info(f"🎉 MILESTONE ACHIEVED: {milestone_data.get('milestone_type')} - ${milestone_data.get('milestone_value'):,.2f}")
            return True
        
        except Exception as e:
            logger.log_error("Record Billionaire Milestone", str(e))
            conn.rollback()
            return False    

    def add_price_history_table(self):
        """
        Add price_history table for tracking historical price data
        This allows us to calculate price changes ourselves instead of relying on external APIs
        """
        conn, cursor = self._get_connection()
        try:
            # Create price_history table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price REAL NOT NULL,
                    volume REAL,
                    market_cap REAL,
                    total_supply REAL,
                    circulating_supply REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(token, timestamp)
                )
            """)
        
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_token ON price_history(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_timestamp ON price_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_token_timestamp ON price_history(token, timestamp)")
        
            conn.commit()
            logger.logger.info("Added price_history table to database")
            return True
        except Exception as e:
            logger.log_error("Add Price History Table", str(e))
            conn.rollback()
            return False
        
    def _ensure_critical_tables_exist(self):
        """
        Ensure critical tables exist - NUMBA thread safe version
        Called before database operations that might run in NUMBA worker threads
        """
        conn, cursor = self._get_connection()
        try:
            # Check and create smart_money_indicators table if needed
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS smart_money_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    chain TEXT NOT NULL,
                    volume_z_score REAL,
                    price_volume_divergence BOOLEAN,
                    stealth_accumulation BOOLEAN,
                    abnormal_volume BOOLEAN,
                    volume_vs_hourly_avg REAL,
                    volume_vs_daily_avg REAL,
                    volume_cluster_detected BOOLEAN,
                    unusual_trading_hours JSON,
                    raw_data JSON
                )
            """)
            
            # Check and create token_market_comparison table if needed  
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_market_comparison (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    vs_market_avg_change REAL,
                    vs_market_volume_growth REAL,
                    outperforming_market BOOLEAN,
                    correlations JSON
                )
            """)
            
            conn.commit()
            logger.logger.debug("✅ Critical tables validated/created successfully")
            
        except Exception as e:
            logger.logger.error(f"❌ Error ensuring critical tables exist: {e}")
            if conn:
                conn.rollback()
            raise    

    def store_price_history(self, token: str, price: float, volume: Optional[float] = None, 
                            market_cap: Optional[float] = None, total_supply: Optional[float] = None, 
                            circulating_supply: Optional[float] = None, timestamp: Optional[datetime] = None):
        """
        Store price data in the price_history table with enhanced timestamp handling
    
        Args:
            token: Token symbol
            price: Current price
            volume: Trading volume (optional)
            market_cap: Market capitalization (optional)
            total_supply: Total supply (optional)
            circulating_supply: Circulating supply (optional)
            timestamp: Optional timestamp (defaults to current time)
        
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            # Input validation with logging
            logger.logger.debug(f"store_price_history called for token: {token}")
        
            if not token:
                logger.logger.error("store_price_history called with empty token")
                return False
        
            if not isinstance(price, (int, float)) or price <= 0:
                logger.logger.error(f"store_price_history called with invalid price for {token}: {price}")
                return False
        
            logger.logger.debug(f"Price value for {token}: {price}")
        
            # Handle timestamp with standardization
            if timestamp is None:
                timestamp = datetime.now()
                logger.logger.debug(f"Using current time for {token}: {timestamp}")
        
            # Standardize timestamp for consistent storage
            std_timestamp = self._standardize_timestamp(timestamp)
            logger.logger.debug(f"Standardized timestamp for {token}: {std_timestamp}")
        
            conn, cursor = self._get_connection()
        
            # Make sure the table exists
            logger.logger.debug(f"Ensuring price_history table exists for {token}")
            self._ensure_price_history_table_exists()
        
            # Check if we already have data for this token at this timestamp
            cursor.execute("""
                SELECT id, price FROM price_history
                WHERE token = ? AND timestamp = ?
            """, (token, std_timestamp))
        
            existing_record = cursor.fetchone()
            if existing_record:
                logger.logger.debug(f"Found existing record for {token} at {std_timestamp}: id={existing_record['id']}, price={existing_record['price']}")
            else:
                logger.logger.debug(f"No existing record for {token} at {std_timestamp}")
        
            # Insert or replace price data
            logger.logger.debug(f"Inserting/replacing price data for {token}")
            cursor.execute("""
                INSERT OR REPLACE INTO price_history (
                    token, timestamp, price, volume, market_cap, 
                    total_supply, circulating_supply
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                token,
                std_timestamp,
                price,
                volume,
                market_cap,
                total_supply,
                circulating_supply
            ))
        
            conn.commit()
            logger.logger.debug(f"Successfully committed price history for {token}")
            return True
        
        except Exception as e:
            conn = None
            try:
                conn, cursor = self._get_connection()
            except:
                pass
            
            logger.log_error(f"Store Price History - {token}", str(e))
            logger.logger.error(f"Error in store_price_history for {token}: {str(e)}")
            logger.logger.debug(f"Traceback: {traceback.format_exc()}")
            if conn:
                conn.rollback()
            return False

    def _ensure_price_history_table_exists(self):
        """Ensure price_history table exists in the database with enhanced logging"""
        conn, cursor = self._get_connection()
        try:
            logger.logger.debug("_ensure_price_history_table_exists called")
        
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_history'")
            table_exists = cursor.fetchone() is not None
        
            if table_exists:
                logger.logger.debug("price_history table already exists")
            
                # Additional validation: check table structure
                cursor.execute("PRAGMA table_info(price_history)")
                columns = cursor.fetchall()
                column_names = [column[1] for column in columns]
            
                logger.logger.debug(f"price_history table has columns: {column_names}")
            
                # Check for required columns
                required_columns = ['id', 'token', 'timestamp', 'price', 'volume', 'market_cap', 
                                  'total_supply', 'circulating_supply', 'created_at']
            
                missing_columns = [col for col in required_columns if col not in column_names]
            
                if missing_columns:
                    logger.logger.warning(f"price_history table is missing columns: {missing_columns}")
                    # We could add code here to alter the table and add missing columns if needed
                else:
                    logger.logger.debug("price_history table structure is valid")
                
                # Check table indices
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='price_history'")
                indices = [idx[0] for idx in cursor.fetchall()]
            
                logger.logger.debug(f"price_history table has indices: {indices}")
            
                required_indices = ['idx_price_history_token', 'idx_price_history_timestamp', 
                              'idx_price_history_token_timestamp']
            
                missing_indices = [idx for idx in required_indices if idx not in indices]
            
                if missing_indices:
                    logger.logger.warning(f"price_history table is missing indices: {missing_indices}")
                
                    # Create missing indices
                    for idx in missing_indices:
                        try:
                            if idx == 'idx_price_history_token':
                                logger.logger.debug("Creating index: idx_price_history_token")
                                cursor.execute("CREATE INDEX idx_price_history_token ON price_history(token)")
                            elif idx == 'idx_price_history_timestamp':
                                logger.logger.debug("Creating index: idx_price_history_timestamp")
                                cursor.execute("CREATE INDEX idx_price_history_timestamp ON price_history(timestamp)")
                            elif idx == 'idx_price_history_token_timestamp':
                                logger.logger.debug("Creating index: idx_price_history_token_timestamp")
                                cursor.execute("CREATE INDEX idx_price_history_token_timestamp ON price_history(token, timestamp)")
                        except Exception as idx_error:
                            logger.logger.error(f"Error creating index {idx}: {str(idx_error)}")
                            # Continue with other indices even if one fails
                
                    conn.commit()
                    logger.logger.debug("Created missing indices for price_history table")
                else:
                    logger.logger.debug("All required indices exist for price_history table")
                
                # Query table statistics for diagnostics
                cursor.execute("SELECT COUNT(*) as count FROM price_history")
                row_count = cursor.fetchone()['count']
            
                logger.logger.debug(f"price_history table has {row_count} rows")
            
                if row_count > 0:
                    # Query time range of data
                    cursor.execute("""
                        SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time
                        FROM price_history
                    """)
                
                    time_result = cursor.fetchone()
                    if time_result and time_result['min_time'] and time_result['max_time']:
                        min_time = time_result['min_time']
                        max_time = time_result['max_time']
                        logger.logger.debug(f"price_history table has data from {min_time} to {max_time}")
                    
                    # Query distinct tokens
                    cursor.execute("""
                        SELECT token, COUNT(*) as count
                        FROM price_history
                        GROUP BY token
                        ORDER BY count DESC
                    """)
                
                    token_counts = cursor.fetchall()
                    if token_counts:
                        token_info = ", ".join([f"{row['token']}({row['count']})" for row in token_counts[:10]])
                        if len(token_counts) > 10:
                            token_info += f", ... and {len(token_counts)-10} more"
                        logger.logger.debug(f"price_history tokens: {token_info}")
            else:
                logger.logger.info("price_history table does not exist, creating it")
            
                # Create price_history table
                cursor.execute("""
                    CREATE TABLE price_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        token TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        price REAL NOT NULL,
                        volume REAL,
                        market_cap REAL,
                        total_supply REAL,
                        circulating_supply REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(token, timestamp)
                    )
                """)
        
                # Create indexes for better query performance
                logger.logger.debug("Creating indices for new price_history table")
                cursor.execute("CREATE INDEX idx_price_history_token ON price_history(token)")
                cursor.execute("CREATE INDEX idx_price_history_timestamp ON price_history(timestamp)")
                cursor.execute("CREATE INDEX idx_price_history_token_timestamp ON price_history(token, timestamp)")
        
                conn.commit()
                logger.logger.info("Successfully created price_history table")
            
            return True
        except Exception as e:
            logger.log_error("Ensure Price History Table Exists", str(e))
            logger.logger.error(f"Error in _ensure_price_history_table_exists: {str(e)}")
            logger.logger.debug(f"Traceback: {traceback.format_exc()}")
            conn.rollback()
            return False

    def analyze_price_history_coverage(self, token: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """
        Analyze price history data coverage
    
        Args:
            token: Token symbol (optional, analyzes all tokens if None)
            days: Number of days to analyze (default: 7)
        
        Returns:
            Dictionary with coverage analysis
        """
        conn, cursor = self._get_connection()
    
        try:
            results = {}
        
            # Get list of tokens to analyze
            if token:
                tokens = [token]
            else:
                cursor.execute("""
                    SELECT DISTINCT token FROM price_history
                    WHERE timestamp >= datetime('now', '-' || ? || ' days')
                """, (days,))
                tokens = [row['token'] for row in cursor.fetchall()]
        
            # Analyze each token
            for t in tokens:
                # Get time range
                cursor.execute("""
                    SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time, COUNT(*) as count
                    FROM price_history
                    WHERE token = ? AND timestamp >= datetime('now', '-' || ? || ' days')
                """, (t, days))
            
                data = cursor.fetchone()
                if not data or data['count'] == 0:
                    results[t] = {"status": "no_data"}
                    continue
            
                # Get all timestamps to analyze gaps
                cursor.execute("""
                    SELECT timestamp
                    FROM price_history
                    WHERE token = ? AND timestamp >= datetime('now', '-' || ? || ' days')
                    ORDER BY timestamp ASC
                """, (t, days))
            
                timestamps = []
                for row in cursor.fetchall():
                    ts = row['timestamp']
                    if isinstance(ts, str):
                        try:
                            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        except ValueError:
                            try:
                                ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                            except ValueError:
                                try:
                                    ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
                                except ValueError:
                                    continue
                    timestamps.append(ts)
            
                # Calculate gaps between timestamps
                gaps = []
                if len(timestamps) > 1:
                    for i in range(1, len(timestamps)):
                        gap_seconds = (timestamps[i] - timestamps[i-1]).total_seconds()
                        gaps.append(gap_seconds / 3600)  # Convert to hours
            
                # Calculate coverage for different timeframes
                min_time = self._standardize_timestamp(data['min_time'])
                max_time = self._standardize_timestamp(data['max_time'])
                time_span_hours = ((self.strip_timezone(max_time) - self.strip_timezone(min_time)).total_seconds() / 3600) if min_time and max_time else 0
            
                coverage = {
                    "token": t,
                    "record_count": data['count'],
                    "time_span_hours": time_span_hours,
                    "time_span_days": time_span_hours / 24,
                    "min_time": min_time,
                    "max_time": max_time,
                    "avg_interval_hours": time_span_hours / (data['count'] - 1) if data['count'] > 1 else None,
                    "gaps": {
                        "count": len(gaps),
                        "min_gap_hours": min(gaps) if gaps else None,
                        "max_gap_hours": max(gaps) if gaps else None,
                        "avg_gap_hours": sum(gaps) / len(gaps) if gaps else None,
                        "gaps_over_1h": sum(1 for g in gaps if g > 1),
                        "gaps_over_6h": sum(1 for g in gaps if g > 6),
                        "gaps_over_24h": sum(1 for g in gaps if g > 24),
                    }
                }
            
                # Add coverage assessment
                if time_span_hours >= 24*7:
                    coverage["7d_coverage"] = "complete"
                elif time_span_hours >= 24:
                    coverage["7d_coverage"] = "partial"
                else:
                    coverage["7d_coverage"] = "insufficient"
                
                if time_span_hours >= 24:
                    coverage["24h_coverage"] = "complete"
                elif time_span_hours >= 1:
                    coverage["24h_coverage"] = "partial"
                else:
                    coverage["24h_coverage"] = "insufficient"
                
                if time_span_hours >= 1:
                    coverage["1h_coverage"] = "complete"
                else:
                    coverage["1h_coverage"] = "insufficient"
            
                results[t] = coverage
        
            return results
        
        except Exception as e:
            logger.log_error("Analyze Price History Coverage", str(e))
            logger.logger.error(f"Error in analyze_price_history_coverage: {str(e)}")
            logger.logger.debug(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

    def calculate_price_changes(self, token: str, current_price: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate price changes for different periods with enhanced handling of intermittent data
    
        Args:
            token: Token symbol
            current_price: Optional current price (fetches latest if not provided)
    
        Returns:
            Dictionary with price changes for different periods
        """
        try:
            conn, cursor = self._get_connection()
        
            logger.logger.debug(f"calculate_price_changes called for token: {token}, current_price: {current_price}")
        
            # Get current price and timestamp if not provided
            current_time = datetime.now()
            if current_price is None:
                logger.logger.debug(f"No current price provided for {token}, querying database")
                cursor.execute("""
                    SELECT price, timestamp FROM price_history
                    WHERE token = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (token,))
            
                result = cursor.fetchone()
                if result:
                    current_price = result["price"]
                    current_time = self._standardize_timestamp(result["timestamp"])
                    logger.logger.debug(f"Found latest price for {token} in database: {current_price} at {current_time}")
                else:
                    logger.logger.warning(f"No price history found for {token}")
                    return {}  # No data available
        
            # Dictionary to store results
            price_changes = {}
        
            # Define time periods to calculate (in hours)
            periods = {
                'price_change_percentage_1h': 1,
                'price_change_percentage_24h': 24,
                'price_change_percentage_7d': 24 * 7,
                'price_change_percentage_30d': 24 * 30,
            }
        
            # Define maximum time difference for each period (more flexibility for longer periods)
            max_time_differences = {
                'price_change_percentage_1h': 0.5,      # 30 minutes for 1h
                'price_change_percentage_24h': 6,       # 6 hours for 24h
                'price_change_percentage_7d': 24,       # 24 hours for 7d
                'price_change_percentage_30d': 48,      # 48 hours for 30d
            }
        
            logger.logger.debug(f"Calculating price changes for {token} over periods: {list(periods.keys())}")
        
            # Calculate change for each period
            for period_name, hours in periods.items():
                try:
                    # Get target time for historical price
                    target_time = current_time - timedelta(hours=hours)
                    logger.logger.debug(f"Target time for {token} {period_name}: {target_time}")
                
                    # Get closest historical price with appropriate flexibility
                    historical_data = self._get_closest_historical_price(
                        token, 
                        target_time, 
                        max_time_difference_hours=max_time_differences.get(period_name)
                    )
                
                    if historical_data and historical_data['price'] > 0:
                        previous_price = historical_data['price']
                        actual_time = historical_data['timestamp']
                        time_diff_hours = historical_data['time_difference_hours']
                    
                        logger.logger.debug(
                            f"Found historical price for {token} for {period_name}: {previous_price} "
                            f"(target: {target_time}, actual: {actual_time}, diff: {time_diff_hours:.2f} hours)"
                        )
                    
                        # Calculate percentage change
                        percent_change = ((current_price / previous_price) - 1) * 100
                        logger.logger.debug(
                            f"Calculated {period_name} for {token}: {percent_change:.2f}% "
                            f"(from {previous_price} to {current_price})"
                        )
                    
                        # Store in results
                        price_changes[period_name] = percent_change
                    
                        # Also add shorter keys for compatibility
                        if period_name == 'price_change_percentage_24h':
                            price_changes['price_change_24h'] = percent_change
                            logger.logger.debug(f"Added compatibility key 'price_change_24h' for {token}: {percent_change:.2f}%")
                        elif period_name == 'price_change_percentage_7d':
                            price_changes['price_change_7d'] = percent_change
                            logger.logger.debug(f"Added compatibility key 'price_change_7d' for {token}: {percent_change:.2f}%")
                    else:
                        logger.logger.warning(f"No valid historical price found for {token} at {target_time} for {period_name}")
                    
                except Exception as period_error:
                    logger.log_error(f"Calculate {period_name} - {token}", str(period_error))
                    logger.logger.error(f"Error calculating {period_name} for {token}: {str(period_error)}")
                    logger.logger.debug(f"Traceback: {traceback.format_exc()}")
        
            # If we didn't calculate any changes, log it clearly
            if not price_changes:
                logger.logger.warning(f"No price changes calculated for {token} - no market change data available for comparison")
            else:
                logger.logger.debug(f"Successfully calculated price changes for {token}: {price_changes}")
        
            return price_changes
        
        except Exception as e:
            logger.log_error(f"Calculate Price Changes - {token}", str(e))
            logger.logger.error(f"Error in calculate_price_changes for {token}: {str(e)}")
            logger.logger.debug(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def add_replied_posts_table(self):
        """Add the replied_posts table if it doesn't exist"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS replied_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT NOT NULL,
                    post_url TEXT,
                    reply_content TEXT,
                    replied_at DATETIME NOT NULL,
                    UNIQUE(post_id)
                )
            """)
        
            # Create index for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_post_id ON replied_posts(post_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_replied_posts_post_url ON replied_posts(post_url)")
        
            conn.commit()
            logger.logger.info("Added replied_posts table to database")
            return True
        except Exception as e:
            logger.log_error("Add Replied Posts Table", str(e))
            conn.rollback()
            return False 

    def store_reply(self, post_id: str, post_url: Optional[str] = None, post_author: Optional[str] = None,
                post_text: Optional[str] = None, reply_text: Optional[str] = None, reply_time: Optional[datetime] = None):
        """
        Store a reply to a post in the database - Enterprise version with constraint resolution
        
        ENHANCED FEATURES:
        - Resolves UNIQUE constraint violations using UPSERT pattern
        - Handles duplicate reply attempts gracefully (updates existing replies)
        - Maintains full backward compatibility
        - Enhanced logging for conflict detection
        - Improved error classification

        Args:
            post_id: The ID of the post being replied to
            post_url: URL of the post (optional)
            post_author: Author of the original post (optional)
            post_text: The content of the original post (optional)
            reply_text: The content of your reply (optional)
            reply_time: Optional timestamp (defaults to current time)

        Returns:
            bool: True if stored successfully, False otherwise
        """
        conn = None
        try:
            if reply_time is None:
                reply_time = datetime.now()

            conn, cursor = self._get_connection()

            # First check if we need to create the table
            self._ensure_replied_posts_table_exists()

            # 🆕 ENTERPRISE: Check if reply already exists for logging
            cursor.execute("SELECT COUNT(*) FROM replied_posts WHERE post_id = ?", (post_id,))
            reply_exists = cursor.fetchone()[0] > 0

            # 🆕 ENTERPRISE: Use INSERT OR REPLACE to handle duplicate replies gracefully
            cursor.execute("""
                INSERT OR REPLACE INTO replied_posts (
                    post_id, post_url, reply_content, replied_at
                ) VALUES (?, ?, ?, ?)
            """, (
                post_id,
                post_url,
                reply_text,
                reply_time
            ))

            conn.commit()
            
            # 🆕 ENTERPRISE: Enhanced logging to track updates vs new replies
            if reply_exists:
                logger.logger.info(f"🔄 Updated existing reply for post {post_id}")
            else:
                logger.logger.debug(f"✅ Stored new reply for post {post_id}")
            
            return True

        except sqlite3.IntegrityError as e:
            # 🆕 ENTERPRISE: Specific handling for integrity constraint errors
            if conn:
                conn.rollback()
            logger.log_error("Reply Storage Integrity Error", str(e))
            logger.logger.error(f"❌ Integrity constraint failed for reply to post {post_id}: {e}")
            return False

        except sqlite3.OperationalError as e:
            # 🆕 ENTERPRISE: Handle database lock/operational errors
            if conn:
                conn.rollback()
            logger.log_error("Reply Storage Database Lock", str(e))
            logger.logger.error(f"❌ Database operation failed for reply to post {post_id}: {e}")
            return False

        except Exception as e:
            # Existing general error handling
            if conn:
                conn.rollback()
            logger.log_error("Store Reply", str(e))
            logger.logger.error(f"❌ Failed to store reply for post {post_id}: {e}")
            return False
        
    def store_content_analysis(self, post_id, content=None, analysis_data=None, 
                                reply_worthy=False, reply_score=0.0, features=None,
                                engagement_scores=None, response_focus=None, 
                                author_handle=None, post_url=None, timestamp=None):
        """
        Store content analysis results for a post - Enterprise version with constraint resolution
        
        ENHANCED FEATURES:
        - Resolves UNIQUE constraint violations using UPSERT pattern
        - Handles duplicate post_ids gracefully (updates existing records)
        - Maintains full backward compatibility
        - Same method signature and return behavior
        - Enhanced logging for conflict detection
        
        Args:
            post_id: Unique identifier for the post
            content: Original post text content (optional)
            analysis_data: Dictionary containing analysis results
            reply_worthy: Whether the post is worth replying to
            reply_score: Score indicating reply priority
            features: Post features extracted during analysis
            engagement_scores: Engagement metrics for the post
            response_focus: Recommended response approach
            author_handle: Twitter handle of the post author (optional)
            post_url: URL to the original post (optional)
            timestamp: Timestamp for the analysis (defaults to current time)

        Returns:
            bool: True if successfully stored, False otherwise
        """
        conn = None
        cursor = None
        
        try:
            # CRITICAL: Ensure table exists before attempting insert (NUMBA thread safe)
            self._ensure_content_analysis_table_exists()
            
            conn, cursor = self._get_connection()
            
            # Prepare timestamp
            if timestamp is None:
                timestamp = datetime.now()

            # Build analysis data if not provided
            if analysis_data is None:
                analysis_data = {
                    "reply_worthy": reply_worthy,
                    "reply_score": reply_score,
                    "features": features,
                    "engagement_scores": engagement_scores,
                    "response_focus": response_focus
                }

            # Convert analysis_data to JSON string
            analysis_json = json.dumps(analysis_data)

            # 🆕 ENTERPRISE FIX: Check if record exists first for logging
            cursor.execute("SELECT COUNT(*) FROM content_analysis WHERE post_id = ?", (post_id,))
            record_exists = cursor.fetchone()[0] > 0

            # 🆕 ENTERPRISE: Use INSERT OR REPLACE to handle duplicates gracefully
            cursor.execute("""
                INSERT OR REPLACE INTO content_analysis (
                    post_id, content, analysis_data, author_handle,
                    post_url, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                post_id,
                content,
                analysis_json,
                author_handle,
                post_url,
                timestamp
            ))

            conn.commit()
            
            # 🆕 ENTERPRISE: Enhanced logging to track updates vs inserts
            if record_exists:
                logger.logger.info(f"🔄 Updated existing content analysis for post {post_id}")
            else:
                logger.logger.debug(f"✅ Stored new content analysis for post {post_id}")
            
            return True

        except sqlite3.IntegrityError as e:
            # 🆕 ENTERPRISE: Specific handling for remaining integrity errors
            logger.log_error("Content Analysis Integrity Error", str(e))
            logger.logger.error(f"❌ Integrity constraint failed for post {post_id}: {e}")
            
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            
            return False

        except sqlite3.OperationalError as e:
            # 🆕 ENTERPRISE: Handle database lock errors
            logger.log_error("Content Analysis Database Lock", str(e))
            logger.logger.error(f"❌ Database operation failed for post {post_id}: {e}")
            
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            
            return False

        except Exception as e:
            # Existing general error handling
            logger.log_error("Store Content Analysis", str(e))
            logger.logger.error(f"❌ Failed to store content analysis for post {post_id}: {e}")
            
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            
            return False

    def _ensure_content_analysis_table_exists(self) -> None:
        """Ensure the content_analysis table exists in the database"""
        conn, cursor = self._get_connection()
        try:
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='content_analysis'")
            if cursor.fetchone() is None:
                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE content_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        post_id TEXT NOT NULL,
                        content TEXT,
                        analysis_data TEXT NOT NULL,
                        author_handle TEXT,
                        post_url TEXT,
                        timestamp DATETIME NOT NULL,
                        UNIQUE(post_id)
                    )
                """)
            
                # Create indexes for better query performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_analysis_post_id ON content_analysis(post_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_analysis_author ON content_analysis(author_handle)")
            
                conn.commit()
                logger.logger.info("Created content_analysis table in database")
        
        except Exception as e:
            logger.log_error("Ensure Content Analysis Table", str(e))
            if conn:
                conn.rollback()    

    def _ensure_replied_posts_table_exists(self):
        """Ensure the replied_posts table exists in the database"""
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='replied_posts'")
            if cursor.fetchone() is None:
                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE replied_posts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        post_id TEXT NOT NULL,
                        original_content TEXT,
                        reply_content TEXT,
                        UNIQUE(post_id)
                    )
                """)
                
            conn.commit()
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.log_error("Ensure Replied Posts Table", str(e))

    def mark_post_as_replied(self, post_id: str, post_url: Optional[str] = None, reply_content: Optional[str] = None) -> bool:
        """
        Mark a post as replied to - Enterprise version with optimized constraint handling
        
        ENHANCED FEATURES:
        - Eliminates UNIQUE constraint violations using INSERT OR IGNORE pattern
        - Idempotent operation (safe to call multiple times)
        - Maintains full backward compatibility
        - Optimized performance (single SQL operation instead of check + insert)
        - Enhanced logging and error handling

        Args:
            post_id: The ID of the post
            post_url: The URL of the post (optional)
            reply_content: The content of the reply (optional)

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            # Ensure the table exists
            self._ensure_replied_posts_table_exists()
        
            # 🆕 ENTERPRISE: Use INSERT OR IGNORE for idempotent operation
            # This approach is more efficient than check-then-insert and eliminates race conditions
            cursor.execute("""
                INSERT OR IGNORE INTO replied_posts (post_id, post_url, reply_content, replied_at)
                VALUES (?, ?, ?, ?)
            """, (post_id, post_url, reply_content, datetime.now()))
            
            # 🆕 ENTERPRISE: Check if row was actually inserted or already existed
            rows_affected = cursor.rowcount
            
            conn.commit()
            
            # 🆕 ENTERPRISE: Enhanced logging based on operation result
            if rows_affected > 0:
                logger.logger.debug(f"✅ Marked post {post_id} as replied (new entry)")
            else:
                logger.logger.debug(f"ℹ️ Post {post_id} was already marked as replied (no change)")
            
            # Always return True for idempotent behavior - operation succeeded regardless
            return True
            
        except sqlite3.IntegrityError as e:
            # 🆕 ENTERPRISE: Handle any remaining integrity constraints
            logger.log_error("Mark Post Replied Integrity Error", str(e))
            logger.logger.error(f"❌ Integrity constraint failed marking post {post_id} as replied: {e}")
            if conn:
                conn.rollback()
            return False

        except sqlite3.OperationalError as e:
            # 🆕 ENTERPRISE: Handle database lock/operational errors
            logger.log_error("Mark Post Replied Database Lock", str(e))
            logger.logger.error(f"❌ Database operation failed marking post {post_id} as replied: {e}")
            if conn:
                conn.rollback()
            return False

        except Exception as e:
            # Existing general error handling
            logger.log_error("Mark Post As Replied", str(e))
            logger.logger.error(f"❌ Failed to mark post {post_id} as replied: {e}")
            if conn:
                conn.rollback()
            return False

    def add_missing_columns(self):
        """Add missing columns to technical_indicators table if they don't exist"""
        conn, cursor = self._get_connection()
        changes_made = False
    
        try:
            # Check if columns exist
            cursor.execute("PRAGMA table_info(technical_indicators)")
            columns = [column[1] for column in cursor.fetchall()]
    
            # Add the ichimoku_data column if it doesn't exist
            if 'ichimoku_data' not in columns:
                cursor.execute("ALTER TABLE technical_indicators ADD COLUMN ichimoku_data TEXT")
                logger.logger.info("Added missing ichimoku_data column to technical_indicators table")
                changes_made = True
            
            # Add the pivot_points column if it doesn't exist
            if 'pivot_points' not in columns:
                cursor.execute("ALTER TABLE technical_indicators ADD COLUMN pivot_points TEXT")
                logger.logger.info("Added missing pivot_points column to technical_indicators table")
                changes_made = True
            
            conn.commit()
            return changes_made
        except Exception as e:
            logger.log_error("Add Missing Columns", str(e))
            conn.rollback()
            return False
    
    def add_ichimoku_column(self):
        """Add the missing ichimoku_data column to technical_indicators table if it doesn't exist"""
        conn, cursor = self._get_connection()
        try:
            # Check if column exists
            cursor.execute("PRAGMA table_info(technical_indicators)")
            columns = [column[1] for column in cursor.fetchall()]
        
            # Add the column if it doesn't exist
            if 'ichimoku_data' not in columns:
                cursor.execute("ALTER TABLE technical_indicators ADD COLUMN ichimoku_data TEXT")
                conn.commit()
                logger.logger.info("Added missing ichimoku_data column to technical_indicators table")
                return True
            return False
        except Exception as e:
            logger.log_error("Add Ichimoku Column", str(e))
            conn.rollback()
            return False

    @property
    def conn(self) -> sqlite3.Connection:
        """Thread-safe connection property - returns the connection for current thread"""
        conn, _ = self._get_connection()
        return conn
        
    @property
    def cursor(self) -> sqlite3.Cursor:
        """Thread-safe cursor property - returns the cursor for current thread"""
        _, cursor = self._get_connection()
        return cursor

    def _get_connection(self):
        """Get database connection, creating it if necessary - thread-safe version with NUMBA support and debugging"""
        
        # Check if this thread has a connection
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            # Get detailed thread information for debugging
            current_thread = threading.current_thread()
            thread_name = getattr(current_thread, 'name', 'Unknown')
            thread_id = getattr(current_thread, 'ident', 'Unknown')
            has_target = hasattr(current_thread, '_target')
            
            # Log thread details
            logger.logger.debug(f"Database connection requested by thread: {thread_name} (ID: {thread_id}, has_target: {has_target})")
            
            # Enhanced NUMBA worker thread detection
            is_numba_thread = (
                'ThreadPoolExecutor' in thread_name or 
                'Worker' in thread_name or
                'Thread-' in thread_name or  # Add this common pattern
                '_Thread-' in thread_name or  # Add this pattern too
                has_target or
                'numba' in thread_name.lower()  # Direct numba thread detection
            )
            
            logger.logger.debug(f"Thread classification: is_numba_thread={is_numba_thread}")
            
            # Validate database file before connection
            if not os.path.exists(self.db_path):
                logger.logger.error(f"Database file does not exist: {self.db_path}")
                raise sqlite3.OperationalError(f"Database file does not exist: {self.db_path}")
                
            if not os.access(self.db_path, os.R_OK | os.W_OK):
                logger.logger.error(f"Database file permissions issue: {self.db_path}")
                raise sqlite3.OperationalError(f"Database file not accessible: {self.db_path}")
            
            # Create connection with enhanced settings based on thread type
            try:
                if is_numba_thread:
                    logger.logger.debug(f"Creating NUMBA-compatible connection for thread {thread_name}")
                    self.local.conn = sqlite3.connect(
                        self.db_path, 
                        check_same_thread=False, 
                        timeout=30.0
                    )
                else:
                    logger.logger.debug(f"Creating standard connection for thread {thread_name}")
                    self.local.conn = sqlite3.connect(self.db_path, timeout=10.0)
                
                self.local.conn.row_factory = sqlite3.Row
                self.local.cursor = self.local.conn.cursor()
                
                # Test the connection
                self.local.cursor.execute("SELECT 1").fetchone()
                logger.logger.debug(f"Database connection successful for thread {thread_name}")
                
            except Exception as e:
                logger.logger.error(f"Database connection failed for thread {thread_name}: {e}")
                raise
        
        return self.local.conn, self.local.cursor
    
    def close_connections(self):
        """Close all database connections"""
        if hasattr(self.local, 'conn') and self.local.conn:
            self.local.conn.close()
            self.local.conn = None
            self.local.cursor = None

    def _initialize_database(self):
        """Create necessary tables if they don't exist"""
        conn, cursor = self._get_connection()
        cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    rsi REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    stoch_k REAL,
                    stoch_d REAL,
                    obv REAL,
                    adx REAL,
                    ichimoku_data TEXT,
                    pivot_points TEXT,
                    overall_trend TEXT,
                    trend_strength REAL,
                    volatility REAL,
                    raw_data JSON
                )
            """)

        try:
            # Market Data Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    chain TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    price_change_24h REAL,
                    market_cap REAL,
                    ath REAL,
                    ath_change_percentage REAL
                )
            """)

            # Posted Content Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS posted_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    content TEXT NOT NULL,
                    sentiment JSON NOT NULL,
                    trigger_type TEXT NOT NULL,
                    price_data JSON NOT NULL,
                    meme_phrases JSON NOT NULL,
                    is_prediction BOOLEAN DEFAULT 0,
                    prediction_data JSON,
                    timeframe TEXT DEFAULT '1h'
                )
            """)

            # Chain Mood History
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mood_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    chain TEXT NOT NULL,
                    mood TEXT NOT NULL,
                    indicators JSON NOT NULL
                )
            """)
        
            # Smart Money Indicators Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS smart_money_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    chain TEXT NOT NULL,
                    volume_z_score REAL,
                    price_volume_divergence BOOLEAN,
                    stealth_accumulation BOOLEAN,
                    abnormal_volume BOOLEAN,
                    volume_vs_hourly_avg REAL,
                    volume_vs_daily_avg REAL,
                    volume_cluster_detected BOOLEAN,
                    unusual_trading_hours JSON,
                    raw_data JSON
                )
            """)
        
            # Token Market Comparison Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_market_comparison (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    vs_market_avg_change REAL,
                    vs_market_volume_growth REAL,
                    outperforming_market BOOLEAN,
                    correlations JSON
                )
            """)
        
            # Token Correlations Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_correlations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    avg_price_correlation REAL NOT NULL,
                    avg_volume_correlation REAL NOT NULL,
                    full_data JSON NOT NULL
                )
            """)
        
            # Generic JSON Data Table for flexible storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generic_json_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    data_type TEXT NOT NULL,
                    data JSON NOT NULL
                )
            """)
        
            # PREDICTION TABLES
        
            # Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    prediction_value REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    lower_bound REAL,
                    upper_bound REAL,
                    prediction_rationale TEXT,
                    method_weights JSON,
                    model_inputs JSON,
                    technical_signals JSON,
                    expiration_time DATETIME NOT NULL
                )
            """)

            # Prediction Outcomes Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    actual_outcome REAL NOT NULL,
                    accuracy_percentage REAL NOT NULL,
                    was_correct BOOLEAN NOT NULL,
                    evaluation_time DATETIME NOT NULL,
                    deviation_from_prediction REAL NOT NULL,
                    market_conditions JSON,
                    FOREIGN KEY (prediction_id) REFERENCES price_predictions(id)
                )
            """)

            # Prediction Performance Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    correct_predictions INTEGER NOT NULL,
                    accuracy_rate REAL NOT NULL,
                    avg_deviation REAL NOT NULL,
                    updated_at DATETIME NOT NULL
                )
            """)
        
            # REMOVED THE DUPLICATE technical_indicators TABLE CREATION HERE
        
            # Statistical Models Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS statistical_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    forecast_value REAL NOT NULL,
                    confidence_80_lower REAL,
                    confidence_80_upper REAL,
                    confidence_95_lower REAL,
                    confidence_95_upper REAL,
                    model_parameters JSON,
                    input_data_summary JSON
                )
            """)
        
            # Machine Learning Models Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    forecast_value REAL NOT NULL,
                    confidence_80_lower REAL,
                    confidence_80_upper REAL,
                    confidence_95_lower REAL,
                    confidence_95_upper REAL,
                    feature_importance JSON,
                    model_parameters JSON
                )
            """)
        
            # Claude AI Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS claude_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    claude_model TEXT NOT NULL,
                    prediction_value REAL NOT NULL,
                    confidence_level REAL,
                    sentiment TEXT,
                    rationale TEXT,
                    key_factors JSON,
                    input_data JSON
                )
            """)
        
            # Timeframe Metrics Table - New table to track metrics by timeframe
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeframe_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    token TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    avg_accuracy REAL,
                    total_count INTEGER,
                    correct_count INTEGER,
                    model_weights JSON,
                    best_model TEXT,
                    last_updated DATETIME NOT NULL
                )
            """)
        
            # Create indices for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_chain ON market_data(chain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_posted_content_timestamp ON posted_content(timestamp)")
        
            # HERE'S THE FIX: Check if timeframe column exists in posted_content before creating index
            try:
                # Try to get column info
                cursor.execute("PRAGMA table_info(posted_content)")
                columns = [column[1] for column in cursor.fetchall()]
            
                # Check if timeframe column exists
                if 'timeframe' not in columns:
                    # Add the timeframe column if it doesn't exist
                    cursor.execute("ALTER TABLE posted_content ADD COLUMN timeframe TEXT DEFAULT '1h'")
                    conn.commit()
                    logger.logger.info("Added missing timeframe column to posted_content table")
            
                # Now it's safe to create the index
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_posted_content_timeframe ON posted_content(timeframe)")
            except Exception as e:
                logger.log_error("Timeframe Column Check", str(e))
        
            # Check if timeframe column exists in technical_indicators before creating index
            try:
                # Try to get column info
                cursor.execute("PRAGMA table_info(technical_indicators)")
                columns = [column[1] for column in cursor.fetchall()]
            
                # Check if timeframe column exists
                if 'timeframe' not in columns:
                    # Add the timeframe column if it doesn't exist
                    cursor.execute("ALTER TABLE technical_indicators ADD COLUMN timeframe TEXT DEFAULT '1h'")
                    conn.commit()
                    logger.logger.info("Added missing timeframe column to technical_indicators table")
            
                # Now it's safe to create the index
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_timeframe ON technical_indicators(timeframe)")
            except Exception as e:
                logger.log_error("Timeframe Column Check for technical_indicators", str(e))
        
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_history_timestamp ON mood_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_history_chain ON mood_history(chain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_smart_money_timestamp ON smart_money_indicators(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_smart_money_chain ON smart_money_indicators(chain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generic_json_timestamp ON generic_json_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generic_json_type ON generic_json_data(data_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_market_comparison_timestamp ON token_market_comparison(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_market_comparison_token ON token_market_comparison(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_correlations_timestamp ON token_correlations(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_correlations_token ON token_correlations(token)")
        
            # Prediction indices - Enhanced for timeframe support
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_token ON price_predictions(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_timeframe ON price_predictions(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_timestamp ON price_predictions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_expiration ON price_predictions(expiration_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_predictions_token_timeframe ON price_predictions(token, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_prediction_id ON prediction_outcomes(prediction_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_performance_token ON prediction_performance(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_performance_timeframe ON prediction_performance(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_performance_token_timeframe ON prediction_performance(token, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_token ON technical_indicators(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_timestamp ON technical_indicators(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_timeframe ON technical_indicators(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_statistical_forecasts_token ON statistical_forecasts(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_statistical_forecasts_timeframe ON statistical_forecasts(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_forecasts_token ON ml_forecasts(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_forecasts_timeframe ON ml_forecasts(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_claude_predictions_token ON claude_predictions(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_claude_predictions_timeframe ON claude_predictions(timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeframe_metrics_token ON timeframe_metrics(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeframe_metrics_timeframe ON timeframe_metrics(timeframe)")

            conn.commit()
            logger.logger.info("Database initialized successfully")
    
        except Exception as e:
            logger.log_error("Database Initialization", str(e))
            raise

    #########################
    # CORE DATA STORAGE METHODS
    #########################
    def _ensure_tech_columns_exist(self) -> None:
        """Ensure tech-related columns exist in the posted_content table"""
        conn, cursor = self._get_connection()
        try:
            # Check if columns exist
            cursor.execute("PRAGMA table_info(posted_content)")
            columns = [column[1] for column in cursor.fetchall()]
        
            # Add tech_category column if it doesn't exist
            if 'tech_category' not in columns:
                cursor.execute("ALTER TABLE posted_content ADD COLUMN tech_category TEXT")
                logger.logger.info("Added missing tech_category column to posted_content table")

            # Add timeframe column if it doesn't exist  
            if 'timeframe' not in columns:
                cursor.execute("ALTER TABLE posted_content ADD COLUMN timeframe TEXT DEFAULT '1h'")
                logger.logger.info("Added missing timeframe column to posted_content table")    
        
            # Add tech_metadata column if it doesn't exist
            if 'tech_metadata' not in columns:
                cursor.execute("ALTER TABLE posted_content ADD COLUMN tech_metadata TEXT")
                logger.logger.info("Added missing tech_metadata column to posted_content table")
            
                # Initialize tech_metadata as empty JSON for existing rows
                cursor.execute("UPDATE posted_content SET tech_metadata = '{}' WHERE tech_metadata IS NULL")
                logger.logger.info("Initialized tech_metadata as empty JSON for existing rows")
        
            # Add is_educational column if it doesn't exist
            if 'is_educational' not in columns:
                cursor.execute("ALTER TABLE posted_content ADD COLUMN is_educational BOOLEAN DEFAULT 0")
                logger.logger.info("Added missing is_educational column to posted_content table")
        
            # We'll ensure these operations are committed even if later operations fail
            conn.commit()
        
            # Backward compatibility: If there are existing rows with market_context or vs_market_change
            # in other columns, we could migrate them, but that's typically handled in a separate migration method
        
            conn.commit()
        except Exception as e:
            logger.log_error("Ensure Tech Columns Exist", str(e))
            conn.rollback()

    def get_tech_content(self, tech_category: Optional[str] = None, hours: int = 24, limit: int = 10) -> List[Dict]:
        """
        Get recent tech content posts
        Can filter by tech category
        """
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            query = """
                SELECT * FROM posted_content 
                WHERE tech_category IS NOT NULL
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params: List[Union[int, str]] = [hours]
        
            if tech_category:
                query += " AND tech_category = ?"
                params.append(tech_category)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
        
            cursor.execute(query, params)
        
            results = [dict(row) for row in cursor.fetchall()]
        
            # Parse JSON fields
            for result in results:
                result["sentiment"] = json.loads(result["sentiment"]) if result["sentiment"] else {}
                result["price_data"] = json.loads(result["price_data"]) if result["price_data"] else {}
                result["meme_phrases"] = json.loads(result["meme_phrases"]) if result["meme_phrases"] else {}
                result["prediction_data"] = json.loads(result["prediction_data"]) if result["prediction_data"] else None
                result["tech_metadata"] = json.loads(result["tech_metadata"]) if result["tech_metadata"] else None
            
            return results
        except Exception as e:
            if conn:
                conn.rollback()
            logger.log_error("Get Tech Content", str(e))
            return []        
    
    def calculate_market_comparison_data(self, token: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate market comparison data for a specific token
        Compares token performance against the overall market

        Args:
            token: Token to analyze
            market_data: Dictionary with market data for all tokens
    
        Returns:
            Dictionary with comparison metrics
        """
        try:
            # Enhanced logging for input data
            logger.logger.debug(f"calculate_market_comparison_data called for token: {token}")
            logger.logger.debug(f"Market data type: {type(market_data)}")
            logger.logger.debug(f"Market data keys: {list(market_data.keys()) if isinstance(market_data, dict) else 'Not a dictionary'}")
            logger.logger.debug(f"Token in market_data: {token in market_data if isinstance(market_data, dict) else False}")
        
            # Ensure we have valid market data
            if not market_data or not isinstance(market_data, dict):
                logger.logger.warning(f"Invalid market data for {token}: {type(market_data)}")
                return {"error": "Invalid market data"}
        
            # Ensure token exists in market data
            if token not in market_data:
                logger.logger.warning(f"Token {token} not found in market data keys: {list(market_data.keys())}")
                return {"error": f"Token {token} not found in market data"}
        
            # Get token data
            token_data = market_data[token]
            if not isinstance(token_data, dict):
                logger.logger.warning(f"Invalid data format for {token}: {type(token_data)}")
                return {"error": f"Invalid data format for {token}"}
        
            # Get current price with safety check
            token_price = None
            if 'current_price' in token_data:
                try:
                    price_val = token_data['current_price']
                    if price_val is not None and float(price_val) > 0:
                        token_price = float(price_val)
                        logger.logger.debug(f"Found valid price for {token}: {token_price}")
                    else:
                        logger.logger.warning(f"Invalid price value for {token}: {price_val}")
                except (ValueError, TypeError) as e:
                    logger.logger.warning(f"Error converting price for {token}: {str(e)}")
                    pass
    
            if token_price is None:
                logger.logger.warning(f"No valid price found for {token}")
                return {"error": f"No valid price for {token}"}
        
            # Calculate token price changes if not already available
            token_changes = {}
    
            # First try to use existing data
            for change_key in ['price_change_percentage_24h', 'price_change_24h', 'change_24h', '24h_change']:
                if change_key in token_data and token_data[change_key] is not None:
                    try:
                        token_changes['24h'] = float(token_data[change_key])
                        logger.logger.debug(f"Found existing {change_key} for {token}: {token_changes['24h']}")
                        break
                    except (ValueError, TypeError) as e:
                        logger.logger.warning(f"Error converting {change_key} for {token}: {str(e)}")
                        pass
    
            # If not found, calculate our own
            if '24h' not in token_changes:
                logger.logger.debug(f"No existing price change data found for {token}, calculating from history")
                calc_changes = self.calculate_price_changes(token, token_price)
                logger.logger.debug(f"Calculated price changes for {token}: {calc_changes}")
            
                if calc_changes and 'price_change_percentage_24h' in calc_changes:
                    token_changes['24h'] = calc_changes['price_change_percentage_24h']
                    logger.logger.debug(f"Using calculated price change for {token}: {token_changes['24h']}")
                else:
                    token_changes['24h'] = 0  # Default if no data available
                    logger.logger.warning(f"No calculated price change available for {token}, using default 0")
            
            # Calculate market averages
            market_changes = {}
            valid_tokens = 0
            sum_24h_change = 0
    
            # Log the number of tokens to analyze
            other_tokens = [t for t in market_data.keys() if t != token and isinstance(market_data[t], dict)]
            logger.logger.debug(f"Analyzing {len(other_tokens)} other tokens for market average")
        
            # Calculate market average (excluding the token itself)
            for other_token, other_data in market_data.items():
                if other_token == token or not isinstance(other_data, dict):
                    continue
            
                # Get 24h change for this token
                other_change = None
        
                # First try to use existing data
                for change_key in ['price_change_percentage_24h', 'price_change_24h', 'change_24h', '24h_change']:
                    if change_key in other_data and other_data[change_key] is not None:
                        try:
                            other_change = float(other_data[change_key])
                            logger.logger.debug(f"Found existing {change_key} for {other_token}: {other_change}")
                            break
                        except (ValueError, TypeError) as e:
                            logger.logger.debug(f"Error converting {change_key} for {other_token}: {str(e)}")
                            pass
        
                # If not found, calculate our own
                if other_change is None:
                    logger.logger.debug(f"No existing price change found for {other_token}, trying to calculate")
                    other_price = None
                    if 'current_price' in other_data:
                        try:
                            price_val = other_data['current_price']
                            if price_val is not None and float(price_val) > 0:
                                other_price = float(price_val)
                                logger.logger.debug(f"Found valid price for {other_token}: {other_price}")
                            else:
                                logger.logger.warning(f"Invalid price value for {other_token}: {price_val}")
                        except (ValueError, TypeError) as e:
                            logger.logger.warning(f"Error converting price for {other_token}: {str(e)}")
                            pass
            
                    if other_price is not None:
                        calc_changes = self.calculate_price_changes(other_token, other_price)
                        logger.logger.debug(f"Calculated price changes for {other_token}: {calc_changes}")
                    
                        if calc_changes and 'price_change_percentage_24h' in calc_changes:
                            other_change = calc_changes['price_change_percentage_24h']
                            logger.logger.debug(f"Using calculated price change for {other_token}: {other_change}")
        
                # Add to market average if we have valid data
                if other_change is not None:
                    sum_24h_change += other_change
                    valid_tokens += 1
                    logger.logger.debug(f"Added {other_token} to market average with change: {other_change}")
        
            # Calculate market average
            market_avg_24h_change = sum_24h_change / valid_tokens if valid_tokens > 0 else 0
            logger.logger.debug(f"Market average 24h change: {market_avg_24h_change} (from {valid_tokens} tokens)")
            market_changes['24h'] = market_avg_24h_change
    
            # Calculate comparison metrics
            vs_market_avg_change = token_changes['24h'] - market_changes['24h']
            logger.logger.debug(f"{token} vs market change: {vs_market_avg_change} ({token_changes['24h']} vs {market_changes['24h']})")
        
            outperforming_market = token_changes['24h'] > market_changes['24h']
            logger.logger.debug(f"{token} outperforming market: {outperforming_market}")
    
            # Calculate volume comparison if data available
            token_volume = token_data.get('volume', 0)
            logger.logger.debug(f"{token} volume: {token_volume}")
        
            market_volume_sum = 0
            market_volume_tokens = 0
    
            for other_token, other_data in market_data.items():
                if other_token == token or not isinstance(other_data, dict):
                    continue
            
                other_volume = other_data.get('volume', None)
                if other_volume is not None and other_volume > 0:
                    market_volume_sum += other_volume
                    market_volume_tokens += 1
    
            market_avg_volume = market_volume_sum / market_volume_tokens if market_volume_tokens > 0 else 0
            logger.logger.debug(f"Market average volume: {market_avg_volume} (from {market_volume_tokens} tokens)")
        
            vs_market_volume_ratio = token_volume / market_avg_volume if market_avg_volume > 0 else 1
            logger.logger.debug(f"{token} vs market volume ratio: {vs_market_volume_ratio}")
    
            # For volume growth, we need historical data
            # For now, just set a neutral value
            vs_market_volume_growth = 0
            logger.logger.debug(f"{token} vs market volume growth: {vs_market_volume_growth} (default value)")
    
            # Calculate correlations with other tokens
            correlations = {}
    
            # Top tokens to calculate correlations with
            top_tokens = ["BTC", "ETH"]
            logger.logger.debug(f"Calculating correlations with top tokens: {top_tokens}")
    
            # Add any other tokens in the market data, up to a limit
            other_tokens = [t for t in market_data.keys() if t != token and t not in top_tokens]
            top_tokens.extend(other_tokens[:3])  # Limit to 3 additional tokens
            logger.logger.debug(f"Extended correlation token list: {top_tokens}")
    
            for other_token in top_tokens:
                if other_token == token or other_token not in market_data:
                    continue
            
                # Calculate simple price correlation
                # This is a very basic correlation - in a real implementation, 
                # you would use historical price data for a proper correlation
                other_price_change = None
        
                # Try to get 24h change for other token
                other_data = market_data[other_token]
                if not isinstance(other_data, dict):
                    continue
            
                for change_key in ['price_change_percentage_24h', 'price_change_24h', 'change_24h', '24h_change']:
                    if change_key in other_data and other_data[change_key] is not None:
                        try:
                            other_price_change = float(other_data[change_key])
                            logger.logger.debug(f"Found existing {change_key} for correlation with {other_token}: {other_price_change}")
                            break
                        except (ValueError, TypeError) as e:
                            logger.logger.debug(f"Error converting {change_key} for correlation with {other_token}: {str(e)}")
                            pass
        
                if other_price_change is None:
                    logger.logger.debug(f"No existing price change found for correlation with {other_token}, trying to calculate")
                    other_price = None
                    if 'current_price' in other_data:
                        try:
                            price_val = other_data['current_price']
                            if price_val is not None and float(price_val) > 0:
                                other_price = float(price_val)
                            else:
                                logger.logger.warning(f"Invalid price value for {other_token}: {price_val}")
                        except (ValueError, TypeError) as e:
                            logger.logger.warning(f"Error converting price for {other_token}: {str(e)}")
                            pass
            
                    if other_price is not None:
                        calc_changes = self.calculate_price_changes(other_token, other_price)
                        if calc_changes and 'price_change_percentage_24h' in calc_changes:
                            other_price_change = calc_changes['price_change_percentage_24h']
                            logger.logger.debug(f"Using calculated price change for correlation with {other_token}: {other_price_change}")
        
                # Skip if we couldn't get price change data
                if other_price_change is None:
                    logger.logger.warning(f"Could not determine price change for correlation with {other_token}")
                    continue
            
                # Calculate simple directional correlation
                if (token_changes['24h'] > 0 and other_price_change > 0) or (token_changes['24h'] < 0 and other_price_change < 0):
                    direction_correlation = 1.0  # Same direction
                    logger.logger.debug(f"Positive correlation between {token} and {other_token}")
                elif (token_changes['24h'] == 0 or other_price_change == 0):
                    direction_correlation = 0.0  # One is neutral
                    logger.logger.debug(f"Neutral correlation between {token} and {other_token}")
                else:
                    direction_correlation = -1.0  # Opposite directions
                    logger.logger.debug(f"Negative correlation between {token} and {other_token}")
            
                # Add to correlations
                correlations[other_token] = {
                    "price_change_correlation": direction_correlation,
                    "other_token_change": other_price_change
                }
    
            # Prepare result
            result = {
                "token": token,
                "token_change_24h": token_changes['24h'],
                "market_avg_change_24h": market_changes['24h'],
                "vs_market_avg_change": vs_market_avg_change,
                "outperforming_market": outperforming_market,
                "vs_market_volume_ratio": vs_market_volume_ratio,
                "vs_market_volume_growth": vs_market_volume_growth,
                "correlations": correlations
            }
    
            # Store the result in database for future reference
            try:
                logger.logger.debug(f"Storing market comparison data for {token} in database")
                self.store_token_market_comparison(
                    token=token,
                    vs_market_avg_change=vs_market_avg_change,
                    vs_market_volume_growth=vs_market_volume_growth,
                    outperforming_market=outperforming_market,
                    correlations=correlations
                )
                logger.logger.debug(f"Successfully stored market comparison data for {token}")
            except Exception as store_error:
                logger.logger.error(f"Error storing market comparison data for {token}: {str(store_error)}")
    
            return result
    
        except Exception as e:
            logger.log_error(f"Calculate Market Comparison - {token}", str(e))
            logger.logger.error(f"Error in calculate_market_comparison_data for {token}: {str(e)}")
            return {"error": str(e)}
    
    def store_market_data(self, chain: str, data: Dict[str, Any]) -> None:
        """
        Store market data for a specific chain
        Enhanced to also store in price_history table for change calculations
    
        Args:
            chain: Token symbol
            data: Market data dictionary
        """
        conn, cursor = self._get_connection()
        try:
            current_time = datetime.now()
        
            # Helper function to safely extract numeric value with fallback
            def safe_extract(data_dict, key, default=0):
                try:
                    value = data_dict.get(key, default)
                    if value is None:
                        return default
                    if isinstance(value, (int, float)):
                        return value
                    return float(value)  # Try to convert strings or other types
                except (ValueError, TypeError):
                    return default
        
            # Extract data with safety checks and fallbacks
            current_price = safe_extract(data, 'current_price')
            volume = safe_extract(data, 'volume')
        
            # Try multiple keys for price change
            price_change_24h = None
            for change_key in ['price_change_percentage_24h', 'price_change_24h', 'change_24h', '24h_change']:
                if change_key in data:
                    price_change_24h = safe_extract(data, change_key)
                    if price_change_24h != 0:  # Found a non-zero value
                        break
        
            # If no value found, use calculated value or default
            if price_change_24h is None:
                # Try to calculate from price history
                price_changes = self.calculate_price_changes(chain, current_price)
                price_change_24h = price_changes.get('price_change_percentage_24h', 0)
        
            # Extract other fields with fallbacks
            market_cap = safe_extract(data, 'market_cap')
            ath = safe_extract(data, 'ath')
            ath_change_percentage = safe_extract(data, 'ath_change_percentage')
            total_supply = safe_extract(data, 'total_supply')
            circulating_supply = safe_extract(data, 'circulating_supply')
        
            # Handle additional fields that might be in the data
            # Convert any datetime objects to ISO format
            additional_data = {}
            for key, value in data.items():
                if key not in ['current_price', 'volume', 'price_change_percentage_24h', 'price_change_24h', 
                              'market_cap', 'ath', 'ath_change_percentage', 'total_supply', 'circulating_supply']:
                    if isinstance(value, datetime):
                        additional_data[key] = value.isoformat()
                    elif isinstance(value, (list, dict)):
                        additional_data[key] = json.dumps(value)
                    else:
                        additional_data[key] = value
        
            # Only store in market_data if price is valid
            if current_price > 0:
                # Insert into original market_data table
                cursor.execute("""
                    INSERT INTO market_data (
                        timestamp, chain, price, volume, price_change_24h, 
                        market_cap, ath, ath_change_percentage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    current_time,
                    chain,
                    current_price,
                    volume,
                    price_change_24h,
                    market_cap,
                    ath,
                    ath_change_percentage
                ))
            else:
                logger.logger.warning(f"Skipping market_data insert for {chain}: Invalid price {current_price}")
        
            # Make sure price_history table exists
            self._ensure_price_history_table_exists()
        
            # Always store in price_history for accurate historical tracking
            # Even with missing price, we might have other useful data
            if current_price > 0:  # Only store valid prices
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO price_history (
                            token, timestamp, price, volume, market_cap,
                            total_supply, circulating_supply, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chain,
                        current_time,
                        current_price,
                        volume,
                        market_cap,
                        total_supply,
                        circulating_supply,
                        current_time
                    ))
                except Exception as price_hist_error:
                    # Log but continue - this shouldn't break the main market_data insert
                    logger.log_error(f"Price History Insert - {chain}", str(price_hist_error))
        
            # Store any additional data in generic_json_data if needed
            if additional_data:
                try:
                    cursor.execute("""
                        INSERT INTO generic_json_data (
                            timestamp, data_type, data
                        ) VALUES (?, ?, ?)
                    """, (
                        current_time,
                        f"market_data_extended_{chain}",
                        json.dumps(additional_data)
                    ))
                except Exception as json_error:
                    # Log but continue - this is supplementary data
                    logger.log_error(f"Generic JSON Data - {chain}", str(json_error))
        
            conn.commit()
            logger.logger.debug(f"Stored market data for {chain} at {current_time.isoformat()}")
        
        except Exception as e:
            logger.log_error(f"Store Market Data - {chain}", str(e))
            if 'conn' in locals() and conn:
                conn.rollback()

    def enhance_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance market data with calculated price changes
        Handles various input formats and ensures backward compatibility

        Args:
            market_data: Market data dictionary or list
    
        Returns:
            Enhanced market data dictionary
        """
        try:
            logger.logger.debug(f"enhance_market_data called with data type: {type(market_data)}")
        
            # Log input data structure overview
            if isinstance(market_data, dict):
                logger.logger.debug(f"Input is dictionary with {len(market_data)} keys")
                sample_keys = list(market_data.keys())[:5]
                logger.logger.debug(f"Sample keys: {sample_keys}")
            
                # Log a sample value structure if available
                if sample_keys and sample_keys[0] in market_data:
                    sample_value = market_data[sample_keys[0]]
                    logger.logger.debug(f"Sample value type for key '{sample_keys[0]}': {type(sample_value)}")
                
                    if isinstance(sample_value, dict):
                        logger.logger.debug(f"Sample value keys: {list(sample_value.keys())[:5]}")
            elif isinstance(market_data, list):
                logger.logger.debug(f"Input is list with {len(market_data)} items")
            
                # Log a sample item structure if available
                if market_data and len(market_data) > 0:
                    sample_item = market_data[0]
                    logger.logger.debug(f"Sample item type: {type(sample_item)}")
                
                    if isinstance(sample_item, dict):
                        logger.logger.debug(f"Sample item keys: {list(sample_item.keys())[:5]}")
            else:
                logger.logger.warning(f"Unexpected market_data type: {type(market_data)}")
                return market_data  # Return as-is if format unknown
        
            # Handle different input formats
            market_dict = {}
    
            # Convert from list if needed
            if isinstance(market_data, list):
                logger.logger.debug(f"Converting list market_data to dict, length: {len(market_data)}")

                for item in market_data:
                    if isinstance(item, dict):
                        # Force type casting to resolve inheritance conflicts
                        dict_item = cast(Dict[str, Any], item)
                        if 'symbol' in dict_item:
                            # Use uppercase symbol as key for consistency
                            symbol = dict_item['symbol'].upper()
                            market_dict[symbol] = dict_item
                            logger.logger.debug(f"Added symbol to market_dict: {symbol}")
                        
                            # Also add using id if available
                            if 'id' in dict_item:
                                market_dict[dict_item['id']] = dict_item
                                logger.logger.debug(f"Also added id to market_dict: {dict_item['id']}")
                        else:
                            logger.logger.warning(f"Skipping dictionary item without 'symbol' key. Keys: {list(dict_item.keys())}")
                    else:
                        logger.logger.warning(f"Skipping non-dictionary item in list: {type(item)}")
            elif isinstance(market_data, dict):
                # Already in dict format
                market_dict = market_data
                logger.logger.debug("Input already in dictionary format")
            else:
                logger.logger.warning(f"Unexpected market_data type: {type(market_data)}")
                return market_data  # Return as-is if format unknown
        
            # Log the converted market_dict structure
            logger.logger.debug(f"Converted market_dict has {len(market_dict)} entries")
        
            # Process each token
            processed_tokens = 0
            price_history_stored = 0
            price_changes_calculated = 0
            errors_encountered = 0
        
            for token, data in market_dict.items():
                try:
                    logger.logger.debug(f"Processing token: {token}")
                
                    if not isinstance(data, dict):
                        logger.logger.warning(f"Skipping non-dictionary data for token {token}: {type(data)}")
                        continue
                
                    processed_tokens += 1
                
                    # Extract current price with safety check
                    current_price = None
                    if 'current_price' in data:
                        try:
                            price_val = data['current_price']
                            if price_val is not None and float(price_val) > 0:
                                current_price = float(price_val)
                                logger.logger.debug(f"Found valid price for {token}: {current_price}")
                            else:
                                logger.logger.warning(f"Invalid price value for {token}: {price_val}")
                        except (ValueError, TypeError) as e:
                            logger.logger.warning(f"Error converting price for {token}: {str(e)}")
                            pass
            
                    # Skip if no valid price
                    if current_price is None or current_price <= 0:
                        logger.logger.warning(f"Skipping {token} due to invalid price: {current_price}")
                        continue
                
                    # Store in price_history for future calculations
                    # Use try/except to prevent one token's errors from affecting others
                    try:
                        # Extract volume with safety check
                        volume = None
                        if 'volume' in data:
                            try:
                                vol_val = data['volume']
                                if vol_val is not None:
                                    volume = float(vol_val)
                                    logger.logger.debug(f"Found valid volume for {token}: {volume}")
                                else:
                                    logger.logger.debug(f"No volume data for {token}")
                            except (ValueError, TypeError) as e:
                                logger.logger.warning(f"Error converting volume for {token}: {str(e)}")
                                pass
            
                        # Extract market cap with safety check
                        market_cap = None
                        if 'market_cap' in data:
                            try:
                                cap_val = data['market_cap']
                                if cap_val is not None:
                                    market_cap = float(cap_val)
                                    logger.logger.debug(f"Found valid market cap for {token}: {market_cap}")
                                else:
                                    logger.logger.debug(f"No market cap data for {token}")
                            except (ValueError, TypeError) as e:
                                logger.logger.warning(f"Error converting market cap for {token}: {str(e)}")
                                pass
                    
                        # Extract total supply with safety check
                        total_supply = None
                        if 'total_supply' in data:
                            try:
                                supply_val = data['total_supply']
                                if supply_val is not None:
                                    total_supply = float(supply_val)
                                    logger.logger.debug(f"Found valid total supply for {token}: {total_supply}")
                                else:
                                    logger.logger.debug(f"No total supply data for {token}")
                            except (ValueError, TypeError) as e:
                                logger.logger.warning(f"Error converting total supply for {token}: {str(e)}")
                                pass
                                
                        # Extract circulating supply with safety check
                        circulating_supply = None
                        if 'circulating_supply' in data:
                            try:
                                supply_val = data['circulating_supply']
                                if supply_val is not None:
                                    circulating_supply = float(supply_val)
                                    logger.logger.debug(f"Found valid circulating supply for {token}: {circulating_supply}")
                                else:
                                    logger.logger.debug(f"No circulating supply data for {token}")
                            except (ValueError, TypeError) as e:
                                logger.logger.warning(f"Error converting circulating supply for {token}: {str(e)}")
                                pass
            
                        # Store in price_history
                        logger.logger.debug(f"Storing price history for {token}")
                        stored = self.store_price_history(
                            token=token,
                            price=current_price,
                            volume=volume,
                            market_cap=market_cap,
                            total_supply=total_supply,
                            circulating_supply=circulating_supply
                        )
                    
                        if stored:
                            logger.logger.debug(f"Successfully stored price history for {token}")
                            price_history_stored += 1
                        else:
                            logger.logger.warning(f"Failed to store price history for {token}")
                    except Exception as store_error:
                        logger.log_error(f"Store Price History - {token}", str(store_error))
                        logger.logger.error(f"Error storing price history for {token}: {str(store_error)}")
                        # Continue processing other tokens even if one fails
                        errors_encountered += 1
                        continue
            
                    # Calculate price changes
                    try:
                        logger.logger.debug(f"Calculating price changes for {token}")
                        price_changes = self.calculate_price_changes(token, current_price)
                    
                        if price_changes:
                            logger.logger.debug(f"Successfully calculated price changes for {token}: {price_changes}")
                        
                            # Update data with calculated changes
                            for change_key, change_value in price_changes.items():
                                if change_value is not None:
                                    data[change_key] = change_value
                                    logger.logger.debug(f"Added calculated change to {token} data: {change_key}={change_value}")
                            
                                # Also set the original key if it exists but is None/0
                                if change_key == 'price_change_percentage_24h':
                                    # For backward compatibility with existing code
                                    if 'price_change_24h' not in data or data['price_change_24h'] is None or data['price_change_24h'] == 0:
                                        data['price_change_24h'] = change_value
                                        logger.logger.debug(f"Added calculated change to {token} data for compatibility: price_change_24h={change_value}")
                        
                            price_changes_calculated += 1
                        else:
                            logger.logger.warning(f"No price changes calculated for {token}")
                    except Exception as calc_error:
                        logger.log_error(f"Calculate Price Changes - {token}", str(calc_error))
                        logger.logger.error(f"Error calculating price changes for {token}: {str(calc_error)}")
                        errors_encountered += 1
            
                except Exception as token_error:
                    logger.log_error(f"Enhance Market Data - {token}", str(token_error))
                    logger.logger.error(f"Error processing token {token}: {str(token_error)}")
                    errors_encountered += 1
        
            # Log summary statistics
            logger.logger.info(f"enhance_market_data summary: processed {processed_tokens} tokens, "
                             f"stored {price_history_stored} to price history, "
                             f"calculated changes for {price_changes_calculated}, "
                             f"encountered {errors_encountered} errors")
        
            # Return in the same format that was provided
            if isinstance(market_data, list):
                # Convert back to list
                logger.logger.debug("Converting enhanced market_dict back to list")
                result = list(market_dict.values())
            
                # Remove duplicates (tokens added by both symbol and id)
                seen_items = set()
                unique_result = []

                for item in result:
                    # Use a unique identifier for deduplication
                    item_id = item.get('id') or item.get('symbol', '')
                    if item_id and item_id not in seen_items:
                        seen_items.add(item_id)
                        unique_result.append(item)

                logger.logger.debug(f"Returning enhanced market data list with {len(unique_result)} unique entries")
                return market_dict  # Return the dictionary, not the list
    
            logger.logger.debug(f"Returning enhanced market data dictionary with {len(market_dict)} entries")
            return market_dict
    
        except Exception as e:
            logger.log_error("Enhance Market Data", str(e))
            logger.logger.error(f"Error in enhance_market_data: {str(e)}")
            logger.logger.debug(f"Traceback: {traceback.format_exc()}")
            # Return original to avoid breaking anything
            return market_data

    def store_token_correlations(self, token: str, correlations: Dict[str, Any]) -> None:
        """Store token-specific correlation data"""
        conn, cursor = self._get_connection()
        try:
            # Extract average correlations
            avg_price_corr = correlations.get('avg_price_correlation', 0)
            avg_volume_corr = correlations.get('avg_volume_correlation', 0)
            
            cursor.execute("""
                INSERT INTO token_correlations (
                    timestamp, token, avg_price_correlation, avg_volume_correlation, full_data
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                avg_price_corr,
                avg_volume_corr,
                json.dumps(correlations)
            ))
            conn.commit()
            logger.logger.debug(f"Stored correlation data for {token}")
        except Exception as e:
            logger.log_error(f"Store Token Correlations - {token}", str(e))
            conn.rollback()
            
    def store_token_market_comparison(self, token: str, vs_market_avg_change: float,
                                    vs_market_volume_growth: float, outperforming_market: bool,
                                    correlations: Dict[str, Any]) -> None:
        """Store token vs market comparison data - NUMBA thread safe version"""
        
        # CRITICAL: Ensure table exists before attempting insert
        self._ensure_critical_tables_exist()
        
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                INSERT INTO token_market_comparison (
                    timestamp, token, vs_market_avg_change, vs_market_volume_growth,
                    outperforming_market, correlations
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                vs_market_avg_change,
                vs_market_volume_growth,
                1 if outperforming_market else 0,
                json.dumps(correlations)
            ))
            conn.commit()
            logger.logger.debug(f"✅ Stored market comparison data for {token}")
            
        except Exception as e:
            logger.log_error(f"Store Token Market Comparison - {token}", str(e))
            logger.logger.error(f"❌ Failed to store market comparison for {token}: {e}")
            if conn:
                conn.rollback()

    def store_posted_content(self, content: str, sentiment: Dict,
                            trigger_type: str, price_data: Optional[Dict] = None,
                            meme_phrases: Optional[Dict] = None, is_prediction: bool = False,
                            prediction_data: Optional[Dict] = None, timeframe: str = "1h",
                            tech_category: Optional[str] = None, tech_metadata: Optional[Dict] = None,
                            is_educational: bool = False, market_context: Optional[Dict] = None,
                            vs_market_change: Optional[float] = None, market_sentiment: Optional[str] = None,
                            timestamp: Optional[datetime] = None) -> bool:
        """
        Store posted content with metadata, timeframe and tech-related fields
        
        Args:
            content: The posted content text
            sentiment: Dictionary containing sentiment data
            trigger_type: Type of trigger that caused the post
            price_data: Optional price data dictionary
            meme_phrases: Optional meme phrases dictionary
            is_prediction: Whether this is a prediction post
            prediction_data: Optional prediction data dictionary
            timeframe: Timeframe for the content (default "1h")
            tech_category: Optional tech category classification
            tech_metadata: Optional tech metadata dictionary
            is_educational: Whether this is educational content
            market_context: Optional market context dictionary
            vs_market_change: Optional vs market change percentage
            market_sentiment: Optional market sentiment string
            timestamp: Optional timestamp (defaults to current time)
        
        Returns:
            bool: True if successfully stored, False otherwise
        """
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            # Input validation
            if content is None or content.strip() == "":
                logger.logger.warning("Attempt to store empty content in posted_content table")
                content = "[Empty Content]"  # Provide a default value to avoid NOT NULL constraint
            
            if not isinstance(sentiment, dict):
                logger.logger.warning("Invalid sentiment data type, using empty dict")
                sentiment = {}
                
            if not isinstance(trigger_type, str) or not trigger_type.strip():
                logger.logger.error("Invalid trigger_type provided")
                return False
            
            # First check if tech columns exist, add them if they don't
            self._ensure_tech_columns_exist()
            
            # Set defaults for mutable parameters
            if price_data is None:
                price_data = {}
            if meme_phrases is None:
                meme_phrases = {}
            
            # Use provided timestamp or default to current time
            current_time = timestamp if timestamp is not None else datetime.now()

            # Helper function to convert datetime objects in a dictionary to ISO format strings
            def datetime_to_iso(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: datetime_to_iso(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [datetime_to_iso(item) for item in obj]
                return obj

            # Process JSON data to handle datetime objects
            try:
                sentiment_json = json.dumps(datetime_to_iso(sentiment))
            except (TypeError, ValueError) as e:
                logger.logger.warning(f"Failed to serialize sentiment data: {e}")
                sentiment_json = json.dumps({})

            # Include vs_market_change in price_data if provided
            if vs_market_change is not None:
                price_data_copy = price_data.copy() if price_data else {}
                price_data_copy['vs_market_change'] = vs_market_change
                try:
                    price_data_json = json.dumps(datetime_to_iso(price_data_copy))
                except (TypeError, ValueError) as e:
                    logger.logger.warning(f"Failed to serialize price_data: {e}")
                    price_data_json = json.dumps({})
            else:
                try:
                    price_data_json = json.dumps(datetime_to_iso(price_data))
                except (TypeError, ValueError) as e:
                    logger.logger.warning(f"Failed to serialize price_data: {e}")
                    price_data_json = json.dumps({})
        
            try:
                meme_phrases_json = json.dumps(datetime_to_iso(meme_phrases))
            except (TypeError, ValueError) as e:
                logger.logger.warning(f"Failed to serialize meme_phrases: {e}")
                meme_phrases_json = json.dumps({})
                
            prediction_data_json = None
            if prediction_data:
                try:
                    prediction_data_json = json.dumps(datetime_to_iso(prediction_data))
                except (TypeError, ValueError) as e:
                    logger.logger.warning(f"Failed to serialize prediction_data: {e}")
                    prediction_data_json = None
                    
            tech_metadata_json = None
            if tech_metadata:
                try:
                    tech_metadata_json = json.dumps(datetime_to_iso(tech_metadata))
                except (TypeError, ValueError) as e:
                    logger.logger.warning(f"Failed to serialize tech_metadata: {e}")
                    tech_metadata_json = None

            cursor.execute("""
                INSERT INTO posted_content (
                    timestamp, content, sentiment, trigger_type, 
                    price_data, meme_phrases, is_prediction, prediction_data, timeframe,
                    tech_category, tech_metadata, is_educational
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                current_time,
                content,
                sentiment_json,
                trigger_type,
                price_data_json,
                meme_phrases_json,
                1 if is_prediction else 0,
                prediction_data_json,
                timeframe,
                tech_category,
                tech_metadata_json,
                1 if is_educational else 0
            ))
            
            # Verify the insert was successful
            if cursor.rowcount == 1:
                conn.commit()
                logger.logger.debug(f"Successfully stored posted content: {content[:50]}...")
                return True
            else:
                logger.logger.warning("Insert operation did not affect any rows")
                conn.rollback()
                return False
                
        except sqlite3.IntegrityError as e:
            logger.log_error("Store Posted Content - Integrity Error", str(e))
            if conn:
                conn.rollback()
            return False
        except sqlite3.OperationalError as e:
            logger.log_error("Store Posted Content - Database Operation Error", str(e))
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logger.log_error("Store Posted Content", str(e))
            if conn:
                conn.rollback()
            return False
    
    def check_if_post_replied(self, post_id: str, post_url: Optional[str] = None) -> bool:
        """
        Check if we've already replied to a post - NUMBA thread safe version
        
        Args:
            post_id: The ID of the post
            post_url: The URL of the post (optional)
        
        Returns:
            True if we've already replied to this post, False otherwise
        """
        conn = None
        try:
            # CRITICAL: Ensure table exists before attempting query
            self._ensure_replied_posts_table_exists()
            
            conn, cursor = self._get_connection()

            # Check for post_id first
            if post_id:
                cursor.execute("""
                    SELECT COUNT(*) FROM replied_posts
                    WHERE post_id = ?
                """, (post_id,))
                count = cursor.fetchone()[0]
                if count > 0:
                    logger.logger.debug(f"✅ Found existing reply for post_id: {post_id}")
                    return True
                
            # If post_url is provided and post_id check failed, try with URL
            if post_url:
                cursor.execute("""
                    SELECT COUNT(*) FROM replied_posts
                    WHERE post_url = ?
                """, (post_url,))
                count = cursor.fetchone()[0]
                if count > 0:
                    logger.logger.debug(f"✅ Found existing reply for post_url: {post_url}")
                    return True
                    
            logger.logger.debug(f"❌ No existing reply found for post_id: {post_id}, post_url: {post_url}")
            return False
            
        except Exception as e:
            logger.log_error("Check If Post Replied", str(e))
            logger.logger.error(f"❌ Error checking reply status for post {post_id}: {e}")
            # On error, assume we haven't replied (safer to potentially duplicate than miss)
            return False

    def store_mood(self, chain: str, mood: str, indicators: Dict) -> None:
        """Store mood data for a specific chain"""
        conn = None
        try:
            conn, cursor = self._get_connection()
            cursor.execute("""
                INSERT INTO mood_history (
                    timestamp, chain, mood, indicators
                ) VALUES (?, ?, ?, ?)
            """, (
                datetime.now(),
                chain,
                mood,
                json.dumps(indicators)  # indicators is already a Dict, no need for asdict()
            ))
            conn.commit()
        except Exception as e:
            logger.log_error(f"Store Mood - {chain}", str(e))
            if conn:
                conn.rollback()
            
    def store_smart_money_indicators(self, chain: str, indicators: Dict[str, Any]) -> None:
        """Store smart money indicators for a chain - NUMBA thread safe version"""
        
        # CRITICAL: Ensure table exists before attempting insert
        self._ensure_critical_tables_exist()
        
        conn, cursor = self._get_connection()
        try:
            # Extract values with defaults for potential missing keys
            volume_z_score = indicators.get('volume_z_score', 0.0)
            price_volume_divergence = 1 if indicators.get('price_volume_divergence', False) else 0
            stealth_accumulation = 1 if indicators.get('stealth_accumulation', False) else 0
            abnormal_volume = 1 if indicators.get('abnormal_volume', False) else 0
            volume_vs_hourly_avg = indicators.get('volume_vs_hourly_avg', 0.0)
            volume_vs_daily_avg = indicators.get('volume_vs_daily_avg', 0.0)
            volume_cluster_detected = 1 if indicators.get('volume_cluster_detected', False) else 0
            
            # Convert unusual_trading_hours to JSON if present
            unusual_hours = json.dumps(indicators.get('unusual_trading_hours', []))
            
            # Store all raw data for future reference
            raw_data = json.dumps(indicators)
            
            cursor.execute("""
                INSERT INTO smart_money_indicators (
                    timestamp, chain, volume_z_score, price_volume_divergence,
                    stealth_accumulation, abnormal_volume, volume_vs_hourly_avg,
                    volume_vs_daily_avg, volume_cluster_detected, unusual_trading_hours,
                    raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                chain,
                volume_z_score,
                price_volume_divergence,
                stealth_accumulation,
                abnormal_volume,
                volume_vs_hourly_avg,
                volume_vs_daily_avg,
                volume_cluster_detected,
                unusual_hours,
                raw_data
            ))
            conn.commit()
            logger.logger.debug(f"✅ Stored smart money indicators for {chain}")
        
        except Exception as e:
            logger.log_error(f"Store Smart Money Indicators - {chain}", str(e))
            logger.logger.error(f"❌ Failed to store smart money indicators for {chain}: {e}")
            if conn:
                conn.rollback()
            
    def _store_json_data(self, data_type: str, data: Dict[str, Any]) -> None:
        """Generic method to store JSON data in a generic_json_data table"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                INSERT INTO generic_json_data (
                    timestamp, data_type, data
                ) VALUES (?, ?, ?)
            """, (
                datetime.now(),
                data_type,
                json.dumps(data)
            ))
            conn.commit()
        except Exception as e:
            logger.log_error(f"Store JSON Data - {data_type}", str(e))
            conn.rollback()

    #########################
    # DATA RETRIEVAL METHODS
    #########################

    def get_recent_market_data(self, chain: str, hours: int = 24) -> List[Dict]:
        """Enhanced multi-source data retrieval"""
        return self.enhanced_data_system.get_recent_market_data(chain, hours)
            
    def get_token_correlations(self, token: str, hours: int = 24) -> List[Dict]:
        """Get token-specific correlation data"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM token_correlations 
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (token, hours))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON field
            for result in results:
                result["full_data"] = json.loads(result["full_data"]) if result["full_data"] else {}
                
            return results
        except Exception as e:
            logger.log_error(f"Get Token Correlations - {token}", str(e))
            return []
            
    def get_token_market_comparison(self, token: str, hours: int = 24) -> List[Dict]:
        """Get token vs market comparison data"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM token_market_comparison 
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (token, hours))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON field
            for result in results:
                result["correlations"] = json.loads(result["correlations"]) if result["correlations"] else {}
                
            return results
        except Exception as e:
            logger.log_error(f"Get Token Market Comparison - {token}", str(e))
            return []
        
    def get_recent_posts(self, hours: int = 24, timeframe: Optional[str] = None) -> List[Dict]:
        """
        Get recent posted content
        Can filter by timeframe
        """
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            query = """
                SELECT * FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params: List[Union[int, str]] = [hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for result in results:
                result["sentiment"] = json.loads(result["sentiment"]) if result["sentiment"] else {}
                result["price_data"] = json.loads(result["price_data"]) if result["price_data"] else {}
                result["meme_phrases"] = json.loads(result["meme_phrases"]) if result["meme_phrases"] else {}
                result["prediction_data"] = json.loads(result["prediction_data"]) if result["prediction_data"] else None
                
            return results
        except Exception as e:
            if conn:
                conn.rollback()
            logger.log_error("Get Recent Posts", str(e))
            return []

    def get_chain_stats(self, chain: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistical summary for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT 
                    AVG(price) as avg_price,
                    MAX(price) as max_price,
                    MIN(price) as min_price,
                    AVG(volume) as avg_volume,
                    MAX(volume) as max_volume,
                    AVG(price_change_24h) as avg_price_change
                FROM market_data 
                WHERE chain = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """, (chain, hours))
            result = cursor.fetchone()
            if result:
                return dict(result)
            return {}
        except Exception as e:
            logger.log_error(f"Get Chain Stats - {chain}", str(e))
            return {}
            
    def get_smart_money_indicators(self, chain: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent smart money indicators for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM smart_money_indicators
                WHERE chain = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (chain, hours))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for result in results:
                result["unusual_trading_hours"] = json.loads(result["unusual_trading_hours"]) if result["unusual_trading_hours"] else []
                result["raw_data"] = json.loads(result["raw_data"]) if result["raw_data"] else {}
                
            return results
        except Exception as e:
            logger.log_error(f"Get Smart Money Indicators - {chain}", str(e))
            return []
            
    def get_token_market_stats(self, token: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistical summary of token vs market performance"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT 
                    AVG(vs_market_avg_change) as avg_performance_diff,
                    AVG(vs_market_volume_growth) as avg_volume_growth_diff,
                    SUM(CASE WHEN outperforming_market = 1 THEN 1 ELSE 0 END) as outperforming_count,
                    COUNT(*) as total_records
                FROM token_market_comparison
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """, (token, hours))
            result = cursor.fetchone()
            if result:
                result_dict = dict(result)
                
                # Calculate percentage of time outperforming
                if result_dict['total_records'] > 0:
                    result_dict['outperforming_percentage'] = (result_dict['outperforming_count'] / result_dict['total_records']) * 100
                else:
                    result_dict['outperforming_percentage'] = 0
                    
                return result_dict
            return {}
        except Exception as e:
            logger.log_error(f"Get Token Market Stats - {token}", str(e))
            return {}

    def get_latest_smart_money_alert(self, chain: str) -> Optional[Dict[str, Any]]:
        """Get the most recent smart money alert for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT * FROM smart_money_indicators
                WHERE chain = ? 
                AND (abnormal_volume = 1 OR stealth_accumulation = 1 OR volume_cluster_detected = 1)
                ORDER BY timestamp DESC
                LIMIT 1
            """, (chain,))
            result = cursor.fetchone()
            if result:
                result_dict = dict(result)
                
                # Parse JSON fields
                result_dict["unusual_trading_hours"] = json.loads(result_dict["unusual_trading_hours"]) if result_dict["unusual_trading_hours"] else []
                result_dict["raw_data"] = json.loads(result_dict["raw_data"]) if result_dict["raw_data"] else {}
                
                return result_dict
            return None
        except Exception as e:
            logger.log_error(f"Get Latest Smart Money Alert - {chain}", str(e))
            return None
    
    def get_volume_trend(self, chain: str, hours: int = 24) -> Dict[str, Any]:
        """Get volume trend analysis for a chain"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT 
                    timestamp,
                    volume
                FROM market_data
                WHERE chain = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp ASC
            """, (chain, hours))
            
            results = cursor.fetchall()
            if not results:
                return {'trend': 'insufficient_data', 'change': 0}
                
            # Calculate trend
            volumes = [row['volume'] for row in results]
            earliest_volume = volumes[0] if volumes else 0
            latest_volume = volumes[-1] if volumes else 0
            
            if earliest_volume > 0:
                change_pct = ((latest_volume - earliest_volume) / earliest_volume) * 100
            else:
                change_pct = 0
                
            # Determine trend description
            if change_pct >= 15:
                trend = "significant_increase"
            elif change_pct <= -15:
                trend = "significant_decrease"
            elif change_pct >= 5:
                trend = "moderate_increase"
            elif change_pct <= -5:
                trend = "moderate_decrease"
            else:
                trend = "stable"
                
            return {
                'trend': trend,
                'change': change_pct,
                'earliest_volume': earliest_volume,
                'latest_volume': latest_volume,
                'data_points': len(volumes)
            }
            
        except Exception as e:
            logger.log_error(f"Get Volume Trend - {chain}", str(e))
            return {'trend': 'error', 'change': 0}
            
    def get_top_performing_tokens(self, hours: int = 24, limit: int = 5) -> List[Dict[str, Any]]:
        """Get list of top performing tokens based on price change"""
        conn, cursor = self._get_connection()
        try:
            # Get unique tokens in database
            cursor.execute("""
                SELECT DISTINCT chain
                FROM market_data
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """, (hours,))
            tokens = [row['chain'] for row in cursor.fetchall()]
            
            results = []
            for token in tokens:
                # Get latest price and 24h change
                cursor.execute("""
                    SELECT price, price_change_24h
                    FROM market_data
                    WHERE chain = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (token,))
                data = cursor.fetchone()
                
                if data:
                    results.append({
                        'token': token,
                        'price': data['price'],
                        'price_change_24h': data['price_change_24h']
                    })
            
            # Sort by price change (descending)
            results.sort(key=lambda x: x.get('price_change_24h', 0), reverse=True)
            
            # Return top N tokens
            return results[:limit]
            
        except Exception as e:
            logger.log_error("Get Top Performing Tokens", str(e))
            return []

    def get_tokens_by_prediction_accuracy(self, timeframe: str = "1h", min_predictions: int = 5) -> List[Dict[str, Any]]:
        """
        Get tokens sorted by prediction accuracy for a specific timeframe
        Only includes tokens with at least min_predictions number of predictions
        """
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                SELECT token, accuracy_rate, total_predictions, correct_predictions
                FROM prediction_performance
                WHERE timeframe = ? AND total_predictions >= ?
                ORDER BY accuracy_rate DESC
            """, (timeframe, min_predictions))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.log_error(f"Get Tokens By Prediction Accuracy - {timeframe}", str(e))
            return []

    #########################
    # DUPLICATE DETECTION METHODS
    #########################
    
    def check_content_similarity(self, content: str, timeframe: Optional[str] = None) -> bool:
        """
        Check if similar content was recently posted
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        try:
            query = """
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-1 hour')
            """
            
            params = []
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            cursor.execute(query, params)
            recent_posts = [row['content'] for row in cursor.fetchall()]
            
            # Simple similarity check - can be enhanced later
            return any(content.strip() == post.strip() for post in recent_posts)
        except Exception as e:
            logger.log_error("Check Content Similarity", str(e))
            return False
            
    def check_exact_content_match(self, content: str, timeframe: Optional[str] = None) -> bool:
        """
        Check for exact match of content within recent posts
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        try:
            query = """
                SELECT COUNT(*) as count FROM posted_content 
                WHERE content = ? 
                AND timestamp >= datetime('now', '-3 hours')
            """
            
            params = [content]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result['count'] > 0 if result else False
        except Exception as e:
            logger.log_error("Check Exact Content Match", str(e))
            return False
            
    def check_content_similarity_with_timeframe(self, content: str, hours: int = 1, timeframe: Optional[str] = None) -> bool:
        """
        Check if similar content was posted within a specified timeframe
        Can filter by prediction timeframe
        """
        conn = None
        try:
            conn, cursor = self._get_connection()
            
            query = """
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """
            
            params: List[Union[int, str]] = [hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            cursor.execute(query, params)
            recent_posts = [row['content'] for row in cursor.fetchall()]
            
            # Split content into main text and hashtags
            content_main = content.split("\n\n#")[0].lower() if "\n\n#" in content else content.lower()
            
            for post in recent_posts:
                post_main = post.split("\n\n#")[0].lower() if "\n\n#" in post else post.lower()
                
                # Calculate similarity based on word overlap
                content_words = set(content_main.split())
                post_words = set(post_main.split())
                
                if content_words and post_words:
                    overlap = len(content_words.intersection(post_words))
                    similarity = overlap / max(len(content_words), len(post_words))
                    
                    # Consider similar if 70% or more words overlap
                    if similarity > 0.7:
                        return True
            
            return False
        except Exception as e:
            if conn:
                conn.rollback()
            logger.log_error("Check Content Similarity With Timeframe", str(e))
            return False

    #########################
    # PREDICTION METHODS
    #########################
    
    def store_prediction(self, token: str, prediction_data: Dict[str, Any], timeframe: str = "1h") -> Optional[int]:
        """
        Store a prediction in the database
        Returns the ID of the inserted prediction, or None if failed
        """
        conn = None
        prediction_id = None
        
        try:
            conn, cursor = self._get_connection()
            
            # Extract prediction details
            prediction = prediction_data.get("prediction", {})
            rationale = prediction_data.get("rationale", "")
            sentiment = prediction_data.get("sentiment", "NEUTRAL")
            key_factors = json.dumps(prediction_data.get("key_factors", []))
            model_weights = json.dumps(prediction_data.get("model_weights", {}))
            model_inputs = json.dumps(prediction_data.get("inputs", {}))
            
            # Calculate expiration time based on timeframe
            if timeframe == "1h":
                expiration_time = datetime.now() + timedelta(hours=1)
            elif timeframe == "24h":
                expiration_time = datetime.now() + timedelta(hours=24)
            elif timeframe == "7d":
                expiration_time = datetime.now() + timedelta(days=7)
            else:
                expiration_time = datetime.now() + timedelta(hours=1)  # Default to 1h
                
            cursor.execute("""
                INSERT INTO price_predictions (
                    timestamp, token, timeframe, prediction_type,
                    prediction_value, confidence_level, lower_bound, upper_bound,
                    prediction_rationale, method_weights, model_inputs, technical_signals,
                    expiration_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    "price",
                    prediction["price"],
                    prediction["confidence"],
                    prediction["lower_bound"],
                    prediction["upper_bound"],
                    rationale,
                    model_weights,
                    model_inputs,
                    key_factors,
                    expiration_time
                ))
                
            conn.commit()
            prediction_id = cursor.lastrowid
            logger.logger.debug(f"Stored {timeframe} prediction for {token} with ID {prediction_id}")
            
            # Also store in specialized tables based on the prediction models used
            
            # Store Claude prediction if it was used
            if prediction_data.get("model_weights", {}).get("claude_enhanced", 0) > 0:
                self._store_claude_prediction(token, prediction_data, timeframe)
                
            # Store technical analysis if available
            if "inputs" in prediction_data and "technical_analysis" in prediction_data["inputs"]:
                self._store_technical_indicators(token, prediction_data["inputs"]["technical_analysis"], timeframe)
                
            # Store statistical forecast if available
            if "inputs" in prediction_data and "statistical_forecast" in prediction_data["inputs"]:
                self._store_statistical_forecast(token, prediction_data["inputs"]["statistical_forecast"], timeframe)
                
            # Store ML forecast if available
            if "inputs" in prediction_data and "ml_forecast" in prediction_data["inputs"]:
                self._store_ml_forecast(token, prediction_data["inputs"]["ml_forecast"], timeframe)
                
            # Update timeframe metrics
            self._update_timeframe_metrics(token, timeframe, prediction_data)
            
        except Exception as e:
            logger.log_error(f"Store Prediction - {token} ({timeframe})", str(e))
            if conn:
                conn.rollback()
            
        return prediction_id
    
    def _store_technical_indicators(self, token: str, technical_analysis: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store technical indicator data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract indicator values
            overall_trend = technical_analysis.get("overall_trend", "neutral")
            trend_strength = technical_analysis.get("trend_strength", 50)
            signals = technical_analysis.get("signals", {})
            
            # Extract individual indicators if available
            indicators = technical_analysis.get("indicators", {})
            
            # Get RSI
            rsi = indicators.get("rsi", None)
            
            # Get MACD
            macd = indicators.get("macd", {})
            macd_line = macd.get("macd_line", None)
            signal_line = macd.get("signal_line", None)
            histogram = macd.get("histogram", None)
            
            # Get Bollinger Bands
            bb = indicators.get("bollinger_bands", {})
            bb_upper = bb.get("upper", None)
            bb_middle = bb.get("middle", None)
            bb_lower = bb.get("lower", None)
            
            # Get Stochastic
            stoch = indicators.get("stochastic", {})
            stoch_k = stoch.get("k", None)
            stoch_d = stoch.get("d", None)
            
            # Get OBV
            obv = indicators.get("obv", None)
            
            # Get ADX
            adx = indicators.get("adx", None)
            
            # Get additional timeframe-specific indicators
            ichimoku_data = json.dumps(indicators.get("ichimoku", {}))
            pivot_points = json.dumps(indicators.get("pivot_points", {}))
            
            # Get volatility
            volatility = technical_analysis.get("volatility", None)
            
            # Store in database
            cursor.execute("""
                INSERT INTO technical_indicators (
                    timestamp, token, timeframe, rsi, macd_line, macd_signal, 
                    macd_histogram, bb_upper, bb_middle, bb_lower,
                    stoch_k, stoch_d, obv, adx, ichimoku_data, pivot_points,
                    overall_trend, trend_strength, volatility, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                rsi,
                macd_line,
                signal_line,
                histogram,
                bb_upper,
                bb_middle,
                bb_lower,
                stoch_k,
                stoch_d,
                obv,
                adx,
                ichimoku_data,
                pivot_points,
                overall_trend,
                trend_strength,
                volatility,
                json.dumps(technical_analysis)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Technical Indicators - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _store_statistical_forecast(self, token: str, forecast_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store statistical forecast data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract forecast and confidence intervals
            forecast_value = forecast_data.get("prediction", 0)
            confidence = forecast_data.get("confidence", [0, 0])
            
            # Get model type from model_info if available
            model_info = forecast_data.get("model_info", {})
            model_type = model_info.get("method", "ARIMA")
            
            # Extract model parameters if available
            model_parameters = json.dumps(model_info)
            
            # Store in database
            cursor.execute("""
                INSERT INTO statistical_forecasts (
                    timestamp, token, timeframe, model_type,
                    forecast_value, confidence_80_lower, confidence_80_upper,
                    confidence_95_lower, confidence_95_upper, 
                    model_parameters, input_data_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                model_type,
                forecast_value,
                confidence[0],  # 80% confidence lower
                confidence[1],  # 80% confidence upper
                confidence[0] * 0.9,  # Approximate 95% confidence lower
                confidence[1] * 1.1,  # Approximate 95% confidence upper
                model_parameters,
                "{}"   # Input data summary (empty for now)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Statistical Forecast - {token} ({timeframe})", str(e))
            conn.rollback()

    def _store_ml_forecast(self, token: str, forecast_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store machine learning forecast data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract forecast and confidence intervals
            forecast_value = forecast_data.get("prediction", 0)
            confidence = forecast_data.get("confidence", [0, 0])
            
            # Get model type and parameters if available
            model_info = forecast_data.get("model_info", {})
            model_type = model_info.get("method", "RandomForest")
            
            # Extract feature importance if available
            feature_importance = json.dumps(forecast_data.get("feature_importance", {}))
            
            # Store model parameters
            model_parameters = json.dumps(model_info)
            
            # Store in database
            cursor.execute("""
                INSERT INTO ml_forecasts (
                    timestamp, token, timeframe, model_type,
                    forecast_value, confidence_80_lower, confidence_80_upper,
                    confidence_95_lower, confidence_95_upper, 
                    feature_importance, model_parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                model_type,
                forecast_value,
                confidence[0],  # 80% confidence lower
                confidence[1],  # 80% confidence upper
                confidence[0] * 0.9,  # Approximate 95% confidence lower
                confidence[1] * 1.1,  # Approximate 95% confidence upper
                feature_importance,
                model_parameters
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store ML Forecast - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _store_claude_prediction(self, token: str, prediction_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store Claude AI prediction data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract prediction details
            prediction = prediction_data.get("prediction", {})
            rationale = prediction_data.get("rationale", "")
            sentiment = prediction_data.get("sentiment", "NEUTRAL")
            key_factors = json.dumps(prediction_data.get("key_factors", []))
            
            # Default Claude model
            claude_model = "claude-3-7-sonnet-20250219"
            
            # Store inputs if available
            input_data = json.dumps(prediction_data.get("inputs", {}))
            
            # Store in database
            cursor.execute("""
                INSERT INTO claude_predictions (
                    timestamp, token, timeframe, claude_model,
                    prediction_value, confidence_level, sentiment,
                    rationale, key_factors, input_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                claude_model,
                prediction.get("price", 0),
                prediction.get("confidence", 70),
                sentiment,
                rationale,
                key_factors,
                input_data
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Claude Prediction - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _update_timeframe_metrics(self, token: str, timeframe: str, prediction_data: Dict[str, Any]) -> None:
        """Update timeframe metrics based on new prediction"""
        conn, cursor = self._get_connection()
        
        try:
            # Get current metrics for this token and timeframe
            cursor.execute("""
                SELECT * FROM timeframe_metrics
                WHERE token = ? AND timeframe = ?
            """, (token, timeframe))
            
            metrics = cursor.fetchone()
            
            # Get prediction performance
            performance = self.get_prediction_performance(token=token, timeframe=timeframe)
            
            if performance:
                avg_accuracy = performance[0]["accuracy_rate"]
                total_count = performance[0]["total_predictions"]
                correct_count = performance[0]["correct_predictions"]
            else:
                avg_accuracy = 0
                total_count = 0
                correct_count = 0
            
            # Extract model weights
            model_weights = prediction_data.get("model_weights", {})
            
            # Determine best model
            if model_weights:
                best_model = max(model_weights.items(), key=lambda x: x[1])[0]
            else:
                best_model = "unknown"
            
            if metrics:
                # Update existing metrics
                cursor.execute("""
                    UPDATE timeframe_metrics
                    SET avg_accuracy = ?,
                        total_count = ?,
                        correct_count = ?,
                        model_weights = ?,
                        best_model = ?,
                        last_updated = ?
                    WHERE token = ? AND timeframe = ?
                """, (
                    avg_accuracy,
                    total_count,
                    correct_count,
                    json.dumps(model_weights),
                    best_model,
                    datetime.now(),
                    token,
                    timeframe
                ))
            else:
                # Insert new metrics
                cursor.execute("""
                    INSERT INTO timeframe_metrics (
                        timestamp, token, timeframe, avg_accuracy,
                        total_count, correct_count, model_weights,
                        best_model, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    avg_accuracy,
                    total_count,
                    correct_count,
                    json.dumps(model_weights),
                    best_model,
                    datetime.now()
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Update Timeframe Metrics - {token} ({timeframe})", str(e))
            conn.rollback()

    def get_active_predictions(self, token: Optional[str] = None, timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all active (non-expired) predictions
        Can filter by token and/or timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT * FROM price_predictions
                WHERE expiration_time > datetime('now')
            """
            params = []
            
            if token:
                query += " AND token = ?"
                params.append(token)
                
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            predictions = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for prediction in predictions:
                prediction["method_weights"] = json.loads(prediction["method_weights"]) if prediction["method_weights"] else {}
                prediction["model_inputs"] = json.loads(prediction["model_inputs"]) if prediction["model_inputs"] else {}
                prediction["technical_signals"] = json.loads(prediction["technical_signals"]) if prediction["technical_signals"] else []
                
            return predictions
            
        except Exception as e:
            logger.log_error("Get Active Predictions", str(e))
            return []

    def get_all_timeframe_predictions(self, token: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get active predictions for a token across all timeframes
        Returns a dictionary of predictions keyed by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            # Get all supported timeframes
            timeframes = ["1h", "24h", "7d"]
            
            result = {}
            
            for tf in timeframes:
                query = """
                    SELECT * FROM price_predictions
                    WHERE token = ? AND timeframe = ? AND expiration_time > datetime('now')
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                
                cursor.execute(query, (token, tf))
                prediction = cursor.fetchone()
                
                if prediction:
                    # Convert to dict and parse JSON fields
                    pred_dict = dict(prediction)
                    pred_dict["method_weights"] = json.loads(pred_dict["method_weights"]) if pred_dict["method_weights"] else {}
                    pred_dict["model_inputs"] = json.loads(pred_dict["model_inputs"]) if pred_dict["model_inputs"] else {}
                    pred_dict["technical_signals"] = json.loads(pred_dict["technical_signals"]) if pred_dict["technical_signals"] else []
                    
                    result[tf] = pred_dict
                
            return result
            
        except Exception as e:
            logger.log_error(f"Get All Timeframe Predictions - {token}", str(e))
            return {}

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Get a prediction by its ID"""
        conn, cursor = self._get_connection()
        
        try:
            cursor.execute("""
                SELECT * FROM price_predictions
                WHERE id = ?
            """, (prediction_id,))
            
            prediction = cursor.fetchone()
            if not prediction:
                return None
                
            # Convert to dict and parse JSON fields
            result = dict(prediction)
            result["method_weights"] = json.loads(result["method_weights"]) if result["method_weights"] else {}
            result["model_inputs"] = json.loads(result["model_inputs"]) if result["model_inputs"] else {}
            result["technical_signals"] = json.loads(result["technical_signals"]) if result["technical_signals"] else []
            
            return result
            
        except Exception as e:
            logger.log_error(f"Get Prediction By ID - {prediction_id}", str(e))
            return None

    def get_expired_unevaluated_predictions(self, timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all expired predictions that haven't been evaluated yet
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT p.* FROM price_predictions p
                LEFT JOIN prediction_outcomes o ON p.id = o.prediction_id
                WHERE p.expiration_time <= datetime('now')
                AND o.id IS NULL
            """
            
            params = []
            
            if timeframe:
                query += " AND p.timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY p.timeframe ASC, p.expiration_time ASC"
            
            cursor.execute(query, params)
            
            predictions = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for prediction in predictions:
                prediction["method_weights"] = json.loads(prediction["method_weights"]) if prediction["method_weights"] else {}
                prediction["model_inputs"] = json.loads(prediction["model_inputs"]) if prediction["model_inputs"] else {}
                prediction["technical_signals"] = json.loads(prediction["technical_signals"]) if prediction["technical_signals"] else []
                
            return predictions
            
        except Exception as e:
            logger.log_error("Get Expired Unevaluated Predictions", str(e))
            return []

    def record_prediction_outcome(self, prediction_id: int, actual_price: float) -> bool:
        """Record the outcome of a prediction"""
        conn, cursor = self._get_connection()
        
        try:
            # Get the prediction details
            prediction = self.get_prediction_by_id(prediction_id)
            if not prediction:
                return False
                
            # Calculate accuracy metrics
            prediction_value = prediction["prediction_value"]
            lower_bound = prediction["lower_bound"]
            upper_bound = prediction["upper_bound"]
            timeframe = prediction["timeframe"]
            
            # Percentage accuracy (how close the prediction was)
            price_diff = abs(actual_price - prediction_value)
            accuracy_percentage = (1 - (price_diff / prediction_value)) * 100 if prediction_value > 0 else 0
            
            # Whether the actual price fell within the predicted range
            was_correct = lower_bound <= actual_price <= upper_bound
            
            # Deviation from prediction (for tracking bias)
            deviation = ((actual_price / prediction_value) - 1) * 100 if prediction_value > 0 else 0
            
            # Get market conditions at evaluation time
            market_data = self.get_recent_market_data(prediction["token"], 1)

            # Safely serialize the data, converting datetime objects to ISO strings
            safe_market_data = serialize_datetime_objects(market_data[:1] if market_data else [])

            market_conditions = json.dumps({
                "evaluation_time": datetime.now().isoformat(),
                "token": prediction["token"],
                "market_data": safe_market_data
            })
            
            # Store the outcome
            cursor.execute("""
                INSERT INTO prediction_outcomes (
                    prediction_id, actual_outcome, accuracy_percentage,
                    was_correct, evaluation_time, deviation_from_prediction,
                    market_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                actual_price,
                accuracy_percentage,
                1 if was_correct else 0,
                datetime.now(),
                deviation,
                market_conditions
            ))
            
            # Update the performance summary
            token = prediction["token"]
            prediction_type = prediction["prediction_type"]
            
            self._update_prediction_performance(token, timeframe, prediction_type, was_correct, abs(deviation))
            
            # Update timeframe metrics
            self._update_timeframe_outcome_metrics(token, timeframe, was_correct, accuracy_percentage)
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.log_error(f"Record Prediction Outcome - {prediction_id}", str(e))
            conn.rollback()
            return False

    def _update_prediction_performance(self, token: str, timeframe: str, prediction_type: str, was_correct: bool, deviation: float) -> None:
        """Update prediction performance summary"""
        conn, cursor = self._get_connection()
        
        try:
            # Check if performance record exists
            cursor.execute("""
                SELECT * FROM prediction_performance
                WHERE token = ? AND timeframe = ? AND prediction_type = ?
            """, (token, timeframe, prediction_type))
            
            performance = cursor.fetchone()
            
            if performance:
                # Update existing record
                performance_dict = dict(performance)
                total_predictions = performance_dict["total_predictions"] + 1
                correct_predictions = performance_dict["correct_predictions"] + (1 if was_correct else 0)
                accuracy_rate = (correct_predictions / total_predictions) * 100
                
                # Update average deviation (weighted average)
                avg_deviation = (performance_dict["avg_deviation"] * performance_dict["total_predictions"] + deviation) / total_predictions
                
                cursor.execute("""
                    UPDATE prediction_performance
                    SET total_predictions = ?,
                        correct_predictions = ?,
                        accuracy_rate = ?,
                        avg_deviation = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    total_predictions,
                    correct_predictions,
                    accuracy_rate,
                    avg_deviation,
                    datetime.now(),
                    performance_dict["id"]
                ))
                
            else:
                # Create new record
                cursor.execute("""
                    INSERT INTO prediction_performance (
                        token, timeframe, prediction_type, total_predictions,
                        correct_predictions, accuracy_rate, avg_deviation, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    token,
                    timeframe,
                    prediction_type,
                    1,
                    1 if was_correct else 0,
                    100 if was_correct else 0,
                    deviation,
                    datetime.now()
                ))
                
        except Exception as e:
            logger.log_error(f"Update Prediction Performance - {token}", str(e))
            raise

    def _update_timeframe_outcome_metrics(self, token: str, timeframe: str, was_correct: bool, accuracy_percentage: float) -> None:
        """Update timeframe metrics with outcome data"""
        conn, cursor = self._get_connection()
        
        try:
            # Check if metrics record exists
            cursor.execute("""
                SELECT * FROM timeframe_metrics
                WHERE token = ? AND timeframe = ?
            """, (token, timeframe))
            
            metrics = cursor.fetchone()
            
            if metrics:
                # Update existing metrics
                metrics_dict = dict(metrics)
                total_count = metrics_dict["total_count"] + 1
                correct_count = metrics_dict["correct_count"] + (1 if was_correct else 0)
                
                # Recalculate average accuracy with new data point
                # Use weighted average based on number of predictions
                old_weight = (total_count - 1) / total_count
                new_weight = 1 / total_count
                avg_accuracy = (metrics_dict["avg_accuracy"] * old_weight) + (accuracy_percentage * new_weight)
                
                cursor.execute("""
                    UPDATE timeframe_metrics
                    SET avg_accuracy = ?,
                        total_count = ?,
                        correct_count = ?,
                        last_updated = ?
                    WHERE token = ? AND timeframe = ?
                """, (
                    avg_accuracy,
                    total_count,
                    correct_count,
                    datetime.now(),
                    token,
                    timeframe
                ))
            else:
                # Should not happen normally, but create metrics if missing
                cursor.execute("""
                    INSERT INTO timeframe_metrics (
                        timestamp, token, timeframe, avg_accuracy,
                        total_count, correct_count, model_weights,
                        best_model, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    accuracy_percentage,
                    1,
                    1 if was_correct else 0,
                    "{}",
                    "unknown",
                    datetime.now()
                ))
                
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Update Timeframe Outcome Metrics - {token} ({timeframe})", str(e))
            conn.rollback()

    def get_prediction_performance(self, token: Optional[str] = None, timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get prediction performance statistics
        Can filter by token and/or timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = "SELECT * FROM prediction_performance"
            params = []
            
            if token or timeframe:
                query += " WHERE "
                
            if token:
                query += "token = ?"
                params.append(token)
                
            if token and timeframe:
                query += " AND "
                
            if timeframe:
                query += "timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY updated_at DESC"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.log_error("Get Prediction Performance", str(e))
            return []

    def get_timeframe_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance summary across all timeframes
        Returns a dictionary with metrics for each timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            # Get performance for each timeframe across all tokens
            timeframes = ["1h", "24h", "7d"]
            result = {}
            
            for tf in timeframes:
                cursor.execute("""
                    SELECT 
                        AVG(accuracy_rate) as avg_accuracy,
                        SUM(total_predictions) as total_predictions,
                        SUM(correct_predictions) as correct_predictions,
                        AVG(avg_deviation) as avg_deviation
                    FROM prediction_performance
                    WHERE timeframe = ?
                """, (tf,))
                
                stats = cursor.fetchone()
                
                if stats:
                    stats_dict = dict(stats)
                    
                    # Calculate overall accuracy
                    total = stats_dict["total_predictions"] or 0
                    correct = stats_dict["correct_predictions"] or 0
                    accuracy = (correct / total * 100) if total > 0 else 0
                    
                    result[tf] = {
                        "accuracy": accuracy,
                        "total_predictions": total,
                        "correct_predictions": correct,
                        "avg_deviation": stats_dict["avg_deviation"] or 0
                    }
                    
                    # Get best performing token for this timeframe
                    cursor.execute("""
                        SELECT token, accuracy_rate, total_predictions
                        FROM prediction_performance
                        WHERE timeframe = ? AND total_predictions >= 5
                        ORDER BY accuracy_rate DESC
                        LIMIT 1
                    """, (tf,))
                    
                    best_token = cursor.fetchone()
                    if best_token:
                        result[tf]["best_token"] = {
                            "token": best_token["token"],
                            "accuracy": best_token["accuracy_rate"],
                            "predictions": best_token["total_predictions"]
                        }
                    
                    # Get worst performing token for this timeframe
                    cursor.execute("""
                        SELECT token, accuracy_rate, total_predictions
                        FROM prediction_performance
                        WHERE timeframe = ? AND total_predictions >= 5
                        ORDER BY accuracy_rate ASC
                        LIMIT 1
                    """, (tf,))
                    
                    worst_token = cursor.fetchone()
                    if worst_token:
                        result[tf]["worst_token"] = {
                            "token": worst_token["token"],
                            "accuracy": worst_token["accuracy_rate"],
                            "predictions": worst_token["total_predictions"]
                        }
            
            return result
            
        except Exception as e:
            logger.log_error("Get Timeframe Performance Summary", str(e))
            return {}

    def get_recent_prediction_outcomes(self, token: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent prediction outcomes with their original predictions
        Can filter by token and/or timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT p.*, o.actual_outcome, o.accuracy_percentage, o.was_correct, 
                       o.evaluation_time, o.deviation_from_prediction
                FROM prediction_outcomes o
                JOIN price_predictions p ON o.prediction_id = p.id
                WHERE 1=1
            """
            params = []
            
            if token:
                query += " AND p.token = ?"
                params.append(token)
                
            query += " ORDER BY o.evaluation_time DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            outcomes = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for outcome in outcomes:
                outcome["method_weights"] = json.loads(outcome["method_weights"]) if outcome["method_weights"] else {}
                outcome["model_inputs"] = json.loads(outcome["model_inputs"]) if outcome["model_inputs"] else {}
                outcome["technical_signals"] = json.loads(outcome["technical_signals"]) if outcome["technical_signals"] else []
                
            return outcomes
            
        except Exception as e:
            logger.log_error("Get Recent Prediction Outcomes", str(e))
            return []

    def get_timeframe_metrics(self, token: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for different timeframes
        Returns a dictionary with metrics for each timeframe, optionally filtered by token
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT * FROM timeframe_metrics
                WHERE 1=1
            """
            params = []
            
            if token:
                query += " AND token = ?"
                params.append(token)
                
            query += " ORDER BY token, timeframe"
            
            cursor.execute(query, params)
            metrics = cursor.fetchall()
            
            result = {}
            
            for metric in metrics:
                metric_dict = dict(metric)
                timeframe = metric_dict["timeframe"]
                
                # Parse JSON fields
                metric_dict["model_weights"] = json.loads(metric_dict["model_weights"]) if metric_dict["model_weights"] else {}
                
                if token:
                    # If filtering by token, return metrics keyed by timeframe
                    result[timeframe] = metric_dict
                else:
                    # If not filtering by token, organize by token then timeframe
                    token_name = metric_dict["token"]
                    if token_name not in result:
                        result[token_name] = {}
                        
                    result[token_name][timeframe] = metric_dict
            
            return result
            
        except Exception as e:
            logger.log_error("Get Timeframe Metrics", str(e))
            return {}
            
    def get_technical_indicators(self, token: str, timeframe: str = "1h", hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent technical indicators for a token and timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            cursor.execute("""
                SELECT * FROM technical_indicators
                WHERE token = ? AND timeframe = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (token, timeframe, hours))
            
            indicators = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for indicator in indicators:
                indicator["raw_data"] = json.loads(indicator["raw_data"]) if indicator["raw_data"] else {}
                indicator["ichimoku_data"] = json.loads(indicator["ichimoku_data"]) if indicator["ichimoku_data"] else {}
                indicator["pivot_points"] = json.loads(indicator["pivot_points"]) if indicator["pivot_points"] else {}
                
            return indicators
            
        except Exception as e:
            logger.log_error(f"Get Technical Indicators - {token} ({timeframe})", str(e))
            return []
            
    def get_statistical_forecasts(self, token: str, timeframe: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent statistical forecasts for a token
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT * FROM statistical_forecasts
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params = [token, hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            forecasts = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for forecast in forecasts:
                forecast["model_parameters"] = json.loads(forecast["model_parameters"]) if forecast["model_parameters"] else {}
                forecast["input_data_summary"] = json.loads(forecast["input_data_summary"]) if forecast["input_data_summary"] else {}
                
            return forecasts
            
        except Exception as e:
            logger.log_error(f"Get Statistical Forecasts - {token}", str(e))
            return []
            
    def get_ml_forecasts(self, token: str, timeframe: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent machine learning forecasts for a token
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT * FROM ml_forecasts
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params = [token, hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            forecasts = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for forecast in forecasts:
                forecast["feature_importance"] = json.loads(forecast["feature_importance"]) if forecast["feature_importance"] else {}
                forecast["model_parameters"] = json.loads(forecast["model_parameters"]) if forecast["model_parameters"] else {}
                
            return forecasts
            
        except Exception as e:
            logger.log_error(f"Get ML Forecasts - {token}", str(e))
            return []
            
    def get_claude_predictions(self, token: str, timeframe: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent Claude AI predictions for a token
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT * FROM claude_predictions
                WHERE token = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params = [token, hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            predictions = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for prediction in predictions:
                prediction["key_factors"] = json.loads(prediction["key_factors"]) if prediction["key_factors"] else []
                prediction["input_data"] = json.loads(prediction["input_data"]) if prediction["input_data"] else {}
                
            return predictions
            
        except Exception as e:
            logger.log_error(f"Get Claude Predictions - {token}", str(e))
            return []

    def get_prediction_accuracy_by_model(self, timeframe: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
            """
            Calculate prediction accuracy statistics by model type
            Returns accuracy metrics for different prediction approaches
            Can filter by timeframe
            """
            conn, cursor = self._get_connection()
            
            try:
                # Base query for predictions and outcomes
                query = """
                    SELECT p.id, p.token, p.timeframe, p.method_weights, 
                            o.was_correct, o.deviation_from_prediction
                    FROM price_predictions p
                    JOIN prediction_outcomes o ON p.id = o.prediction_id
                    WHERE p.timestamp >= datetime('now', '-' || ? || ' days')
                """
                params = []
                params.append(days)
                
                if timeframe:
                    query += " AND p.timeframe = ?"
                    params.append(timeframe)
                    
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Initialize counters for each model type
                model_stats = {
                    "technical_analysis": {"correct": 0, "total": 0, "deviation_sum": 0},
                    "statistical_models": {"correct": 0, "total": 0, "deviation_sum": 0},
                    "machine_learning": {"correct": 0, "total": 0, "deviation_sum": 0},
                    "claude_enhanced": {"correct": 0, "total": 0, "deviation_sum": 0},
                    "combined": {"correct": 0, "total": 0, "deviation_sum": 0}
                }
                
                # Add timeframe-specific counters
                timeframe_stats = {}
                
                # Process results
                for row in results:
                    # Parse model weights
                    weights = json.loads(row["method_weights"]) if row["method_weights"] else {}
                    was_correct = row["was_correct"] == 1
                    deviation = abs(row["deviation_from_prediction"])
                    row_timeframe = row["timeframe"]
                    
                    # Update combined stats
                    model_stats["combined"]["total"] += 1
                    if was_correct:
                        model_stats["combined"]["correct"] += 1
                    model_stats["combined"]["deviation_sum"] += deviation
                    
                    # Update timeframe stats
                    if row_timeframe not in timeframe_stats:
                        timeframe_stats[row_timeframe] = {"correct": 0, "total": 0, "deviation_sum": 0}
                    
                    timeframe_stats[row_timeframe]["total"] += 1
                    if was_correct:
                        timeframe_stats[row_timeframe]["correct"] += 1
                    timeframe_stats[row_timeframe]["deviation_sum"] += deviation
                    
                    # Determine primary model based on weights
                    if weights:
                        primary_model = max(weights.items(), key=lambda x: x[1])[0]
                        
                        # Update model-specific stats
                        if primary_model in model_stats:
                            model_stats[primary_model]["total"] += 1
                            if was_correct:
                                model_stats[primary_model]["correct"] += 1
                            model_stats[primary_model]["deviation_sum"] += deviation
                        
                        # Update stats for all models used in this prediction
                        for model, weight in weights.items():
                            if model in model_stats and weight > 0:
                                # Add fractional count based on weight
                                model_stats[model]["total"] += weight
                                if was_correct:
                                    model_stats[model]["correct"] += weight
                                model_stats[model]["deviation_sum"] += deviation * weight
                
                # Calculate accuracy rates and average deviations
                model_results = {}
                for model, stats in model_stats.items():
                    if stats["total"] > 0:
                        accuracy = (stats["correct"] / stats["total"]) * 100
                        avg_deviation = stats["deviation_sum"] / stats["total"]
                        
                        model_results[model] = {
                            "accuracy_rate": accuracy,
                            "avg_deviation": avg_deviation,
                            "total_predictions": stats["total"]
                        }
                
                # Calculate timeframe statistics
                tf_results = {}
                for tf, stats in timeframe_stats.items():
                    if stats["total"] > 0:
                        accuracy = (stats["correct"] / stats["total"]) * 100
                        avg_deviation = stats["deviation_sum"] / stats["total"]
                        
                        tf_results[tf] = {
                            "accuracy_rate": accuracy,
                            "avg_deviation": avg_deviation,
                            "total_predictions": stats["total"]
                        }
                
                # Combine results
                return {
                    "models": model_results,
                    "timeframes": tf_results,
                    "total_predictions": model_stats["combined"]["total"],
                    "overall_accuracy": (model_stats["combined"]["correct"] / model_stats["combined"]["total"] * 100) 
                                        if model_stats["combined"]["total"] > 0 else 0
                }
                
            except Exception as e:
                logger.log_error("Get Prediction Accuracy By Model", str(e))
                return {}
    
    def get_prediction_comparison_across_timeframes(self, token: str, limit: int = 5) -> Dict[str, Any]:
        """
        Compare prediction performance across different timeframes for a specific token
        Returns latest predictions and their outcomes for each timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            # Get performance summary for each timeframe
            timeframes = ["1h", "24h", "7d"]
            result = {
                "summary": {},
                "recent_predictions": {}
            }
            
            # Get performance stats for each timeframe
            for tf in timeframes:
                cursor.execute("""
                    SELECT * FROM prediction_performance
                    WHERE token = ? AND timeframe = ?
                """, (token, tf))
                
                performance = cursor.fetchone()
                
                if performance:
                    perf_dict = dict(performance)
                    result["summary"][tf] = {
                        "accuracy": perf_dict["accuracy_rate"],
                        "total_predictions": perf_dict["total_predictions"],
                        "correct_predictions": perf_dict["correct_predictions"],
                        "avg_deviation": perf_dict["avg_deviation"]
                    }
                
                # Get recent predictions for this timeframe
                cursor.execute("""
                    SELECT p.*, o.actual_outcome, o.was_correct, o.deviation_from_prediction
                    FROM price_predictions p
                    LEFT JOIN prediction_outcomes o ON p.id = o.prediction_id
                    WHERE p.token = ? AND p.timeframe = ?
                    ORDER BY p.timestamp DESC
                    LIMIT ?
                """, (token, tf, limit))
                
                predictions = [dict(row) for row in cursor.fetchall()]
                
                # Parse JSON fields
                for pred in predictions:
                    pred["method_weights"] = json.loads(pred["method_weights"]) if pred["method_weights"] else {}
                    pred["technical_signals"] = json.loads(pred["technical_signals"]) if pred["technical_signals"] else []
                
                result["recent_predictions"][tf] = predictions
            
            # Add overall statistics
            if result["summary"]:
                total_correct = sum(tf_stats.get("correct_predictions", 0) for tf_stats in result["summary"].values())
                total_predictions = sum(tf_stats.get("total_predictions", 0) for tf_stats in result["summary"].values())
                
                if total_predictions > 0:
                    overall_accuracy = (total_correct / total_predictions) * 100
                else:
                    overall_accuracy = 0
                    
                result["overall"] = {
                    "accuracy": overall_accuracy,
                    "total_predictions": total_predictions,
                    "total_correct": total_correct
                }
                
                # Find best timeframe for this token
                best_timeframe = max(result["summary"].items(), key=lambda x: x[1]["accuracy"])
                result["best_timeframe"] = {
                    "timeframe": best_timeframe[0],
                    "accuracy": best_timeframe[1]["accuracy"]
                }
            
            return result
            
        except Exception as e:
            logger.log_error(f"Get Prediction Comparison Across Timeframes - {token}", str(e))
            return {}

    #########################
    # DATABASE MAINTENANCE METHODS
    #########################
            
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Clean up old data to prevent database bloat
        Returns count of deleted records by table
        """
        conn, cursor = self._get_connection()
        
        tables_to_clean = [
            "market_data",
            "posted_content",
            "mood_history",
            "smart_money_indicators",
            "token_market_comparison",
            "token_correlations",
            "generic_json_data",
            "technical_indicators",
            "statistical_forecasts",
            "ml_forecasts",
            "claude_predictions"
        ]
        
        deleted_counts = {}
        
        try:
            for table in tables_to_clean:
                # Keep prediction-related tables longer
                retention_days = days_to_keep * 2 if table in ["price_predictions", "prediction_outcomes"] else days_to_keep
                
                cursor.execute(f"""
                    DELETE FROM {table}
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                """, (retention_days,))
                
                deleted_counts[table] = cursor.rowcount
                
            # Special handling for evaluated predictions
            cursor.execute("""
                DELETE FROM price_predictions
                WHERE id IN (
                    SELECT p.id
                    FROM price_predictions p
                    JOIN prediction_outcomes o ON p.id = o.prediction_id
                    WHERE p.timestamp < datetime('now', '-' || ? || ' days')
                )
            """, (days_to_keep * 2,))
            
            deleted_counts["price_predictions"] = cursor.rowcount
            
            conn.commit()
            logger.logger.info(f"Database cleanup completed: {deleted_counts}")
            
            return deleted_counts
            
        except Exception as e:
            logger.log_error("Database Cleanup", str(e))
            conn.rollback()
            return {}
            
    def optimize_database(self) -> bool:
        """
        Optimize database performance by running VACUUM and ANALYZE
        """
        conn, cursor = self._get_connection()
        
        try:
            # Backup current connection settings
            old_isolation_level = conn.isolation_level
            
            # Set isolation level to None for VACUUM
            conn.isolation_level = None
            
            # Run VACUUM to reclaim space
            cursor.execute("VACUUM")
            
            # Run ANALYZE to update statistics
            cursor.execute("ANALYZE")
            
            # Restore original isolation level
            conn.isolation_level = old_isolation_level
            
            logger.logger.info("Database optimization completed")
            return True
            
        except Exception as e:
            logger.log_error("Database Optimization", str(e))
            return False
            
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics including table sizes and row counts
        """
        conn, cursor = self._get_connection()
        
        try:
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row["name"] for row in cursor.fetchall()]
            
            stats = {
                "tables": {},
                "total_rows": 0,
                "last_optimized": None
            }
            
            # Get row count for each table
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                row_count = cursor.fetchone()["count"]
                
                # Get most recent timestamp if available
                try:
                    cursor.execute(f"SELECT MAX(timestamp) as last_update FROM {table}")
                    last_update = cursor.fetchone()["last_update"]
                except:
                    last_update = None
                
                stats["tables"][table] = {
                    "rows": row_count,
                    "last_update": last_update
                }
                
                stats["total_rows"] += row_count
                
            # Get database size (approximate)
            stats["database_size_kb"] = os.path.getsize(self.db_path) / 1024
            
            # Get last VACUUM time (if available in generic_json_data)
            cursor.execute("""
                SELECT timestamp FROM generic_json_data
                WHERE data_type = 'database_maintenance'
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            last_maintenance = cursor.fetchone()
            if last_maintenance:
                stats["last_optimized"] = last_maintenance["timestamp"]
                
            return stats
            
        except Exception as e:
            logger.log_error("Get Database Stats", str(e))
            return {"error": str(e)}
            
    def get_timeframe_prediction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of predictions and accuracy across all timeframes
        """
        conn, cursor = self._get_connection()
        
        try:
            summary = {
                "timeframes": {},
                "total": {
                    "predictions": 0,
                    "correct": 0,
                    "accuracy": 0
                }
            }
            
            # Get stats for each timeframe
            for timeframe in ["1h", "24h", "7d"]:
                # Get overall stats
                cursor.execute("""
                    SELECT 
                        SUM(total_predictions) as total,
                        SUM(correct_predictions) as correct
                    FROM prediction_performance
                    WHERE timeframe = ?
                """, (timeframe,))
                
                stats = cursor.fetchone()
                
                if stats and stats["total"]:
                    total = stats["total"]
                    correct = stats["correct"]
                    accuracy = (correct / total * 100) if total > 0 else 0
                    
                    summary["timeframes"][timeframe] = {
                        "predictions": total,
                        "correct": correct,
                        "accuracy": accuracy
                    }
                    
                    # Get top performing token
                    cursor.execute("""
                        SELECT token, accuracy_rate
                        FROM prediction_performance
                        WHERE timeframe = ? AND total_predictions >= 5
                        ORDER BY accuracy_rate DESC
                        LIMIT 1
                    """, (timeframe,))
                    
                    best = cursor.fetchone()
                    if best:
                        summary["timeframes"][timeframe]["best_token"] = {
                            "token": best["token"],
                            "accuracy": best["accuracy_rate"]
                        }
                        
                    # Update totals
                    summary["total"]["predictions"] += total
                    summary["total"]["correct"] += correct
            
            # Calculate overall accuracy
            if summary["total"]["predictions"] > 0:
                summary["total"]["accuracy"] = (summary["total"]["correct"] / summary["total"]["predictions"]) * 100
                
            # Add prediction counts by timeframe
            cursor.execute("""
                SELECT timeframe, COUNT(*) as count
                FROM price_predictions
                GROUP BY timeframe
            """)
            
            counts = cursor.fetchall()
            for row in counts:
                tf = row["timeframe"]
                if tf in summary["timeframes"]:
                    summary["timeframes"][tf]["total_stored"] = row["count"]
                    
            # Add active prediction counts
            cursor.execute("""
                SELECT timeframe, COUNT(*) as count
                FROM price_predictions
                WHERE expiration_time > datetime('now')
                GROUP BY timeframe
            """)
            
            active_counts = cursor.fetchall()
            for row in active_counts:
                tf = row["timeframe"]
                if tf in summary["timeframes"]:
                    summary["timeframes"][tf]["active"] = row["count"]
                    
            return summary
            
        except Exception as e:
            logger.log_error("Get Timeframe Prediction Summary", str(e))
            return {}
    
class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies for database operations"""
    UPSERT = "UPSERT"           # INSERT OR REPLACE (default)
    IGNORE = "IGNORE"           # INSERT OR IGNORE  
    RETRY = "RETRY"             # Wait and retry
    UPDATE_EXISTING = "UPDATE"  # Update existing record
    FAIL = "FAIL"               # Let the error propagate

class DatabaseOperationManager:
    """
    Enterprise operation manager for audit trails, conflict resolution, and retry logic
    Integrates with existing CryptoDatabase class methods
    """
    
    def __init__(self, database_instance):
        """
        Initialize the operation manager
        
        Args:
            database_instance: CryptoDatabase instance to wrap
        """
        self.db = database_instance
        self.logger = database_instance.logger if hasattr(database_instance, 'logger') else logger
        
        # Load retry policies from database
        self.retry_policies = self._load_retry_policies()
        
        # Load conflict resolution strategies from config
        self.conflict_strategies = self._load_conflict_strategies()
        
    def _load_retry_policies(self) -> Dict[str, Dict[str, Any]]:
        """Load retry policies from database"""
        try:
            conn, cursor = self.db._get_connection()
            
            cursor.execute("""
                SELECT policy_name, table_name, max_retries, backoff_strategy,
                       base_delay_ms, max_delay_ms, retry_on_constraint_violation,
                       retry_on_lock_timeout, retry_on_connection_error, enabled
                FROM retry_policies
                WHERE enabled = TRUE
            """)
            
            policies = {}
            for row in cursor.fetchall():
                policies[row['table_name']] = {
                    'max_retries': row['max_retries'],
                    'backoff_strategy': row['backoff_strategy'],
                    'base_delay_ms': row['base_delay_ms'],
                    'max_delay_ms': row['max_delay_ms'],
                    'retry_on_constraint_violation': bool(row['retry_on_constraint_violation']),
                    'retry_on_lock_timeout': bool(row['retry_on_lock_timeout']),
                    'retry_on_connection_error': bool(row['retry_on_connection_error'])
                }
            
            logger.logger.debug(f"Loaded retry policies for {len(policies)} tables")
            return policies
            
        except Exception as e:
            logger.logger.warning(f"Could not load retry policies: {e}")
            return self._get_default_retry_policies()
    
    def _load_conflict_strategies(self) -> Dict[str, ConflictResolutionStrategy]:
        """Load conflict resolution strategies from system config"""
        try:
            strategies = {}
            
            # Get config values for different tables
            config_mappings = {
                'content_analysis': 'conflict_resolution_content_analysis',
                'replied_posts': 'conflict_resolution_replied_posts', 
                'posted_content': 'conflict_resolution_posted_content',
                'market_data': 'conflict_resolution_market_data',
                'reply_restrictions': 'conflict_resolution_reply_restrictions'
            }
            
            for table, config_key in config_mappings.items():
                strategy_name = self.db.get_enterprise_config(config_key)
                if strategy_name and hasattr(ConflictResolutionStrategy, strategy_name):
                    strategies[table] = ConflictResolutionStrategy[strategy_name]
                else:
                    strategies[table] = ConflictResolutionStrategy.UPSERT  # Default
            
            logger.logger.debug(f"Loaded conflict strategies for {len(strategies)} tables")
            return strategies
            
        except Exception as e:
            logger.logger.warning(f"Could not load conflict strategies: {e}")
            return self._get_default_conflict_strategies()
    
    def _get_default_retry_policies(self) -> Dict[str, Dict[str, Any]]:
        """Default retry policies if database loading fails"""
        return {
            'content_analysis': {
                'max_retries': 3,
                'backoff_strategy': 'EXPONENTIAL',
                'base_delay_ms': 100,
                'max_delay_ms': 2000,
                'retry_on_constraint_violation': True,
                'retry_on_lock_timeout': True,
                'retry_on_connection_error': True
            },
            'replied_posts': {
                'max_retries': 2,
                'backoff_strategy': 'LINEAR',
                'base_delay_ms': 200,
                'max_delay_ms': 1000,
                'retry_on_constraint_violation': True,
                'retry_on_lock_timeout': True,
                'retry_on_connection_error': True
            },
            'reply_restrictions': {
                'max_retries': 2,
                'backoff_strategy': 'LINEAR',
                'base_delay_ms': 100,
                'max_delay_ms': 1000,
                'retry_on_constraint_violation': True,
                'retry_on_lock_timeout': True,
                'retry_on_connection_error': True
            }
        }
    
    def _get_default_conflict_strategies(self) -> Dict[str, ConflictResolutionStrategy]:
        """Default conflict resolution strategies"""
        return {
            'content_analysis': ConflictResolutionStrategy.UPSERT,
            'replied_posts': ConflictResolutionStrategy.UPSERT,
            'posted_content': ConflictResolutionStrategy.RETRY,
            'market_data': ConflictResolutionStrategy.IGNORE,
            'reply_restrictions': ConflictResolutionStrategy.UPSERT
        }
    
    def execute_with_audit(self, operation_func: Callable, table_name: str, 
                          record_key: str, *args, **kwargs) -> Any:
        """
        Execute a database operation with full audit trail and enterprise features
        
        Args:
            operation_func: The database operation function to execute
            table_name: Name of the table being operated on
            record_key: Key identifying the record (for conflict resolution)
            *args, **kwargs: Arguments to pass to the operation function
            
        Returns:
            Result of the operation function
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Get retry policy for this table
        retry_policy = self.retry_policies.get(table_name, self.retry_policies.get('content_analysis', {}))
        max_retries = retry_policy.get('max_retries', 3)
        
        # Get conflict resolution strategy
        conflict_strategy = self.conflict_strategies.get(table_name, ConflictResolutionStrategy.UPSERT)
        
        attempt_count = 0
        last_error = None
        
        while attempt_count <= max_retries:
            attempt_count += 1
            
            try:
                # Log operation start
                self._log_operation_start(operation_id, table_name, record_key, attempt_count)
                
                # Execute the operation
                result = operation_func(*args, **kwargs)
                
                # Log successful operation
                execution_time_ms = int((time.time() - start_time) * 1000)
                self._log_operation_success(operation_id, table_name, record_key, 
                                          attempt_count, execution_time_ms)
                
                return result
                
            except sqlite3.IntegrityError as e:
                last_error = e
                execution_time_ms = int((time.time() - start_time) * 1000)
                
                # Handle constraint violation based on strategy
                if self._should_retry_constraint_violation(e, retry_policy):
                    resolution_result = self._handle_constraint_violation(
                        e, conflict_strategy, table_name, record_key, operation_id, 
                        operation_func, *args, **kwargs
                    )
                    
                    if resolution_result['resolved']:
                        self._log_operation_conflict_resolved(
                            operation_id, table_name, record_key, attempt_count,
                            execution_time_ms, conflict_strategy.value, str(e)
                        )
                        return resolution_result['result']
                    else:
                        self._log_operation_retry(
                            operation_id, table_name, record_key, attempt_count,
                            execution_time_ms, str(e)
                        )
                        
                        if attempt_count <= max_retries:
                            self._wait_before_retry(retry_policy, attempt_count)
                            continue
                else:
                    # Don't retry constraint violations for this policy
                    break
                    
            except sqlite3.OperationalError as e:
                last_error = e
                execution_time_ms = int((time.time() - start_time) * 1000)
                
                # Handle lock timeout or connection error
                if self._should_retry_operational_error(e, retry_policy):
                    self._log_operation_retry(
                        operation_id, table_name, record_key, attempt_count,
                        execution_time_ms, str(e)
                    )
                    
                    if attempt_count <= max_retries:
                        self._wait_before_retry(retry_policy, attempt_count)
                        continue
                else:
                    break
                    
            except Exception as e:
                last_error = e
                execution_time_ms = int((time.time() - start_time) * 1000)
                
                # Log unexpected error and don't retry
                self._log_operation_failure(
                    operation_id, table_name, record_key, attempt_count,
                    execution_time_ms, str(e)
                )
                break
        
        # All retries exhausted or non-retryable error
        execution_time_ms = int((time.time() - start_time) * 1000)
        self._log_operation_failure(
            operation_id, table_name, record_key, attempt_count,
            execution_time_ms, str(last_error) if last_error else "Unknown error"
        )
        
        # Re-raise the last error or create a generic one if None
        if last_error is not None:
            raise last_error
        else:
            raise RuntimeError("Operation failed with unknown error")
    
    def _handle_constraint_violation(self, error: sqlite3.IntegrityError, 
                                   strategy: ConflictResolutionStrategy,
                                   table_name: str, record_key: str, operation_id: str,
                                   operation_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Handle constraint violations based on resolution strategy
        
        Returns:
            dict: {'resolved': bool, 'result': Any, 'strategy_used': str}
        """
        
        try:
            if strategy == ConflictResolutionStrategy.UPSERT:
                # Try to convert INSERT to INSERT OR REPLACE
                result = self._convert_to_upsert(operation_func, *args, **kwargs)
                return {'resolved': True, 'result': result, 'strategy_used': 'UPSERT'}
                
            elif strategy == ConflictResolutionStrategy.IGNORE:
                # Try to convert INSERT to INSERT OR IGNORE
                result = self._convert_to_ignore(operation_func, *args, **kwargs)
                return {'resolved': True, 'result': result, 'strategy_used': 'IGNORE'}
                
            elif strategy == ConflictResolutionStrategy.UPDATE_EXISTING:
                # Try to update existing record instead of inserting
                result = self._convert_to_update(operation_func, table_name, record_key, *args, **kwargs)
                return {'resolved': True, 'result': result, 'strategy_used': 'UPDATE'}
                
            elif strategy == ConflictResolutionStrategy.RETRY:
                # Return false to trigger retry logic
                return {'resolved': False, 'result': None, 'strategy_used': 'RETRY'}
                
            else:  # FAIL
                # Let the error propagate
                return {'resolved': False, 'result': None, 'strategy_used': 'FAIL'}
                
        except Exception as resolution_error:
            logger.logger.error(f"Conflict resolution failed for {operation_id}: {resolution_error}")
            return {'resolved': False, 'result': None, 'strategy_used': f'FAILED_{strategy.value}'}
    
    def _convert_to_upsert(self, operation_func: Callable, *args, **kwargs) -> Any:
        """Convert INSERT operation to INSERT OR REPLACE"""
        # This is already handled in Phase 1 by modifying the original methods
        # to use INSERT OR REPLACE, so this should not be needed
        return operation_func(*args, **kwargs)
    
    def _convert_to_ignore(self, operation_func: Callable, *args, **kwargs) -> Any:
        """Convert INSERT operation to INSERT OR IGNORE"""
        # This is already handled for mark_post_as_replied in Phase 1
        return operation_func(*args, **kwargs)
    
    def _convert_to_update(self, operation_func: Callable, table_name: str, 
                        record_key: str, *args, **kwargs) -> Any:
        """Convert INSERT operation to UPDATE operation"""
        
        try:
            conn, cursor = self.db._get_connection()
            
            if table_name == 'content_analysis':
                # Update existing content analysis record
                post_id = args[0] if args else kwargs.get('post_id')
                content = args[1] if len(args) > 1 else kwargs.get('content')
                analysis_data = args[2] if len(args) > 2 else kwargs.get('analysis_data')
                author_handle = kwargs.get('author_handle')
                post_url = kwargs.get('post_url')
                
                # Build analysis data if not provided
                if analysis_data is None:
                    analysis_data = {
                        "reply_worthy": kwargs.get('reply_worthy', False),
                        "reply_score": kwargs.get('reply_score', 0.0),
                        "features": kwargs.get('features'),
                        "engagement_scores": kwargs.get('engagement_scores'),
                        "response_focus": kwargs.get('response_focus')
                    }
                
                analysis_json = json.dumps(analysis_data)
                
                cursor.execute("""
                    UPDATE content_analysis 
                    SET content = ?, analysis_data = ?, author_handle = ?, 
                        post_url = ?, updated_at = ?
                    WHERE post_id = ?
                """, (content, analysis_json, author_handle, post_url, datetime.now(), post_id))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    return True
                else:
                    # Record doesn't exist, fall back to original operation
                    return operation_func(*args, **kwargs)
                    
            elif table_name == 'replied_posts':
                # Update existing reply record
                post_id = args[0] if args else kwargs.get('post_id')
                post_url = kwargs.get('post_url')
                reply_text = kwargs.get('reply_text')
                
                cursor.execute("""
                    UPDATE replied_posts 
                    SET post_url = ?, reply_content = ?, replied_at = ?
                    WHERE post_id = ?
                """, (post_url, reply_text, datetime.now(), post_id))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    return True
                else:
                    # Record doesn't exist, fall back to original operation
                    return operation_func(*args, **kwargs)
                    
            else:
                # For other tables, fall back to original operation
                logger.logger.debug(f"UPDATE_EXISTING strategy not implemented for {table_name}, falling back")
                return operation_func(*args, **kwargs)
                
        except Exception as e:
            logger.logger.error(f"UPDATE conversion failed for {table_name}: {e}")
            # Fall back to original operation
            return operation_func(*args, **kwargs)
    
    def _should_retry_constraint_violation(self, error: sqlite3.IntegrityError, 
                                         retry_policy: Dict[str, Any]) -> bool:
        """Check if constraint violations should be retried for this policy"""
        return retry_policy.get('retry_on_constraint_violation', True)
    
    def _should_retry_operational_error(self, error: sqlite3.OperationalError,
                                      retry_policy: Dict[str, Any]) -> bool:
        """Check if operational errors should be retried"""
        error_str = str(error).lower()
        
        if 'database is locked' in error_str or 'database locked' in error_str:
            return retry_policy.get('retry_on_lock_timeout', True)
        elif 'unable to open database' in error_str or 'connection' in error_str:
            return retry_policy.get('retry_on_connection_error', True)
        
        return False
    
    def _wait_before_retry(self, retry_policy: Dict[str, Any], attempt_count: int):
        """Wait before retrying based on backoff strategy"""
        base_delay_ms = retry_policy.get('base_delay_ms', 100)
        max_delay_ms = retry_policy.get('max_delay_ms', 2000)
        strategy = retry_policy.get('backoff_strategy', 'EXPONENTIAL')
        
        if strategy == 'EXPONENTIAL':
            delay_ms = min(base_delay_ms * (2 ** (attempt_count - 1)), max_delay_ms)
        elif strategy == 'LINEAR':
            delay_ms = min(base_delay_ms * attempt_count, max_delay_ms)
        else:  # FIXED
            delay_ms = base_delay_ms
        
        logger.logger.debug(f"Waiting {delay_ms}ms before retry attempt {attempt_count}")
        time.sleep(delay_ms / 1000.0)
    
    def _log_operation_start(self, operation_id: str, table_name: str, 
                           record_key: str, attempt_count: int):
        """Log operation start"""
        try:
            conn, cursor = self.db._get_connection()
            
            cursor.execute("""
                INSERT INTO operation_audit (
                    operation_id, table_name, operation_type, record_key,
                    attempt_count, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                operation_id, table_name, 'INSERT', record_key,
                attempt_count, 'STARTED', datetime.now()
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.logger.warning(f"Could not log operation start: {e}")
    
    def _log_operation_success(self, operation_id: str, table_name: str,
                             record_key: str, attempt_count: int, execution_time_ms: int):
        """Log successful operation"""
        try:
            conn, cursor = self.db._get_connection()
            
            cursor.execute("""
                UPDATE operation_audit 
                SET status = 'SUCCESS', 
                    completed_at = ?,
                    execution_time_ms = ?
                WHERE operation_id = ?
            """, (datetime.now(), execution_time_ms, operation_id))
            
            conn.commit()
            
        except Exception as e:
            logger.logger.warning(f"Could not log operation success: {e}")
    
    def _log_operation_failure(self, operation_id: str, table_name: str,
                             record_key: str, attempt_count: int, 
                             execution_time_ms: int, error_details: str):
        """Log failed operation"""
        try:
            conn, cursor = self.db._get_connection()
            
            cursor.execute("""
                UPDATE operation_audit 
                SET status = 'FAILED',
                    completed_at = ?,
                    execution_time_ms = ?,
                    error_details = ?
                WHERE operation_id = ?
            """, (datetime.now(), execution_time_ms, error_details, operation_id))
            
            conn.commit()
            
        except Exception as e:
            logger.logger.warning(f"Could not log operation failure: {e}")
    
    def _log_operation_retry(self, operation_id: str, table_name: str,
                           record_key: str, attempt_count: int,
                           execution_time_ms: int, error_details: str):
        """Log operation retry"""
        try:
            conn, cursor = self.db._get_connection()
            
            cursor.execute("""
                UPDATE operation_audit 
                SET status = 'RETRYING',
                    error_details = ?,
                    execution_time_ms = ?
                WHERE operation_id = ?
            """, (error_details, execution_time_ms, operation_id))
            
            conn.commit()
            
        except Exception as e:
            logger.logger.warning(f"Could not log operation retry: {e}")
    
    def _log_operation_conflict_resolved(self, operation_id: str, table_name: str,
                                       record_key: str, attempt_count: int,
                                       execution_time_ms: int, strategy: str, 
                                       error_details: str):
        """Log conflict resolution"""
        try:
            conn, cursor = self.db._get_connection()
            
            # Update operation audit
            cursor.execute("""
                UPDATE operation_audit 
                SET status = 'CONFLICT_RESOLVED',
                    completed_at = ?,
                    execution_time_ms = ?,
                    conflict_resolution_strategy = ?
                WHERE operation_id = ?
            """, (datetime.now(), execution_time_ms, strategy, operation_id))
            
            # Log in conflict resolution table
            cursor.execute("""
                INSERT INTO conflict_resolution_log (
                    timestamp, operation_id, table_name, record_id,
                    conflict_type, resolution_strategy, resolution_successful,
                    resolution_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(), operation_id, table_name, record_key,
                'UNIQUE_CONSTRAINT', strategy, True, execution_time_ms
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.logger.warning(f"Could not log conflict resolution: {e}")            
        
# Apply enterprise operation wrappers if this module is loaded
if __name__ != "__main__":  # Only when imported, not when run directly
    wrap_enterprise_methods()   

class DatabaseHealthMonitor:
    """
    Database health monitoring system for enterprise database operations
    
    Monitors performance metrics, detects patterns, and generates alerts
    for database operations and system health.
    """
    
    def __init__(self, database_instance):
        """
        Initialize the health monitor
        
        Args:
            database_instance: CryptoDatabase instance to monitor
        """
        self.db = database_instance
        self.logger = logger
        
        # Health thresholds
        self.thresholds = {
            'max_operation_time_ms': 5000,
            'max_failure_rate_percent': 10.0,
            'max_constraint_violation_rate_percent': 5.0,
            'min_success_rate_percent': 90.0,
            'max_avg_operation_time_ms': 2000
        }
    
    def collect_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """
        Collect performance metrics from database operations
        
        Args:
            hours: Number of hours of data to collect
            
        Returns:
            Dictionary with collected metrics
        """
        try:
            conn, cursor = self.db._get_connection()
            
            since_timestamp = datetime.now() - timedelta(hours=hours)
            
            # Collect operation metrics
            cursor.execute("""
                SELECT 
                    table_name,
                    COUNT(*) as total_operations,
                    SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful_ops,
                    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed_ops,
                    SUM(CASE WHEN status = 'CONFLICT_RESOLVED' THEN 1 ELSE 0 END) as conflicts_resolved,
                    AVG(execution_time_ms) as avg_execution_time,
                    MAX(execution_time_ms) as max_execution_time,
                    MIN(execution_time_ms) as min_execution_time
                FROM operation_audit 
                WHERE created_at >= ?
                GROUP BY table_name
            """, (since_timestamp,))
            
            operation_metrics = []
            for row in cursor.fetchall():
                table_metrics = {
                    'table_name': row[0],
                    'total_operations': row[1],
                    'successful_operations': row[2],
                    'failed_operations': row[3],
                    'conflicts_resolved': row[4],
                    'avg_execution_time_ms': round(row[5] or 0, 2),
                    'max_execution_time_ms': row[6] or 0,
                    'min_execution_time_ms': row[7] or 0,
                    'success_rate_percent': round((row[2] / row[1] * 100) if row[1] > 0 else 0, 2),
                    'failure_rate_percent': round((row[3] / row[1] * 100) if row[1] > 0 else 0, 2)
                }
                operation_metrics.append(table_metrics)
            
            # Collect health metrics summary
            cursor.execute("""
                SELECT 
                    SUM(successful_operations) as total_success,
                    SUM(failed_operations) as total_failed,
                    SUM(constraint_violations) as total_violations,
                    AVG(avg_operation_time_ms) as overall_avg_time
                FROM db_health_metrics 
                WHERE timestamp >= ?
            """, (since_timestamp,))
            
            health_row = cursor.fetchone()
            health_summary = {
                'total_successful_operations': health_row[0] or 0,
                'total_failed_operations': health_row[1] or 0,
                'total_constraint_violations': health_row[2] or 0,
                'overall_avg_time_ms': round(health_row[3] or 0, 2)
            }
            
            return {
                'collection_timestamp': datetime.now().isoformat(),
                'time_period_hours': hours,
                'operation_metrics_by_table': operation_metrics,
                'health_summary': health_summary,
                'metrics_collected': True
            }
            
        except Exception as e:
            self.logger.log_error("Collect Health Metrics", str(e))
            return {
                'collection_timestamp': datetime.now().isoformat(),
                'time_period_hours': hours,
                'error': str(e),
                'metrics_collected': False
            }
    
    def detect_patterns(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Detect concerning patterns in database operations
        
        Args:
            hours: Number of hours of data to analyze
            
        Returns:
            List of detected patterns and issues
        """
        patterns = []
        
        try:
            # Collect current metrics
            metrics = self.collect_metrics(hours)
            
            if not metrics.get('metrics_collected', False):
                patterns.append({
                    'pattern_type': 'METRICS_COLLECTION_FAILURE',
                    'severity': 'ERROR',
                    'message': f"Failed to collect metrics: {metrics.get('error', 'Unknown error')}",
                    'detected_at': datetime.now().isoformat()
                })
                return patterns
            
            # Analyze operation metrics by table
            for table_metrics in metrics.get('operation_metrics_by_table', []):
                table_name = table_metrics['table_name']
                
                # Check high failure rate
                failure_rate = table_metrics.get('failure_rate_percent', 0)
                if failure_rate > self.thresholds['max_failure_rate_percent']:
                    patterns.append({
                        'pattern_type': 'HIGH_FAILURE_RATE',
                        'severity': 'WARNING',
                        'table_name': table_name,
                        'failure_rate_percent': failure_rate,
                        'threshold_percent': self.thresholds['max_failure_rate_percent'],
                        'message': f"Table {table_name} has high failure rate: {failure_rate}%",
                        'detected_at': datetime.now().isoformat()
                    })
                
                # Check slow operations
                avg_time = table_metrics.get('avg_execution_time_ms', 0)
                if avg_time > self.thresholds['max_avg_operation_time_ms']:
                    patterns.append({
                        'pattern_type': 'SLOW_OPERATIONS',
                        'severity': 'WARNING',
                        'table_name': table_name,
                        'avg_execution_time_ms': avg_time,
                        'threshold_ms': self.thresholds['max_avg_operation_time_ms'],
                        'message': f"Table {table_name} has slow operations: {avg_time}ms average",
                        'detected_at': datetime.now().isoformat()
                    })
                
                # Check very slow individual operations
                max_time = table_metrics.get('max_execution_time_ms', 0)
                if max_time > self.thresholds['max_operation_time_ms']:
                    patterns.append({
                        'pattern_type': 'VERY_SLOW_OPERATION',
                        'severity': 'ERROR',
                        'table_name': table_name,
                        'max_execution_time_ms': max_time,
                        'threshold_ms': self.thresholds['max_operation_time_ms'],
                        'message': f"Table {table_name} had very slow operation: {max_time}ms",
                        'detected_at': datetime.now().isoformat()
                    })
            
            # Check for repeated constraint violations
            self._detect_constraint_violation_patterns(patterns, hours)
            
            # Check for database lock patterns
            self._detect_lock_patterns(patterns, hours)
            
        except Exception as e:
            patterns.append({
                'pattern_type': 'PATTERN_DETECTION_ERROR',
                'severity': 'ERROR',
                'message': f"Pattern detection failed: {str(e)}",
                'detected_at': datetime.now().isoformat()
            })
        
        return patterns
    
    def generate_alerts(self, patterns: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Generate alerts based on detected patterns
        
        Args:
            patterns: List of patterns to generate alerts for (optional)
            
        Returns:
            List of generated alerts
        """
        if patterns is None:
            patterns = self.detect_patterns()
        
        alerts = []
        
        try:
            for pattern in patterns:
                severity = pattern.get('severity', 'INFO')
                pattern_type = pattern.get('pattern_type', 'UNKNOWN')
                
                # Generate alert based on pattern severity
                if severity == 'ERROR':
                    alert = {
                        'alert_id': f"alert_{int(time.time())}_{len(alerts)}",
                        'alert_type': 'CRITICAL',
                        'priority': 'HIGH',
                        'title': f"Critical Database Issue: {pattern_type}",
                        'message': pattern.get('message', 'Critical database issue detected'),
                        'pattern': pattern,
                        'recommended_actions': self._get_recommended_actions(pattern_type),
                        'generated_at': datetime.now().isoformat()
                    }
                    alerts.append(alert)
                    
                elif severity == 'WARNING':
                    alert = {
                        'alert_id': f"alert_{int(time.time())}_{len(alerts)}",
                        'alert_type': 'WARNING',
                        'priority': 'MEDIUM',
                        'title': f"Database Performance Warning: {pattern_type}",
                        'message': pattern.get('message', 'Database performance issue detected'),
                        'pattern': pattern,
                        'recommended_actions': self._get_recommended_actions(pattern_type),
                        'generated_at': datetime.now().isoformat()
                    }
                    alerts.append(alert)
            
            # Log alerts
            if alerts:
                self.logger.logger.warning(f"Generated {len(alerts)} database health alerts")
                for alert in alerts:
                    if alert['alert_type'] == 'CRITICAL':
                        self.logger.logger.error(f"CRITICAL ALERT: {alert['title']}")
                    else:
                        self.logger.logger.warning(f"WARNING ALERT: {alert['title']}")
            
        except Exception as e:
            alerts.append({
                'alert_id': f"alert_{int(time.time())}_error",
                'alert_type': 'CRITICAL',
                'priority': 'HIGH',
                'title': 'Alert Generation Failed',
                'message': f"Failed to generate alerts: {str(e)}",
                'generated_at': datetime.now().isoformat()
            })
        
        return alerts
    
    def _detect_constraint_violation_patterns(self, patterns: List[Dict[str, Any]], hours: int):
        """Detect patterns in constraint violations"""
        try:
            conn, cursor = self.db._get_connection()
            
            since_timestamp = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT table_name, COUNT(*) as violation_count
                FROM conflict_resolution_log 
                WHERE timestamp >= ? AND conflict_type = 'UNIQUE_CONSTRAINT'
                GROUP BY table_name
                HAVING violation_count > 10
            """, (since_timestamp,))
            
            for row in cursor.fetchall():
                patterns.append({
                    'pattern_type': 'REPEATED_CONSTRAINT_VIOLATIONS',
                    'severity': 'WARNING',
                    'table_name': row[0],
                    'violation_count': row[1],
                    'time_period_hours': hours,
                    'message': f"Table {row[0]} has {row[1]} constraint violations in {hours} hours",
                    'detected_at': datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.logger.debug(f"Could not detect constraint patterns: {e}")
    
    def _detect_lock_patterns(self, patterns: List[Dict[str, Any]], hours: int):
        """Detect patterns in database locks"""
        try:
            conn, cursor = self.db._get_connection()
            
            since_timestamp = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT table_name, COUNT(*) as lock_count
                FROM operation_audit 
                WHERE created_at >= ? 
                AND error_details LIKE '%database is locked%'
                GROUP BY table_name
                HAVING lock_count > 5
            """, (since_timestamp,))
            
            for row in cursor.fetchall():
                patterns.append({
                    'pattern_type': 'FREQUENT_DATABASE_LOCKS',
                    'severity': 'WARNING',
                    'table_name': row[0],
                    'lock_count': row[1],
                    'time_period_hours': hours,
                    'message': f"Table {row[0]} had {row[1]} lock conflicts in {hours} hours",
                    'detected_at': datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.logger.debug(f"Could not detect lock patterns: {e}")
    
    def _get_recommended_actions(self, pattern_type: str) -> List[str]:
        """Get recommended actions for different pattern types"""
        
        action_map = {
            'HIGH_FAILURE_RATE': [
                "Review recent code changes affecting this table",
                "Check database connection stability", 
                "Verify table schema and constraints",
                "Consider increasing retry limits"
            ],
            'SLOW_OPERATIONS': [
                "Analyze query performance and add indexes if needed",
                "Check for table locks or blocking operations",
                "Consider database optimization (VACUUM, ANALYZE)",
                "Review operation complexity"
            ],
            'VERY_SLOW_OPERATION': [
                "Investigate specific slow operation in audit logs",
                "Check for concurrent operations causing locks",
                "Consider query optimization or caching"
            ],
            'REPEATED_CONSTRAINT_VIOLATIONS': [
                "Review application logic for duplicate handling",
                "Consider adjusting conflict resolution strategy",
                "Check for race conditions in concurrent operations"
            ],
            'FREQUENT_DATABASE_LOCKS': [
                "Reduce transaction duration",
                "Implement connection pooling",
                "Consider database optimization",
                "Review concurrent operation patterns"
            ]
        }
        
        return action_map.get(pattern_type, ["Review system logs and investigate manually"])         