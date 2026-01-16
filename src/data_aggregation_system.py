"""
Enhanced Multi-Source Data Aggregation System
Modular design for ALL data sources with future CoinGecko/CoinMarketCap integration

File: src/data_aggregation_system.py
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import json
import time
from utils.logger import logger

@dataclass
class DataPoint:
    """Standardized data point across all sources"""
    timestamp: datetime
    token: str
    price: float
    volume: float
    high: Optional[float] = None
    low: Optional[float] = None
    market_cap: Optional[float] = None
    total_supply: Optional[float] = None
    circulating_supply: Optional[float] = None
    source: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class AggregatedDataset:
    """Complete aggregated dataset for predictions"""
    token: str
    timeframe: str
    data_points: List[DataPoint]
    current_price: float
    price_change_24h: float
    volume_24h: float
    market_cap: Optional[float] = None
    data_quality_score: float = 0.0
    source_distribution: Dict[str, int] = field(default_factory=dict)  # FIXED
    coverage_hours: float = 0.0

class DataSourceAdapter(ABC):
    """Abstract base class for all data source adapters"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this data source"""
        pass
    
    @abstractmethod
    def fetch_data(self, token: str, hours: int) -> List[DataPoint]:
        """Fetch historical data from this source"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Return priority level (1=highest, 10=lowest)"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this data source is currently available"""
        pass

class MarketDataAdapter(DataSourceAdapter):
    """Adapter for market_data table (current API data)"""
    
    def __init__(self, database):
        # Store reference but use singleton for actual operations
        self.db = database
    
    def get_name(self) -> str:
        return "market_data"
    
    def get_priority(self) -> int:
        return 1  # Highest priority - current API data
    
    def is_available(self) -> bool:
        return self.db is not None
    
    def fetch_data(self, token: str, hours: int) -> List[DataPoint]:
        """Fetch from market_data table using DatabaseManager singleton"""
        try:
            if not self.is_available():
                return []
            
            # Use DatabaseManager singleton to get raw database connection
            from database import DatabaseManager
            db_manager = DatabaseManager()
            db_instance = db_manager.get_database()
            
            # Make DIRECT database query to avoid recursion
            conn, cursor = db_instance._get_connection()
            cursor.execute("""
                SELECT * FROM market_data 
                WHERE chain = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (token, hours))
            
            historical_data = [dict(row) for row in cursor.fetchall()]
            
            data_points = []
            for entry in historical_data:
                # Safe timestamp handling
                timestamp_value = entry.get('timestamp')
                if timestamp_value is None:
                    timestamp_value = datetime.now().isoformat()
                
                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value is not None else default
                    except (ValueError, TypeError):
                        return default
                
                data_point = DataPoint(
                    timestamp=self._parse_timestamp(timestamp_value),
                    token=token,
                    price=safe_float(entry.get('price'), 0.0),
                    volume=safe_float(entry.get('volume'), 0.0),
                    high=safe_float(entry.get('high_24h')) if entry.get('high_24h') is not None else None,
                    low=safe_float(entry.get('low_24h')) if entry.get('low_24h') is not None else None,
                    market_cap=safe_float(entry.get('market_cap')) if entry.get('market_cap') is not None else None,
                    source="market_data",
                    metadata={"original_entry": entry}
                )
                data_points.append(data_point)
            
            logger.logger.debug(f"MarketDataAdapter fetched {len(data_points)} points for {token}")
            return data_points
            
        except Exception as e:
            logger.logger.warning(f"MarketDataAdapter error for {token}: {str(e)}")
            return []
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from database"""
        try:
            if isinstance(timestamp_str, datetime):
                return timestamp_str
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            return datetime.now()

class PriceHistoryAdapter(DataSourceAdapter):
    """Adapter for price_history table (bootstrap + stored data)"""
    
    def __init__(self, database):
        self.db = database
    
    def get_name(self) -> str:
        return "price_history"
    
    def get_priority(self) -> int:
        return 2  # Second priority - includes bootstrap data
    
    def is_available(self) -> bool:
        return self.db is not None
    
    def fetch_data(self, token: str, hours: int) -> List[DataPoint]:
        """Fetch from price_history table"""
        try:
            if not self.is_available():
                return []
            
            # Direct query to price_history table
            conn, cursor = self.db._get_connection()
            cursor.execute("""
                SELECT * FROM price_history 
                WHERE token = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """, (token, hours))
            
            historical_data = [dict(row) for row in cursor.fetchall()]
            
            data_points = []
            for entry in historical_data:
                # Safe timestamp handling
                timestamp_value = entry.get('timestamp')
                if timestamp_value is None:
                    timestamp_value = datetime.now().isoformat()
                
                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value is not None else default
                    except (ValueError, TypeError):
                        return default
                
                data_point = DataPoint(
                    timestamp=self._parse_timestamp(timestamp_value),
                    token=token,
                    price=safe_float(entry.get('price'), 0.0),
                    volume=safe_float(entry.get('volume'), 0.0),
                    high=safe_float(entry.get('high')) if entry.get('high') is not None else None,
                    low=safe_float(entry.get('low')) if entry.get('low') is not None else None,
                    market_cap=safe_float(entry.get('market_cap')) if entry.get('market_cap') is not None else None,
                    total_supply=safe_float(entry.get('total_supply')) if entry.get('total_supply') is not None else None,
                    circulating_supply=safe_float(entry.get('circulating_supply')) if entry.get('circulating_supply') is not None else None,
                    source="price_history",
                    metadata={"original_entry": entry}
                )
                data_points.append(data_point)
            
            logger.logger.debug(f"PriceHistoryAdapter fetched {len(data_points)} points for {token}")
            return data_points
            
        except Exception as e:
            logger.logger.warning(f"PriceHistoryAdapter error for {token}: {str(e)}")
            return []
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from database"""
        try:
            if isinstance(timestamp_str, datetime):
                return timestamp_str
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            return datetime.now()

# Future adapters (ready for implementation)
class CoinGeckoAdapter(DataSourceAdapter):
    """Future adapter for CoinGecko dedicated table"""
    
    def __init__(self, database):
        self.db = database
    
    def get_name(self) -> str:
        return "coingecko_data"
    
    def get_priority(self) -> int:
        return 3  # Third priority
    
    def is_available(self) -> bool:
        # TODO: Check if coingecko_data table exists
        return False
    
    def fetch_data(self, token: str, hours: int) -> List[DataPoint]:
        """Future implementation for CoinGecko table"""
        # TODO: Implement when CoinGecko table is added
        return []

class CoinMarketCapAdapter(DataSourceAdapter):
    """Future adapter for CoinMarketCap dedicated table"""
    
    def __init__(self, database):
        self.db = database
    
    def get_name(self) -> str:
        return "coinmarketcap_data"
    
    def get_priority(self) -> int:
        return 4  # Fourth priority
    
    def is_available(self) -> bool:
        # TODO: Check if coinmarketcap_data table exists
        return False
    
    def fetch_data(self, token: str, hours: int) -> List[DataPoint]:
        """Future implementation for CoinMarketCap table"""
        # TODO: Implement when CoinMarketCap table is added
        return []

class MultiSourceDataAggregator:
    """
    Core aggregation engine that combines data from all sources
    """
    
    def __init__(self, database):
        self.db = database
        self.adapters: List[DataSourceAdapter] = []
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Initialize all adapters
        self._initialize_adapters()
        
        logger.logger.info(f"🔄 MultiSourceDataAggregator initialized with {len(self.adapters)} adapters")
    
    def _initialize_adapters(self):
        """Initialize all available data source adapters"""
        try:
            # Core adapters (always available)
            self.adapters.append(MarketDataAdapter(self.db))
            self.adapters.append(PriceHistoryAdapter(self.db))
            
            # Future adapters (add when ready)
            coingecko_adapter = CoinGeckoAdapter(self.db)
            if coingecko_adapter.is_available():
                self.adapters.append(coingecko_adapter)
            
            coinmarketcap_adapter = CoinMarketCapAdapter(self.db)
            if coinmarketcap_adapter.is_available():
                self.adapters.append(coinmarketcap_adapter)
            
            # Sort by priority
            self.adapters.sort(key=lambda x: x.get_priority())
            
            active_adapters = [a.get_name() for a in self.adapters if a.is_available()]
            logger.logger.info(f"✅ Active adapters: {active_adapters}")
            
        except Exception as e:
            logger.logger.error(f"Adapter initialization error: {str(e)}")
    
    def get_aggregated_data(self, token: str, hours: int, timeframe: str = "1h") -> AggregatedDataset:
        """
        Get comprehensive aggregated data from ALL sources
        
        Args:
            token: Token symbol
            hours: Hours of historical data needed
            timeframe: Prediction timeframe
            
        Returns:
            Complete aggregated dataset ready for predictions
        """
        try:
            cache_key = f"{token}_{hours}_{timeframe}"
            
            # Check cache
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_duration:
                    logger.logger.debug(f"Cache hit for {token}")
                    return cached_data
            
            logger.logger.info(f"🔄 Aggregating data for {token} from ALL sources")
            
            # Collect data from all available sources
            all_data_points = []
            source_stats = {}
            
            for adapter in self.adapters:
                if not adapter.is_available():
                    continue
                    
                try:
                    data_points = adapter.fetch_data(token, hours)
                    all_data_points.extend(data_points)
                    source_stats[adapter.get_name()] = len(data_points)
                    
                    logger.logger.debug(f"  {adapter.get_name()}: {len(data_points)} points")
                    
                except Exception as e:
                    logger.logger.warning(f"Adapter {adapter.get_name()} failed: {str(e)}")
                    source_stats[adapter.get_name()] = 0
            
            # Remove duplicates and sort by timestamp
            unique_data_points = self._deduplicate_data_points(all_data_points)
            unique_data_points.sort(key=lambda x: x.timestamp)
            
            # Calculate aggregated metrics
            aggregated_dataset = self._create_aggregated_dataset(
                token, timeframe, unique_data_points, source_stats
            )
            
            # Cache the result
            self.cache[cache_key] = (datetime.now(), aggregated_dataset)
            
            logger.logger.info(f"✅ Aggregated {len(unique_data_points)} unique data points for {token}")
            logger.logger.info(f"📊 Sources: {source_stats}")
            logger.logger.info(f"🎯 Data quality score: {aggregated_dataset.data_quality_score:.2f}")
            
            return aggregated_dataset
            
        except Exception as e:
            logger.logger.error(f"Data aggregation failed for {token}: {str(e)}")
            return self._create_empty_dataset(token, timeframe)
    
    def _deduplicate_data_points(self, data_points: List[DataPoint]) -> List[DataPoint]:
        """Remove duplicate data points by timestamp and price"""
        try:
            seen = set()
            unique_points = []
            
            for point in data_points:
                # Create unique key based on timestamp (rounded to nearest minute) and price
                key = (
                    point.timestamp.replace(second=0, microsecond=0),
                    round(point.price, 8)
                )
                
                if key not in seen:
                    seen.add(key)
                    unique_points.append(point)
            
            return unique_points
            
        except Exception as e:
            logger.logger.warning(f"Deduplication error: {str(e)}")
            return data_points
    
    def _create_aggregated_dataset(self, token: str, timeframe: str, 
                             data_points: List[DataPoint], 
                             source_stats: Dict[str, int]) -> AggregatedDataset:
        """Create final aggregated dataset"""
        try:
            if not data_points:
                return self._create_empty_dataset(token, timeframe)
            
            # Current metrics from most recent data point
            latest_point = data_points[-1]
            current_price = latest_point.price
            
            # Calculate 24h price change
            price_change_24h = 0.0
            cutoff_time = datetime.now() - timedelta(hours=24)
            historical_points = [p for p in data_points if p.timestamp <= cutoff_time]
            
            if historical_points:
                old_price = historical_points[-1].price
                price_change_24h = ((current_price - old_price) / old_price) * 100
            
            # Calculate 24h volume
            recent_points = [p for p in data_points if p.timestamp >= cutoff_time]
            volume_24h = sum(p.volume for p in recent_points) / len(recent_points) if recent_points else latest_point.volume
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality_score(data_points, source_stats)
            
            # Calculate coverage
            if len(data_points) >= 2:
                time_span = (data_points[-1].timestamp - data_points[0].timestamp).total_seconds() / 3600
                coverage_hours = min(time_span, 168)  # Cap at 1 week
            else:
                coverage_hours = 0.0
            
            return AggregatedDataset(
                token=token,
                timeframe=timeframe,
                data_points=data_points,
                current_price=current_price,
                price_change_24h=price_change_24h,
                volume_24h=volume_24h,
                market_cap=latest_point.market_cap,
                data_quality_score=quality_score,
                source_distribution=source_stats,
                coverage_hours=coverage_hours
            )
            
        except Exception as e:
            logger.logger.error(f"Aggregated dataset creation failed: {str(e)}")
            return self._create_empty_dataset(token, timeframe)
    
    def _calculate_data_quality_score(self, data_points: List[DataPoint], 
                                source_stats: Dict[str, int]) -> float:
        """Calculate data quality score based on completeness and source diversity"""
        try:
            score = 0.0
            
            # Points quantity score (0-40)
            num_points = len(data_points)
            if num_points >= 100:
                score += 40
            elif num_points >= 50:
                score += 30
            elif num_points >= 20:
                score += 20
            elif num_points >= 10:
                score += 10
            
            # Source diversity score (0-30)
            active_sources = sum(1 for count in source_stats.values() if count > 0)
            if active_sources >= 3:
                score += 30
            elif active_sources == 2:
                score += 20
            elif active_sources == 1:
                score += 10
            
            # Data completeness score (0-20)
            complete_points = sum(1 for p in data_points if p.volume > 0 and p.price > 0)
            completeness = complete_points / len(data_points) if data_points else 0
            score += completeness * 20
            
            # Time coverage score (0-10)
            if len(data_points) >= 2:
                time_span_hours = (data_points[-1].timestamp - data_points[0].timestamp).total_seconds() / 3600
                coverage_score = min(time_span_hours / 24, 1.0) * 10  # Up to 24h coverage
                score += coverage_score
            
            return min(score, 100.0)
            
        except Exception as e:
            logger.logger.warning(f"Quality score calculation error: {str(e)}")
            return 50.0
    
    def _create_empty_dataset(self, token: str, timeframe: str) -> AggregatedDataset:
        """Create empty dataset for error cases"""
        return AggregatedDataset(
            token=token,
            timeframe=timeframe,
            data_points=[],
            current_price=0.0,
            price_change_24h=0.0,
            volume_24h=0.0,
            data_quality_score=0.0,
            source_distribution={},
            coverage_hours=0.0
        )
    
    def get_adapter_status(self) -> Dict[str, Any]:
        """Get status of all adapters"""
        status = {}
        for adapter in self.adapters:
            status[adapter.get_name()] = {
                'available': adapter.is_available(),
                'priority': adapter.get_priority()
            }
        return status

class EnhancedMarketDataRetriever:
    """
    Enhanced replacement for get_recent_market_data() method
    Uses multi-source aggregation system
    """
    
    def __init__(self, database):
        self.aggregator = MultiSourceDataAggregator(database)
    
    def get_recent_market_data(self, token: str, hours: int = 24) -> List[Dict]:
        """
        Enhanced method that replaces the original get_recent_market_data()
        Returns data in the same format but from ALL sources
        """
        try:
            # Get aggregated data from all sources
            aggregated_data = self.aggregator.get_aggregated_data(token, hours)
            
            # Convert back to original format for compatibility
            result = []
            for data_point in aggregated_data.data_points:
                entry = {
                    'timestamp': data_point.timestamp,
                    'token': data_point.token,
                    'price': data_point.price,
                    'volume': data_point.volume,
                    'high': data_point.high,
                    'low': data_point.low,
                    'market_cap': data_point.market_cap,
                    'total_supply': data_point.total_supply,
                    'circulating_supply': data_point.circulating_supply,
                    'source': data_point.source
                }
                result.append(entry)
            
            return result
            
        except Exception as e:
            logger.logger.error(f"Enhanced market data retrieval failed for {token}: {str(e)}")
            return []

# Factory function for easy integration
def create_enhanced_data_system(database) -> EnhancedMarketDataRetriever:
    """Factory function to create enhanced data system"""
    return EnhancedMarketDataRetriever(database)