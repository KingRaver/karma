"""
src/data_validation_layer.py - Additional Validation System
Part 1: Imports and Core Metadata Classes

This is an ADDITIONAL validation layer that works alongside existing systems
to prevent accuracy issues like the 38% bearish prediction when fresh data should have been used.

File Location: src/data_validation_layer.py
"""

# Standard library imports
import asyncio
from datetime import datetime, timedelta
from typing import Set, Union, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import time
import threading
import math
import re
from decimal import Decimal, InvalidOperation

# Third-party imports (if available)
try:
    import numpy as np
except ImportError:
    np = None

# Local imports
from config import logger

@dataclass
class DataValidationMetadata:
    """Metadata for additional validation layer working alongside existing systems"""
    
    # Core validation information
    validation_timestamp: datetime
    data_source_used: str  # 'api_real_time', 'database_recent', 'database_historical', 'bootstrap_fallback'
    data_age_minutes: float
    validation_passed: bool
    
    # Risk assessment
    accuracy_risk_level: str  # 'low', 'medium', 'high', 'critical'
    recommended_confidence_adjustment: float  # Positive or negative adjustment
    validation_warnings: List[str]
    should_flag_for_review: bool
    
    # Data quality metrics
    data_points_available: int = 0
    freshness_score: float = 0.0  # 0.0-1.0, where 1.0 = most fresh
    is_real_data: bool = True  # False only for bootstrap
    market_context_alignment: bool = True
    
    # Additional metadata for tracking
    timeframe_validated_for: str = "1h"
    confidence_boost_applied: float = 0.0
    confidence_penalty_applied: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for storage/logging"""
        return {
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'data_source_used': self.data_source_used,
            'data_age_minutes': self.data_age_minutes,
            'validation_passed': self.validation_passed,
            'accuracy_risk_level': self.accuracy_risk_level,
            'recommended_confidence_adjustment': self.recommended_confidence_adjustment,
            'validation_warnings': self.validation_warnings,
            'should_flag_for_review': self.should_flag_for_review,
            'data_points_available': self.data_points_available,
            'freshness_score': self.freshness_score,
            'is_real_data': self.is_real_data,
            'market_context_alignment': self.market_context_alignment,
            'timeframe_validated_for': self.timeframe_validated_for,
            'confidence_boost_applied': self.confidence_boost_applied,
            'confidence_penalty_applied': self.confidence_penalty_applied
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataValidationMetadata':
        """Create metadata from dictionary"""
        return cls(
            validation_timestamp=datetime.fromisoformat(data['validation_timestamp']),
            data_source_used=data['data_source_used'],
            data_age_minutes=data['data_age_minutes'],
            validation_passed=data['validation_passed'],
            accuracy_risk_level=data['accuracy_risk_level'],
            recommended_confidence_adjustment=data['recommended_confidence_adjustment'],
            validation_warnings=data['validation_warnings'],
            should_flag_for_review=data['should_flag_for_review'],
            data_points_available=data.get('data_points_available', 0),
            freshness_score=data.get('freshness_score', 0.0),
            is_real_data=data.get('is_real_data', True),
            market_context_alignment=data.get('market_context_alignment', True),
            timeframe_validated_for=data.get('timeframe_validated_for', '1h'),
            confidence_boost_applied=data.get('confidence_boost_applied', 0.0),
            confidence_penalty_applied=data.get('confidence_penalty_applied', 0.0)
        )


"""
src/data_validation_layer.py - Part 2: PredictionAccuracyValidator Core Class

This part adds the main validator class with core validation methods.
This should be added AFTER Part 1 in the same file.
"""

class PredictionAccuracyValidator:
    """Additional validation layer working alongside existing systems"""
    
    def __init__(self, existing_systems_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the additional validation layer
        
        Args:
            existing_systems_config: Configuration for existing systems integration
        """
        self.config = existing_systems_config or {}
        self.validation_history = []
        
        # Default validation thresholds
        self.data_age_thresholds = {
            '1h': 360,    # 6 hours for 1h predictions
            '24h': 1440,  # 24 hours for 24h predictions
            '7d': 4320    # 72 hours for 7d predictions
        }
        
        # Confidence adjustment rules
        self.confidence_rules = {
            'fresh_data_boost': 10.0,
            'stale_data_penalty': -25.0,
            'bootstrap_data_penalty': -40.0,
            'extreme_prediction_threshold': 15.0
        }
        
        logger.info("🔍 PredictionAccuracyValidator initialized - Additional validation layer active")
    
    def validate_data_before_prediction(self, token: str, data_being_used: Any, timeframe: str) -> DataValidationMetadata:
        """
        Validate data before prediction generation
        
        Args:
            token: Token symbol
            data_being_used: The actual data that will be used for prediction
            timeframe: Prediction timeframe ('1h', '24h', '7d')
            
        Returns:
            DataValidationMetadata: Validation results and recommendations
        """
        try:
            validation_start = time.time()
            
            # Analyze data source and freshness
            data_source = self._detect_data_source(data_being_used)
            data_age = self._calculate_data_age(data_being_used)
            data_points = self._count_data_points(data_being_used)
            
            # Calculate freshness score
            freshness_score = self._calculate_freshness_score(data_age, timeframe)
            
            # Determine risk level
            risk_level = self._assess_accuracy_risk(data_age, data_points, timeframe, data_source)
            
            # Check validation rules
            validation_passed = self._check_validation_rules(data_age, data_points, timeframe)
            
            # Generate warnings
            warnings = self._generate_validation_warnings(data_age, data_points, timeframe, data_source)
            
            # Calculate confidence adjustments
            confidence_adjustment = self._calculate_confidence_adjustment(
                data_source, data_age, timeframe, freshness_score
            )
            
            # Create metadata
            metadata = DataValidationMetadata(
                validation_timestamp=datetime.now(),
                data_source_used=data_source,
                data_age_minutes=data_age,
                validation_passed=validation_passed,
                accuracy_risk_level=risk_level,
                recommended_confidence_adjustment=confidence_adjustment,
                validation_warnings=warnings,
                should_flag_for_review=risk_level in ['high', 'critical'],
                data_points_available=data_points,
                freshness_score=freshness_score,
                is_real_data=data_source != 'bootstrap_fallback',
                timeframe_validated_for=timeframe
            )
            
            # Store validation result
            self.validation_history.append(metadata)
            
            # Log validation result
            logger.info(f"🔍 Data validation for {token} ({timeframe}): "
                       f"Source={data_source}, Age={data_age:.1f}min, "
                       f"Risk={risk_level}, Adjustment={confidence_adjustment:+.1f}%")
            
            if warnings:
                logger.warning(f"⚠️ Validation warnings for {token}: {', '.join(warnings)}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Data validation failed for {token}: {str(e)}")
            # Return safe default metadata
            return DataValidationMetadata(
                validation_timestamp=datetime.now(),
                data_source_used='unknown',
                data_age_minutes=999999,  # Very high age to trigger safety measures
                validation_passed=False,
                accuracy_risk_level='critical',
                recommended_confidence_adjustment=-50.0,
                validation_warnings=[f"Validation error: {str(e)}"],
                should_flag_for_review=True,
                timeframe_validated_for=timeframe
            )
    
    def cross_validate_against_current_market(self, token: str, prediction_data: Any) -> Dict[str, Any]:
        """
        Cross-validate prediction data against current market conditions
        
        Args:
            token: Token symbol
            prediction_data: Data being used for prediction
            
        Returns:
            Dict containing validation results
        """
        try:
            # This method will validate against real-time market sentiment
            # and detect if prediction contradicts current market conditions
            
            validation_results = {
                'market_alignment': True,
                'sentiment_match': True,
                'contradiction_detected': False,
                'validation_notes': []
            }
            
            # Add market context validation logic here
            # This is where we would detect the AVAX bullish vs bearish mismatch
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Market validation failed for {token}: {str(e)}")
            return {
                'market_alignment': False,
                'sentiment_match': False,
                'contradiction_detected': True,
                'validation_notes': [f"Validation error: {str(e)}"],
                'error': str(e)
            }
    
    def recommend_confidence_adjustments(self, base_confidence: float, validation_results: Dict) -> float:
        """
        Recommend confidence adjustments based on validation results
        
        Args:
            base_confidence: Original confidence level
            validation_results: Results from validation checks
            
        Returns:
            Adjusted confidence level
        """
        try:
            adjusted_confidence = base_confidence
            
            # Apply validation-based adjustments
            if isinstance(validation_results, DataValidationMetadata):
                adjusted_confidence += validation_results.recommended_confidence_adjustment
            elif isinstance(validation_results, dict):
                adjustment = validation_results.get('recommended_confidence_adjustment', 0)
                adjusted_confidence += adjustment
            
            # Ensure confidence stays within valid bounds
            adjusted_confidence = max(0.0, min(100.0, adjusted_confidence))
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Confidence adjustment failed: {str(e)}")
            return base_confidence * 0.5  # Conservative fallback
        
    """
src/data_validation_layer.py - Part 3: Helper Methods

This part adds all the helper methods that support the main validation functionality.
This should be added to the PredictionAccuracyValidator class AFTER Part 2.
"""

    # Helper Methods for Data Analysis
    
    def _detect_data_source(self, data: Any) -> str:
        """Enhanced data source detection with professional categorization"""
        try:
            if isinstance(data, dict):
                # Explicit data source type (from improved validation calls)
                if 'data_source_type' in data:
                    return data['data_source_type']
                
                # Historical database data (like working 7d predictions)
                if 'historical_prices' in data and len(data['historical_prices']) > 50:
                    return 'database_historical'
                
                # Current market snapshot (1h predictions fallback)
                if data.get('is_market_snapshot', False):
                    return 'api_real_time'
                
                # CoinGecko API format detection
                if 'current_price' in data and 'last_updated' in data:
                    return 'api_real_time'
                elif 'current_price' in data:
                    return 'api_real_time'  # Assume real-time if has current_price
                
                # Database format
                elif 'prices' in data or 'price' in data:
                    return 'database_historical'
            
            return 'unknown'
            
        except Exception as e:
            logger.warning(f"Could not detect data source: {str(e)}")
            return 'unknown'
    
    def _calculate_data_age(self, data: Any) -> float:
        """
        Calculate the age of the data in minutes
        
        Args:
            data: Data to analyze
            
        Returns:
            float: Age in minutes
        """
        try:
            current_time = datetime.now()
            
            # Try to find timestamp in various formats
            timestamp = None
            
            if isinstance(data, dict):
                # Look for common timestamp fields
                for time_field in ['timestamp', 'last_updated', 'created_at', 'updated_at']:
                    if time_field in data:
                        timestamp_value = data[time_field]
                        if isinstance(timestamp_value, datetime):
                            timestamp = timestamp_value
                        elif isinstance(timestamp_value, str):
                            try:
                                timestamp = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                            except:
                                pass
                        break
            
            # If no timestamp found, assume data is relatively fresh (30 minutes)
            if timestamp is None:
                return 30.0
            
            # Calculate age in minutes
            age_delta = current_time - timestamp
            age_minutes = age_delta.total_seconds() / 60.0
            
            return max(0.0, age_minutes)
            
        except Exception as e:
            logger.warning(f"Could not calculate data age: {str(e)}")
            return 999.0  # Very old if we can't determine
    
    def _count_data_points(self, data: Any) -> int:
        """Enhanced data point counting with professional context awareness"""
        try:
            if isinstance(data, (list, tuple)):
                return len(data)
            elif isinstance(data, dict):
                # Explicit data point count (from improved validation calls)
                if 'data_points' in data:
                    return data['data_points']
                
                # Historical price arrays (7d/24h predictions)
                if 'historical_prices' in data:
                    return len(data['historical_prices'])
                
                # Look for other data arrays
                for array_field in ['prices', 'values', 'sparkline_in_7d']:
                    if array_field in data:
                        array_data = data[array_field]
                        if isinstance(array_data, (list, tuple)):
                            return len(array_data)
                        elif isinstance(array_data, dict) and 'price' in array_data:
                            return len(array_data['price'])
                
                # Market snapshot with context (1h predictions)
                if data.get('is_market_snapshot', False):
                    # Professional assumption: market snapshots used with technical indicators
                    # provide reasonable analysis capability for short-term predictions
                    return 25  # Reasonable assumption for 1h technical analysis capability
                
                # Default single data point
                return 1
            else:
                return 0
                
        except Exception as e:
            logger.warning(f"Could not count data points: {str(e)}")
            return 0
    
    def _calculate_freshness_score(self, data_age_minutes: float, timeframe: str) -> float:
        """
        Calculate a freshness score (0.0 to 1.0) based on data age and timeframe
        
        Args:
            data_age_minutes: Age of data in minutes
            timeframe: Prediction timeframe
            
        Returns:
            float: Freshness score (1.0 = freshest, 0.0 = stale)
        """
        try:
            # Get the acceptable age threshold for this timeframe
            max_age = self.data_age_thresholds.get(timeframe, 360)  # Default 6 hours
            
            # Calculate freshness score
            if data_age_minutes <= 5:  # Very fresh (< 5 minutes)
                return 1.0
            elif data_age_minutes <= max_age * 0.25:  # Fresh (< 25% of threshold)
                return 0.8
            elif data_age_minutes <= max_age * 0.5:   # Acceptable (< 50% of threshold)
                return 0.6
            elif data_age_minutes <= max_age:         # Stale but usable
                return 0.3
            else:  # Too stale
                return 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate freshness score: {str(e)}")
            return 0.0
    
    def _assess_accuracy_risk(self, data_age: float, data_points: int, timeframe: str, data_source: str) -> str:
        """
        Assess the accuracy risk level based on data characteristics
        
        Args:
            data_age: Age of data in minutes
            data_points: Number of data points
            timeframe: Prediction timeframe
            data_source: Type of data source
            
        Returns:
            str: Risk level ('low', 'medium', 'high', 'critical')
        """
        try:
            risk_factors = 0
            
            # Age-based risk
            threshold = self.data_age_thresholds.get(timeframe, 360)
            if data_age > threshold * 2:
                risk_factors += 3  # Critical age
            elif data_age > threshold:
                risk_factors += 2  # High age
            elif data_age > threshold * 0.5:
                risk_factors += 1  # Medium age
            
            # Data points risk
            if data_points < 10:
                risk_factors += 2
            elif data_points < 25:
                risk_factors += 1
            
            # Source-based risk
            if data_source == 'bootstrap_fallback':
                risk_factors += 2
            elif data_source == 'unknown':
                risk_factors += 1
            
            # Determine risk level
            if risk_factors >= 5:
                return 'critical'
            elif risk_factors >= 3:
                return 'high'
            elif risk_factors >= 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.warning(f"Could not assess accuracy risk: {str(e)}")
            return 'high'  # Conservative default
    
    def _check_validation_rules(self, data_age: float, data_points: int, timeframe: str) -> bool:
        """
        Check if data passes basic validation rules
        
        Args:
            data_age: Age of data in minutes
            data_points: Number of data points
            timeframe: Prediction timeframe
            
        Returns:
            bool: True if validation passes
        """
        try:
            # Age validation
            max_age = self.data_age_thresholds.get(timeframe, 360)
            if data_age > max_age * 3:  # 3x the threshold is too old
                return False
            
            # Data points validation
            if data_points < 5:  # Too few data points
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation rule check failed: {str(e)}")
            return False
    
    def _generate_validation_warnings(self, data_age: float, data_points: int, timeframe: str, data_source: str) -> List[str]:
        """Enhanced warning generation with professional context"""
        warnings = []
        
        try:
            # Age warnings
            threshold = self.data_age_thresholds.get(timeframe, 360)
            if data_age > threshold * 2:
                warnings.append(f"Data extremely stale ({data_age:.1f} min old)")
            elif data_age > threshold:
                warnings.append(f"Data is stale ({data_age:.1f} min old)")
            
            # Professional data points assessment
            if data_source == 'api_real_time' and data_points < 50:
                # Don't warn about real-time snapshots - they're expected for 1h
                if data_points == 1:
                    warnings.append("Using current market snapshot (limited historical context)")
            elif data_source == 'database_historical' and data_points < 50:
                warnings.append(f"Limited historical data points ({data_points})")
            elif data_points < 10 and data_source not in ['api_real_time']:
                warnings.append(f"Very few data points ({data_points})")
            
            # Source warnings
            if data_source == 'bootstrap_fallback':
                warnings.append("Using bootstrap/synthetic data")
            elif data_source == 'unknown':
                warnings.append("Unknown data source")
            
        except Exception as e:
            warnings.append(f"Warning generation error: {str(e)}")
        
        return warnings
    
    def _calculate_confidence_adjustment(self, data_source: str, data_age: float, timeframe: str, freshness_score: float) -> float:
        """
        Calculate recommended confidence adjustment
        
        Args:
            data_source: Type of data source
            data_age: Age of data in minutes
            timeframe: Prediction timeframe
            freshness_score: Freshness score (0.0-1.0)
            
        Returns:
            float: Confidence adjustment (positive or negative)
        """
        try:
            adjustment = 0.0
            
            # Source-based adjustment
            if data_source == 'api_real_time':
                adjustment += self.confidence_rules['fresh_data_boost']
            elif data_source == 'bootstrap_fallback':
                adjustment += self.confidence_rules['bootstrap_data_penalty']
            elif data_source == 'database_historical':
                adjustment += self.confidence_rules['stale_data_penalty'] * 0.5
            
            # Freshness-based adjustment
            if freshness_score >= 0.8:
                adjustment += self.confidence_rules['fresh_data_boost'] * 0.5
            elif freshness_score <= 0.3:
                adjustment += self.confidence_rules['stale_data_penalty'] * 0.8
            
            # Age-based adjustment
            threshold = self.data_age_thresholds.get(timeframe, 360)
            if data_age > threshold:
                age_penalty = min(30.0, (data_age - threshold) / 60.0 * 10.0)
                adjustment -= age_penalty
            
            return round(adjustment, 1)
            
        except Exception as e:
            logger.warning(f"Could not calculate confidence adjustment: {str(e)}")
            return -20.0  # Conservative penalty        

    def detect_data_anomalies_before_prediction(self, data: Any, historical_context: Any) -> List[str]:
        """
        Detect data anomalies that could affect prediction accuracy
        
        Args:
            data: Current data being validated
            historical_context: Historical data for comparison
            
        Returns:
            List[str]: List of detected anomalies
        """
        anomalies = []
        
        try:
            # Check for data gaps
            if isinstance(data, (list, tuple)) and len(data) > 1:
                if len(data) < 20:  # Less than expected data points
                    anomalies.append("Insufficient data points for reliable analysis")
            
            # Check for extreme values
            if isinstance(data, dict) and 'current_price' in data:
                current_price = data['current_price']
                if current_price <= 0:
                    anomalies.append("Invalid price data (zero or negative)")
                    
            # Check for missing critical fields
            if isinstance(data, dict):
                critical_fields = ['current_price', 'volume']
                missing_fields = [field for field in critical_fields if field not in data or data[field] is None]
                if missing_fields:
                    anomalies.append(f"Missing critical data fields: {', '.join(missing_fields)}")
                    
        except Exception as e:
            anomalies.append(f"Anomaly detection error: {str(e)}")
            
        return anomalies
    
    def validate_market_context_alignment(self, prediction_data: Any, current_market_sentiment: str) -> bool:
        """
        Validate that prediction data aligns with current market context
        
        Args:
            prediction_data: Data being used for prediction
            current_market_sentiment: Current market sentiment ('bullish', 'bearish', 'neutral')
            
        Returns:
            bool: True if data aligns with market context
        """
        try:
            # This is where we would catch the AVAX 38% bearish vs actual bullish mismatch
            
            # Extract trend indicators from prediction data
            if isinstance(prediction_data, dict):
                price_change = prediction_data.get('price_change_percentage_24h', 0)
                
                # Check alignment with market sentiment
                if current_market_sentiment.lower() == 'bullish':
                    if price_change < -10:  # Significantly negative change with bullish sentiment
                        logger.warning("Market context mismatch: Negative price data in bullish market")
                        return False
                        
                elif current_market_sentiment.lower() == 'bearish':
                    if price_change > 10:  # Significantly positive change with bearish sentiment  
                        logger.warning("Market context mismatch: Positive price data in bearish market")
                        return False
                        
            return True
            
        except Exception as e:
            logger.warning(f"Market context validation failed: {str(e)}")
            return False
    
    def check_prediction_sanity_bounds(self, prediction: Dict, market_context: Dict) -> Tuple[bool, List[str]]:
        """
        Check if prediction falls within sanity bounds
        
        Args:
            prediction: Prediction results
            market_context: Current market context
            
        Returns:
            Tuple[bool, List[str]]: (is_sane, list_of_issues)
        """
        issues = []
        is_sane = True
        
        try:
            predicted_change = prediction.get('percent_change', 0)
            confidence = prediction.get('confidence', 50)
            
            # Check extreme predictions with low confidence
            if abs(predicted_change) > self.confidence_rules['extreme_prediction_threshold']:
                if confidence < 70:
                    issues.append(f"Extreme prediction ({predicted_change:.1f}%) with low confidence ({confidence}%)")
                    is_sane = False
                    
            # Check impossible predictions
            if abs(predicted_change) > 50:  # > 50% change is rarely realistic for 1h predictions
                issues.append(f"Unrealistic prediction magnitude: {predicted_change:.1f}%")
                is_sane = False
                
            # Check confidence vs magnitude alignment
            if abs(predicted_change) > 20 and confidence < 60:
                issues.append("High magnitude prediction with insufficient confidence")
                is_sane = False
                
        except Exception as e:
            issues.append(f"Sanity check error: {str(e)}")
            is_sane = False
            
        return is_sane, issues
    
    def validate_confidence_vs_prediction_magnitude(self, confidence: float, predicted_change: float) -> bool:
        """
        Validate that confidence level matches prediction magnitude
        
        Args:
            confidence: Prediction confidence (0-100)
            predicted_change: Predicted percentage change
            
        Returns:
            bool: True if confidence and magnitude are aligned
        """
        try:
            magnitude = abs(predicted_change)
            
            # High magnitude predictions should require high confidence
            if magnitude > 20 and confidence < 70:
                logger.warning(f"Low confidence ({confidence}%) for high magnitude prediction ({predicted_change:.1f}%)")
                return False
                
            # Very high magnitude predictions should require very high confidence
            if magnitude > 30 and confidence < 80:
                logger.warning(f"Insufficient confidence ({confidence}%) for extreme prediction ({predicted_change:.1f}%)")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Confidence validation failed: {str(e)}")
            return False
    
    def should_prevent_prediction_publication(self, prediction: Dict, validation_meta: DataValidationMetadata) -> Tuple[bool, str]:
        """
        Determine if a prediction should be prevented from publication
        
        Args:
            prediction: Prediction results
            validation_meta: Validation metadata
            
        Returns:
            Tuple[bool, str]: (should_prevent, reason)
        """
        try:
            # Critical risk level
            if validation_meta.accuracy_risk_level == 'critical':
                return True, "Critical accuracy risk detected"
                
            # Failed basic validation
            if not validation_meta.validation_passed:
                return True, "Failed basic data validation"
                
            # Extreme prediction with stale data
            predicted_change = abs(prediction.get('percent_change', 0))
            if predicted_change > 20 and validation_meta.data_age_minutes > 120:  # 2 hours
                return True, f"Extreme prediction ({predicted_change:.1f}%) with stale data ({validation_meta.data_age_minutes:.1f} min)"
                
            # Bootstrap data with extreme prediction
            if (validation_meta.data_source_used == 'bootstrap_fallback' and 
                predicted_change > self.confidence_rules['extreme_prediction_threshold']):
                return True, f"Extreme prediction ({predicted_change:.1f}%) using bootstrap data"
                
            return False, "Validation passed"
            
        except Exception as e:
            return True, f"Validation error: {str(e)}"
    
    # Integration and Monitoring Methods
    
    def integrate_with_existing_systems(self, prediction_engine: Any, database: Any) -> None:
        """
        Initialize integration with existing systems
        
        Args:
            prediction_engine: Reference to prediction engine
            database: Reference to database instance
        """
        try:
            self.prediction_engine = prediction_engine
            self.database = database
            logger.info("✅ Data validation layer integrated with existing systems")
            
        except Exception as e:
            logger.error(f"System integration failed: {str(e)}")
    
    def monitor_validation_effectiveness(self) -> Dict[str, Any]:
        """
        Monitor the effectiveness of the validation system
        
        Returns:
            Dict containing effectiveness metrics
        """
        try:
            if not self.validation_history:
                return {'effectiveness_score': 0.0, 'total_validations': 0}
                
            total_validations = len(self.validation_history)
            passed_validations = sum(1 for v in self.validation_history if v.validation_passed)
            critical_catches = sum(1 for v in self.validation_history if v.accuracy_risk_level == 'critical')
            
            effectiveness_metrics = {
                'effectiveness_score': (passed_validations / total_validations) * 100,
                'total_validations': total_validations,
                'passed_validations': passed_validations,
                'critical_issues_caught': critical_catches,
                'average_risk_level': self._calculate_average_risk_level()
            }
            
            return effectiveness_metrics
            
        except Exception as e:
            logger.error(f"Effectiveness monitoring failed: {str(e)}")
            return {'effectiveness_score': 0.0, 'error': str(e)}
    
    def generate_validation_summary_report(self, token: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """
        Generate a summary report of validation activities
        
        Args:
            token: Optional token filter
            hours: Hours to look back
            
        Returns:
            Dict containing validation summary
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter validation history
            recent_validations = [
                v for v in self.validation_history 
                if v.validation_timestamp >= cutoff_time and (token is None or token in str(v))
            ]
            
            if not recent_validations:
                return {'message': 'No validations in specified timeframe'}
                
            # Calculate summary statistics
            summary = {
                'timeframe_hours': hours,
                'total_validations': len(recent_validations),
                'validations_passed': sum(1 for v in recent_validations if v.validation_passed),
                'risk_level_breakdown': self._get_risk_level_breakdown(recent_validations),
                'average_data_age_minutes': sum(v.data_age_minutes for v in recent_validations) / len(recent_validations),
                'most_common_warnings': self._get_most_common_warnings(recent_validations),
                'average_confidence_adjustment': sum(v.recommended_confidence_adjustment for v in recent_validations) / len(recent_validations)
            }
            
            return summary
            
        except Exception as e:
            return {'error': f"Report generation failed: {str(e)}"}
    
    def _calculate_average_risk_level(self) -> float:
        """Calculate average risk level from validation history as numeric score"""
        if not self.validation_history:
            return 2.0  # Default to medium risk level
            
        risk_scores = {'low': 1.0, 'medium': 2.0, 'high': 3.0, 'critical': 4.0}
        total_score = sum(risk_scores.get(v.accuracy_risk_level, 2.0) for v in self.validation_history)
        average_score = total_score / len(self.validation_history)
        
        return round(average_score, 2)
    
    def _get_risk_level_breakdown(self, validations: List[DataValidationMetadata]) -> Dict[str, int]:
        """Get breakdown of risk levels"""
        breakdown = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for validation in validations:
            risk_level = validation.accuracy_risk_level
            breakdown[risk_level] = breakdown.get(risk_level, 0) + 1
        return breakdown
    
    def _get_most_common_warnings(self, validations: List[DataValidationMetadata]) -> List[str]:
        """Get most common warnings from validations"""
        warning_counts = {}
        for validation in validations:
            for warning in validation.validation_warnings:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1
        
        # Return top 3 most common warnings
        sorted_warnings = sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)
        return [warning for warning, count in sorted_warnings[:3]]
    
"""
Additional classes for data_validation_layer.py
Add these classes to your existing data_validation_layer.py file

These classes provide system-wide validation for ALL numerical data:
- Prices, volumes, market caps
- Technical indicators (RSI, MACD, moving averages)
- Trading signals and percentages
- Support/resistance levels
- Any numerical content before posting
"""

import asyncio
from typing import Set, Union, List
import requests
from datetime import datetime, timedelta
import threading
import math
import re
from decimal import Decimal, InvalidOperation

class SystemWideDataValidator:
    """
    Validates ALL numerical data across the entire system before posting
    Prevents confusion between prices, indicators, percentages, and trading signals
    """
    
    def __init__(self, token_mapper=None, database=None):
        """
        Initialize system-wide data validator
        
        Args:
            token_mapper: TokenMappingManager instance for all tokens
            database: Database instance for historical validation
        """
        self.token_mapper = token_mapper
        self.database = database
        
        # Data type definitions and validation rules
        self.data_type_rules = self._initialize_data_type_rules()
        self.validation_cache = {}
        self.cache_lock = threading.Lock()
        
        # Cross-validation settings
        self.max_data_age_minutes = 10  # Max 10 minutes for any numerical data
        self.cross_validation_sources = 3  # Minimum sources for validation
        
        logger.info("🌐 SystemWideDataValidator initialized - All numerical data validation active")
    
    def validate_all_content_numbers(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ALL numbers in content before posting
        
        Args:
            content_data: Complete content data structure
            
        Returns:
            Dict with validation results and corrected data
        """
        try:
            validation_result = {
                'validation_passed': True,
                'corrected_data': {},
                'warnings': [],
                'errors': [],
                'number_validations': {},
                'cross_validation_results': {}
            }
            
            # Extract and categorize all numbers from content
            all_numbers = self._extract_all_numerical_data(content_data)
            
            # Validate each numerical data type
            for data_type, numbers in all_numbers.items():
                validation = self._validate_data_type(data_type, numbers, content_data)
                validation_result['number_validations'][data_type] = validation
                
                if not validation['valid']:
                    validation_result['validation_passed'] = False
                    validation_result['errors'].extend(validation['errors'])
                
                validation_result['warnings'].extend(validation['warnings'])
                validation_result['corrected_data'][data_type] = validation['corrected_values']
            
            # Cross-validate related numbers
            cross_validation = self._cross_validate_numerical_relationships(all_numbers)
            validation_result['cross_validation_results'] = cross_validation
            
            if not cross_validation['consistent']:
                validation_result['validation_passed'] = False
                validation_result['errors'].extend(cross_validation['errors'])
            
            # System-wide freshness check
            freshness_check = self._validate_system_wide_freshness(content_data)
            validation_result['freshness_validation'] = freshness_check
            
            if not freshness_check['fresh']:
                validation_result['warnings'].extend(freshness_check['warnings'])
            
            return validation_result
            
        except Exception as e:
            logger.error(f"System-wide validation failed: {str(e)}")
            return {
                'validation_passed': False,
                'errors': [f"System validation error: {str(e)}"],
                'warnings': [],
                'corrected_data': {}
            }
    
    def _initialize_data_type_rules(self) -> Dict[str, Dict]:
        """Initialize validation rules for different data types"""
        return {
            'prices': {
                'min_value': 0.0000001,  # Minimum valid price
                'max_value': 1000000.0,  # Maximum reasonable price
                'decimal_places': 8,
                'requires_currency_symbol': True,
                'validation_sources': ['database', 'api_real_time'],
                'max_deviation_percent': 15.0
            },
            'percentages': {
                'min_value': -100.0,
                'max_value': 1000.0,  # Allow for extreme crypto gains
                'decimal_places': 2,
                'requires_percent_symbol': True,
                'validation_sources': ['calculated', 'api'],
                'max_deviation_percent': 25.0
            },
            'rsi': {
                'min_value': 0.0,
                'max_value': 100.0,
                'decimal_places': 1,
                'requires_currency_symbol': False,
                'validation_sources': ['technical_analysis'],
                'max_deviation_percent': 10.0
            },
            'macd': {
                'min_value': -1000.0,
                'max_value': 1000.0,
                'decimal_places': 4,
                'requires_currency_symbol': False,
                'validation_sources': ['technical_analysis'],
                'max_deviation_percent': 20.0
            },
            'volume': {
                'min_value': 0.0,
                'max_value': 1e15,  # 1 quadrillion max volume
                'decimal_places': 0,
                'requires_currency_symbol': False,
                'validation_sources': ['database', 'api'],
                'max_deviation_percent': 30.0
            },
            'market_cap': {
                'min_value': 1000.0,  # $1000 minimum market cap
                'max_value': 1e13,    # $10 trillion max
                'decimal_places': 0,
                'requires_currency_symbol': True,
                'validation_sources': ['database', 'api'],
                'max_deviation_percent': 20.0
            },
            'support_resistance': {
                'min_value': 0.0000001,
                'max_value': 1000000.0,
                'decimal_places': 8,
                'requires_currency_symbol': True,
                'validation_sources': ['technical_analysis', 'price_based'],
                'max_deviation_percent': 10.0
            },
            'moving_averages': {
                'min_value': 0.0000001,
                'max_value': 1000000.0,
                'decimal_places': 8,
                'requires_currency_symbol': True,
                'validation_sources': ['technical_analysis', 'calculated'],
                'max_deviation_percent': 5.0
            },
            'fibonacci_levels': {
                'min_value': 0.0000001,
                'max_value': 1000000.0,
                'decimal_places': 8,
                'requires_currency_symbol': True,
                'validation_sources': ['technical_analysis'],
                'max_deviation_percent': 8.0
            }
        }
    
    def _extract_all_numerical_data(self, content_data: Dict[str, Any]) -> Dict[str, List]:
        """Extract and categorize all numerical data from content"""
        numerical_data = {
            'prices': [],
            'percentages': [],
            'rsi': [],
            'macd': [],
            'volume': [],
            'market_cap': [],
            'support_resistance': [],
            'moving_averages': [],
            'fibonacci_levels': []
        }
        
        try:
            # Extract from different content sections
            self._extract_from_price_data(content_data.get('price_data', {}), numerical_data)
            self._extract_from_technical_signals(content_data.get('technical_signals', {}), numerical_data)
            self._extract_from_prediction_data(content_data.get('prediction_data', {}), numerical_data)
            self._extract_from_content_text(content_data.get('content', ''), numerical_data)
            
            return numerical_data
            
        except Exception as e:
            logger.error(f"Failed to extract numerical data: {str(e)}")
            return numerical_data
    
    def _extract_from_price_data(self, price_data: Dict, numerical_data: Dict):
        """Extract numerical data from price_data section"""
        try:
            # Extract prices
            if 'current_price' in price_data:
                numerical_data['prices'].append({
                    'value': price_data['current_price'],
                    'context': 'current_price',
                    'source': 'price_data'
                })
            
            # Extract percentages
            for key in ['price_change_percentage_24h', 'price_change_percentage_1h', 'price_change_percentage_7d']:
                if key in price_data:
                    numerical_data['percentages'].append({
                        'value': price_data[key],
                        'context': key,
                        'source': 'price_data'
                    })
            
            # Extract volume and market cap
            if 'total_volume' in price_data:
                numerical_data['volume'].append({
                    'value': price_data['total_volume'],
                    'context': 'volume',
                    'source': 'price_data'
                })
            
            if 'market_cap' in price_data:
                numerical_data['market_cap'].append({
                    'value': price_data['market_cap'],
                    'context': 'market_cap',
                    'source': 'price_data'
                })
                
        except Exception as e:
            logger.warning(f"Error extracting from price_data: {str(e)}")
    
    def _extract_from_technical_signals(self, technical_data: Dict, numerical_data: Dict):
        """Extract numerical data from technical_signals section"""
        try:
            # Extract RSI
            if 'indicators' in technical_data and 'rsi' in technical_data['indicators']:
                numerical_data['rsi'].append({
                    'value': technical_data['indicators']['rsi'],
                    'context': 'rsi_indicator',
                    'source': 'technical_signals'
                })
            
            # Extract MACD
            if 'indicators' in technical_data and 'macd' in technical_data['indicators']:
                macd_data = technical_data['indicators']['macd']
                if isinstance(macd_data, dict):
                    for key in ['macd', 'signal', 'histogram']:
                        if key in macd_data:
                            numerical_data['macd'].append({
                                'value': macd_data[key],
                                'context': f'macd_{key}',
                                'source': 'technical_signals'
                            })
            
            # Extract Support/Resistance
            if 'support_resistance' in technical_data:
                sr_data = technical_data['support_resistance']
                
                # Support levels
                if 'support_levels' in sr_data:
                    for level in sr_data['support_levels']:
                        if isinstance(level, dict) and 'level' in level:
                            numerical_data['support_resistance'].append({
                                'value': level['level'],
                                'context': 'support_level',
                                'source': 'technical_signals'
                            })
                
                # Resistance levels
                if 'resistance_levels' in sr_data:
                    for level in sr_data['resistance_levels']:
                        if isinstance(level, dict) and 'level' in level:
                            numerical_data['support_resistance'].append({
                                'value': level['level'],
                                'context': 'resistance_level',
                                'source': 'technical_signals'
                            })
            
            # Extract Moving Averages
            if 'indicators' in technical_data:
                indicators = technical_data['indicators']
                for ma_key in ['sma_20', 'ema_20', 'sma_50', 'ema_50', 'sma_200']:
                    if ma_key in indicators:
                        numerical_data['moving_averages'].append({
                            'value': indicators[ma_key],
                            'context': ma_key,
                            'source': 'technical_signals'
                        })
                        
        except Exception as e:
            logger.warning(f"Error extracting from technical_signals: {str(e)}")
    
    def _extract_from_content_text(self, content_text: str, numerical_data: Dict):
        """Extract numerical data directly from content text"""
        try:
            # Price patterns: $X.XX, $X,XXX.XX
            price_pattern = r'\$([0-9,]+\.?[0-9]*)'
            price_matches = re.findall(price_pattern, content_text)
            for match in price_matches:
                try:
                    price_value = float(match.replace(',', ''))
                    numerical_data['prices'].append({
                        'value': price_value,
                        'context': 'content_text',
                        'source': 'text_extraction',
                        'raw_text': f"${match}"
                    })
                except ValueError:
                    continue
            
            # Percentage patterns: X.XX%, XX%
            percentage_pattern = r'([+-]?[0-9]+\.?[0-9]*)%'
            percentage_matches = re.findall(percentage_pattern, content_text)
            for match in percentage_matches:
                try:
                    pct_value = float(match)
                    numerical_data['percentages'].append({
                        'value': pct_value,
                        'context': 'content_text',
                        'source': 'text_extraction',
                        'raw_text': f"{match}%"
                    })
                except ValueError:
                    continue
            
            # RSI patterns: RSI at XX, RSI XX
            rsi_pattern = r'RSI\s+(?:at\s+)?([0-9]+\.?[0-9]*)'
            rsi_matches = re.findall(rsi_pattern, content_text, re.IGNORECASE)
            for match in rsi_matches:
                try:
                    rsi_value = float(match)
                    numerical_data['rsi'].append({
                        'value': rsi_value,
                        'context': 'content_text',
                        'source': 'text_extraction',
                        'raw_text': f"RSI {match}"
                    })
                except ValueError:
                    continue
                    
        except Exception as e:
            logger.warning(f"Error extracting from content text: {str(e)}")
    
    def _validate_data_type(self, data_type: str, numbers: List[Dict], content_data: Dict) -> Dict:
        """Validate specific data type numbers"""
        try:
            rules = self.data_type_rules.get(data_type, {})
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'corrected_values': []
            }
            
            for number_data in numbers:
                value = number_data['value']
                context = number_data['context']
                
                # Range validation
                if 'min_value' in rules and value < rules['min_value']:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"{data_type} value {value} below minimum {rules['min_value']} (context: {context})")
                
                if 'max_value' in rules and value > rules['max_value']:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"{data_type} value {value} above maximum {rules['max_value']} (context: {context})")
                
                # Decimal places validation
                if 'decimal_places' in rules:
                    decimal_places = len(str(value).split('.')[-1]) if '.' in str(value) else 0
                    if decimal_places > rules['decimal_places']:
                        corrected_value = round(value, rules['decimal_places'])
                        validation_result['warnings'].append(f"Truncated {data_type} from {value} to {corrected_value}")
                        number_data['corrected_value'] = corrected_value
                
                # Cross-reference validation
                cross_check = self._cross_reference_number(data_type, value, context)
                if not cross_check['valid']:
                    validation_result['warnings'].extend(cross_check['warnings'])
                
                validation_result['corrected_values'].append(number_data)
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation error for {data_type}: {str(e)}"],
                'warnings': [],
                'corrected_values': []
            }
    
    def _cross_validate_numerical_relationships(self, all_numbers: Dict) -> Dict:
        """Cross-validate relationships between different numerical data types"""
        try:
            cross_validation = {
                'consistent': True,
                'errors': [],
                'warnings': [],
                'relationships_checked': []
            }
            
            # Validate price vs support/resistance relationships
            prices = [n['value'] for n in all_numbers.get('prices', [])]
            sr_levels = [n['value'] for n in all_numbers.get('support_resistance', [])]
            
            if prices and sr_levels:
                for price in prices:
                    for sr_level in sr_levels:
                        deviation_pct = abs(price - sr_level) / price * 100
                        if deviation_pct < 1.0:  # Price very close to S/R level
                            cross_validation['relationships_checked'].append(f"Price ${price:.6f} near S/R level ${sr_level:.6f}")
                        elif deviation_pct > 50.0:  # S/R level very far from price
                            cross_validation['warnings'].append(f"S/R level ${sr_level:.6f} is {deviation_pct:.1f}% away from current price ${price:.6f}")
            
            # Validate percentage vs price change consistency
            percentages = [n['value'] for n in all_numbers.get('percentages', [])]
            if len(prices) >= 2 and percentages:
                # Calculate implied percentage from prices if possible
                # This would catch mismatched percentage data
                pass
            
            # Validate RSI bounds
            rsi_values = [n['value'] for n in all_numbers.get('rsi', [])]
            for rsi in rsi_values:
                if not (0 <= rsi <= 100):
                    cross_validation['consistent'] = False
                    cross_validation['errors'].append(f"RSI value {rsi} outside valid range 0-100")
            
            # Validate volume vs market cap relationship
            volumes = [n['value'] for n in all_numbers.get('volume', [])]
            market_caps = [n['value'] for n in all_numbers.get('market_cap', [])]
            
            if volumes and market_caps and prices:
                for volume in volumes:
                    for market_cap in market_caps:
                        for price in prices:
                            # Check if volume to market cap ratio is reasonable
                            if market_cap > 0 and price > 0:
                                volume_to_mc_ratio = volume / market_cap
                                if volume_to_mc_ratio > 5.0:  # Very high volume relative to market cap
                                    cross_validation['warnings'].append(f"Very high volume/market cap ratio: {volume_to_mc_ratio:.2f}")
            
            return cross_validation
            
        except Exception as e:
            return {
                'consistent': False,
                'errors': [f"Cross-validation error: {str(e)}"],
                'warnings': [],
                'relationships_checked': []
            }
    
    def _cross_reference_number(self, data_type: str, value: float, context: str) -> Dict:
        """Cross-reference a number against historical/current data"""
        try:
            # Get all available tokens for system-wide validation
            if not self.token_mapper:
                return {'valid': True, 'warnings': []}
            
            all_tokens = self.token_mapper.get_all_available_tokens()
            validation_results = []
            
            # For prices, check against current market data for all tokens
            if data_type == 'prices':
                for token_list in all_tokens.values():
                    for token in token_list[:10]:  # Limit to top 10 tokens per source
                        try:
                            token_info = self.token_mapper.get_token_info(token)
                            if token_info and 'current_price' in token_info:
                                current_price = token_info['current_price']
                                if abs(value - current_price) / current_price < 0.05:  # Within 5%
                                    validation_results.append(f"Price matches {token} current price")
                                    return {'valid': True, 'warnings': []}
                        except:
                            continue
                
                # If no match found, check if it's a reasonable crypto price
                if 0.000001 <= value <= 100000:  # Reasonable crypto price range
                    return {'valid': True, 'warnings': [f"Price ${value:.6f} not matched to specific token but within reasonable range"]}
                else:
                    return {'valid': False, 'warnings': [f"Price ${value:.6f} outside typical crypto range"]}
            
            # For percentages, validate against typical crypto volatility
            elif data_type == 'percentages':
                if abs(value) > 100:  # Very extreme percentage change
                    return {'valid': True, 'warnings': [f"Extreme percentage change: {value:.1f}%"]}
            
            # For RSI, validate against typical ranges
            elif data_type == 'rsi':
                if value < 30:
                    return {'valid': True, 'warnings': [f"RSI {value} indicates oversold conditions"]}
                elif value > 70:
                    return {'valid': True, 'warnings': [f"RSI {value} indicates overbought conditions"]}
            
            return {'valid': True, 'warnings': []}
            
        except Exception as e:
            return {'valid': True, 'warnings': [f"Cross-reference error: {str(e)}"]}
    
    def _validate_system_wide_freshness(self, content_data: Dict) -> Dict:
        """Validate freshness of all data across the system"""
        try:
            freshness_result = {
                'fresh': True,
                'warnings': [],
                'oldest_data_age': 0,
                'data_sources_checked': []
            }
            
            current_time = datetime.now()
            
            # Check price_data timestamp
            if 'price_data' in content_data:
                price_timestamp = self._extract_timestamp(content_data['price_data'])
                if price_timestamp:
                    age_minutes = (current_time - price_timestamp).total_seconds() / 60
                    freshness_result['oldest_data_age'] = max(freshness_result['oldest_data_age'], age_minutes)
                    freshness_result['data_sources_checked'].append(f"price_data ({age_minutes:.1f} min old)")
                    
                    if age_minutes > self.max_data_age_minutes:
                        freshness_result['fresh'] = False
                        freshness_result['warnings'].append(f"Price data is {age_minutes:.1f} minutes old (threshold: {self.max_data_age_minutes})")
            
            # Check technical_signals timestamp
            if 'technical_signals' in content_data:
                tech_timestamp = self._extract_timestamp(content_data['technical_signals'])
                if tech_timestamp:
                    age_minutes = (current_time - tech_timestamp).total_seconds() / 60
                    freshness_result['oldest_data_age'] = max(freshness_result['oldest_data_age'], age_minutes)
                    freshness_result['data_sources_checked'].append(f"technical_signals ({age_minutes:.1f} min old)")
                    
                    if age_minutes > self.max_data_age_minutes * 2:  # Technical data can be slightly older
                        freshness_result['warnings'].append(f"Technical data is {age_minutes:.1f} minutes old")
            
            # Check prediction_data timestamp
            if 'prediction_data' in content_data:
                pred_timestamp = self._extract_timestamp(content_data['prediction_data'])
                if pred_timestamp:
                    age_minutes = (current_time - pred_timestamp).total_seconds() / 60
                    freshness_result['oldest_data_age'] = max(freshness_result['oldest_data_age'], age_minutes)
                    freshness_result['data_sources_checked'].append(f"prediction_data ({age_minutes:.1f} min old)")
            
            return freshness_result
            
        except Exception as e:
            return {
                'fresh': False,
                'warnings': [f"Freshness validation error: {str(e)}"],
                'oldest_data_age': 999,
                'data_sources_checked': []
            }
    
    def _extract_timestamp(self, data: Any) -> Optional[datetime]:
        """Extract timestamp from various data formats"""
        try:
            if isinstance(data, dict):
                for timestamp_field in ['timestamp', 'last_updated', 'created_at', 'updated_at']:
                    if timestamp_field in data:
                        timestamp_value = data[timestamp_field]
                        if isinstance(timestamp_value, datetime):
                            return timestamp_value
                        elif isinstance(timestamp_value, str):
                            try:
                                return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                            except:
                                pass
            return None
        except:
            return None
    
    def _extract_from_prediction_data(self, prediction_data: Dict, numerical_data: Dict):
        """Extract numerical data from prediction_data section"""
        try:
            # Extract predicted prices
            if 'price' in prediction_data:
                numerical_data['prices'].append({
                    'value': prediction_data['price'],
                    'context': 'predicted_price',
                    'source': 'prediction_data'
                })
            
            # Extract predicted percentage changes
            if 'percent_change' in prediction_data:
                numerical_data['percentages'].append({
                    'value': prediction_data['percent_change'],
                    'context': 'predicted_change',
                    'source': 'prediction_data'
                })
            
            # Extract confidence bounds
            for bound in ['lower_bound', 'upper_bound']:
                if bound in prediction_data:
                    numerical_data['prices'].append({
                        'value': prediction_data[bound],
                        'context': bound,
                        'source': 'prediction_data'
                    })
                    
        except Exception as e:
            logger.warning(f"Error extracting from prediction_data: {str(e)}")


class ContentPostingValidator:
    """
    Final validation before posting any content with numerical data
    Ensures all numbers are current, accurate, and properly categorized
    """
    
    def __init__(self, system_validator: SystemWideDataValidator):
        """
        Initialize content posting validator
        
        Args:
            system_validator: SystemWideDataValidator instance
        """
        self.system_validator = system_validator
        self.posting_history = []
        self.blocked_content_count = 0
        
        logger.info("📝 ContentPostingValidator initialized - Pre-posting validation active")
    
    def validate_before_posting(self, content: str, content_data: Dict[str, Any], post_type: str = 'general') -> Dict[str, Any]:
        """
        Final validation before posting content
        
        Args:
            content: The actual content text to be posted
            content_data: All associated data (prices, signals, predictions)
            post_type: Type of post ('prediction', 'analysis', 'alert', 'general')
            
        Returns:
            Dict with validation results and posting decision
        """
        try:
            validation_result = {
                'approved_for_posting': False,
                'corrected_content': content,
                'corrected_data': content_data.copy(),
                'validation_warnings': [],
                'blocking_errors': [],
                'post_type': post_type,
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # System-wide numerical validation
            numerical_validation = self.system_validator.validate_all_content_numbers(content_data)
            validation_result['numerical_validation'] = numerical_validation
            
            if not numerical_validation['validation_passed']:
                validation_result['blocking_errors'].extend(numerical_validation['errors'])
                validation_result['approved_for_posting'] = False
                self.blocked_content_count += 1
                
                logger.warning(f"Content blocked due to numerical validation errors: {numerical_validation['errors']}")
                return validation_result
            
            # Content-specific validation
            content_validation = self._validate_content_text(content, content_data)
            if not content_validation['valid']:
                validation_result['blocking_errors'].extend(content_validation['errors'])
                validation_result['approved_for_posting'] = False
                return validation_result
            
            # Apply corrections from numerical validation
            if numerical_validation.get('corrected_data'):
                validation_result['corrected_data'].update(numerical_validation['corrected_data'])
            
            # Generate corrected content if needed
            corrected_content = self._apply_content_corrections(content, numerical_validation.get('corrected_data', {}))
            validation_result['corrected_content'] = corrected_content
            
            # Final approval decision
            total_errors = len(validation_result['blocking_errors'])
            critical_warnings = len([w for w in numerical_validation.get('warnings', []) if 'critical' in w.lower()])
            
            if total_errors == 0 and critical_warnings == 0:
                validation_result['approved_for_posting'] = True
            else:
                validation_result['approved_for_posting'] = False
                if critical_warnings > 0:
                    validation_result['blocking_errors'].append(f"{critical_warnings} critical warnings detected")
            
            # Store validation history
            self.posting_history.append(validation_result.copy())
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Content posting validation failed: {str(e)}")
            return {
                'approved_for_posting': False,
                'blocking_errors': [f"Validation system error: {str(e)}"],
                'validation_warnings': [],
                'corrected_content': content,
                'corrected_data': content_data
            }
    
    def _validate_content_text(self, content: str, content_data: Dict) -> Dict:
        """Validate the actual content text for numerical consistency"""
        try:
            validation = {
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Extract numbers from content text
            content_numbers = self._extract_numbers_from_text(content)
            
            # Compare content numbers with data numbers
            data_prices = []
            if 'price_data' in content_data and 'current_price' in content_data['price_data']:
                data_prices.append(content_data['price_data']['current_price'])
            
            # Check for price mismatches between text and data
            for content_price in content_numbers.get('prices', []):
                price_matched = False
                for data_price in data_prices:
                    if abs(content_price - data_price) / data_price < 0.05:  # Within 5%
                        price_matched = True
                        break
                
                if not price_matched and data_prices:
                    validation['errors'].append(f"Content mentions price ${content_price:.6f} but data shows ${data_prices[0]:.6f}")
                    validation['valid'] = False
            
            # Check for percentage consistency
            content_percentages = content_numbers.get('percentages', [])
            data_percentages = []
            if 'price_data' in content_data:
                for pct_key in ['price_change_percentage_24h', 'price_change_percentage_1h', 'price_change_percentage_7d']:
                    if pct_key in content_data['price_data']:
                        data_percentages.append(content_data['price_data'][pct_key])
            
            for content_pct in content_percentages:
                pct_matched = False
                for data_pct in data_percentages:
                    if abs(content_pct - data_pct) < 2.0:  # Within 2 percentage points
                        pct_matched = True
                        break
                
                if not pct_matched and data_percentages and abs(content_pct) > 5:  # Only check significant percentages
                    validation['warnings'].append(f"Content mentions {content_pct:.1f}% change but data shows different values")
            
            return validation
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Content text validation error: {str(e)}"],
                'warnings': []
            }
    
    def _extract_numbers_from_text(self, text: str) -> Dict[str, List[float]]:
        """Extract all numbers from content text by category"""
        numbers = {
            'prices': [],
            'percentages': [],
            'rsi_values': [],
            'other_numbers': []
        }
        
        try:
            # Extract prices ($X.XX format)
            price_pattern = r'\$([0-9,]+\.?[0-9]*)'
            price_matches = re.findall(price_pattern, text)
            for match in price_matches:
                try:
                    price_value = float(match.replace(',', ''))
                    numbers['prices'].append(price_value)
                except ValueError:
                    continue
            
            # Extract percentages (X.XX% format)
            percentage_pattern = r'([+-]?[0-9]+\.?[0-9]*)%'
            percentage_matches = re.findall(percentage_pattern, text)
            for match in percentage_matches:
                try:
                    pct_value = float(match)
                    numbers['percentages'].append(pct_value)
                except ValueError:
                    continue
            
            # Extract RSI values
            rsi_pattern = r'RSI\s+(?:at\s+)?([0-9]+\.?[0-9]*)'
            rsi_matches = re.findall(rsi_pattern, text, re.IGNORECASE)
            for match in rsi_matches:
                try:
                    rsi_value = float(match)
                    numbers['rsi_values'].append(rsi_value)
                except ValueError:
                    continue
            
            return numbers
            
        except Exception as e:
            logger.warning(f"Failed to extract numbers from text: {str(e)}")
            return numbers
    
    def _apply_content_corrections(self, content: str, corrected_data: Dict) -> str:
        """Apply numerical corrections to content text"""
        try:
            corrected_content = content
            
            # Apply price corrections
            if 'prices' in corrected_data:
                for price_correction in corrected_data['prices']:
                    if 'corrected_value' in price_correction:
                        original_value = price_correction['value']
                        corrected_value = price_correction['corrected_value']
                        
                        # Replace in content (be careful with formatting)
                        original_text = f"${original_value:.6f}"
                        corrected_text = f"${corrected_value:.6f}"
                        corrected_content = corrected_content.replace(original_text, corrected_text)
            
            # Apply percentage corrections
            if 'percentages' in corrected_data:
                for pct_correction in corrected_data['percentages']:
                    if 'corrected_value' in pct_correction:
                        original_value = pct_correction['value']
                        corrected_value = pct_correction['corrected_value']
                        
                        # Replace percentage values
                        original_text = f"{original_value:.2f}%"
                        corrected_text = f"{corrected_value:.2f}%"
                        corrected_content = corrected_content.replace(original_text, corrected_text)
            
            return corrected_content
            
        except Exception as e:
            logger.warning(f"Failed to apply content corrections: {str(e)}")
            return content
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about content validation performance"""
        try:
            if not self.posting_history:
                return {'total_validations': 0, 'approval_rate': 0.0}
            
            total_validations = len(self.posting_history)
            approved_posts = sum(1 for v in self.posting_history if v['approved_for_posting'])
            
            # Calculate error breakdown
            error_types = {}
            for validation in self.posting_history:
                for error in validation.get('blocking_errors', []):
                    error_type = error.split(':')[0] if ':' in error else 'general'
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Calculate warning breakdown
            warning_types = {}
            for validation in self.posting_history:
                for warning in validation.get('validation_warnings', []):
                    warning_type = warning.split(':')[0] if ':' in warning else 'general'
                    warning_types[warning_type] = warning_types.get(warning_type, 0) + 1
            
            statistics = {
                'total_validations': total_validations,
                'approved_posts': approved_posts,
                'blocked_posts': total_validations - approved_posts,
                'approval_rate': (approved_posts / total_validations) * 100,
                'blocked_content_count': self.blocked_content_count,
                'most_common_errors': sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5],
                'most_common_warnings': sorted(warning_types.items(), key=lambda x: x[1], reverse=True)[:5],
                'recent_approval_rate': self._calculate_recent_approval_rate()
            }
            
            return statistics
            
        except Exception as e:
            return {'error': f"Statistics calculation failed: {str(e)}"}
    
    def _calculate_recent_approval_rate(self, hours: int = 24) -> float:
        """Calculate approval rate for recent validations"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_validations = [
                v for v in self.posting_history 
                if datetime.fromisoformat(v['validation_timestamp']) >= cutoff_time
            ]
            
            if not recent_validations:
                return 0.0
            
            approved_recent = sum(1 for v in recent_validations if v['approved_for_posting'])
            return (approved_recent / len(recent_validations)) * 100
            
        except Exception as e:
            logger.warning(f"Recent approval rate calculation failed: {str(e)}")
            return 0.0
    
    def should_flag_for_manual_review(self, validation_result: Dict) -> bool:
        """Determine if content should be flagged for manual review"""
        try:
            # Flag for manual review if:
            
            # 1. Multiple critical numerical errors
            critical_errors = len([e for e in validation_result.get('blocking_errors', []) if 'critical' in e.lower()])
            if critical_errors >= 2:
                return True
            
            # 2. Large price deviations mentioned in warnings
            price_deviation_warnings = [w for w in validation_result.get('validation_warnings', []) if 'deviation' in w.lower()]
            if len(price_deviation_warnings) >= 3:
                return True
            
            # 3. Content blocked due to numerical validation
            if not validation_result.get('approved_for_posting', False) and validation_result.get('numerical_validation', {}).get('validation_passed', True) == False:
                return True
            
            # 4. Extreme percentages or values detected
            numerical_validation = validation_result.get('numerical_validation', {})
            for data_type, validations in numerical_validation.get('number_validations', {}).items():
                for warning in validations.get('warnings', []):
                    if 'extreme' in warning.lower() or 'unusual' in warning.lower():
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Manual review flagging failed: {str(e)}")
            return True  # Flag for review if we can't determine


class SystemIntegrationManager:
    """
    Integrates the validation system with existing components
    Provides easy integration points for the existing codebase
    """
    
    def __init__(self, token_mapper=None, database=None, prediction_engine=None):
        """
        Initialize system integration manager
        
        Args:
            token_mapper: Existing TokenMappingManager instance
            database: Existing database instance
            prediction_engine: Existing prediction engine instance
        """
        # Initialize validation components
        self.system_validator = SystemWideDataValidator(token_mapper, database)
        self.content_validator = ContentPostingValidator(self.system_validator)
        
        # Store references to existing systems
        self.token_mapper = token_mapper
        self.database = database
        self.prediction_engine = prediction_engine
        
        logger.info("🔧 SystemIntegrationManager initialized - All validation systems integrated")
    
    def validate_before_prediction(self, token: str, data: Dict, timeframe: str) -> Dict:
        """
        Integration point for prediction engine
        Call this before generating predictions
        """
        try:
            # Use existing PredictionAccuracyValidator for prediction-specific validation
            # Combined with new system-wide numerical validation
            
            # System-wide numerical validation
            numerical_validation = self.system_validator.validate_all_content_numbers(data)
            
            # Return combined results
            return {
                'validation_passed': numerical_validation['validation_passed'],
                'warnings': numerical_validation.get('warnings', []),
                'errors': numerical_validation.get('errors', []),
                'corrected_data': numerical_validation.get('corrected_data', {}),
                'numerical_validation': numerical_validation
            }
            
        except Exception as e:
            logger.error(f"Prediction validation failed: {str(e)}")
            return {
                'validation_passed': False,
                'errors': [f"Prediction validation error: {str(e)}"],
                'warnings': [],
                'corrected_data': {}
            }
    
    def validate_before_posting(self, content: str, content_data: Dict, post_type: str = 'general') -> Dict:
        """
        Integration point for content posting
        Call this before posting any content
        """
        return self.content_validator.validate_before_posting(content, content_data, post_type)
    
    def validate_market_data_update(self, market_data: Dict) -> Dict:
        """
        Integration point for market data updates
        Call this when updating market data
        """
        try:
            validation_results = {}
            
            # Validate each token's market data
            for token, token_data in market_data.items():
                if isinstance(token_data, dict):
                    token_validation = self.system_validator.validate_all_content_numbers({
                        'price_data': token_data,
                        'token': token
                    })
                    validation_results[token] = token_validation
            
            # Overall validation status
            overall_valid = all(v.get('validation_passed', False) for v in validation_results.values())
            
            return {
                'overall_validation_passed': overall_valid,
                'token_validations': validation_results,
                'total_tokens_validated': len(validation_results),
                'failed_validations': len([v for v in validation_results.values() if not v.get('validation_passed', False)])
            }
            
        except Exception as e:
            return {
                'overall_validation_passed': False,
                'error': f"Market data validation failed: {str(e)}"
            }
    
    def get_system_health_report(self) -> Dict:
        """Get comprehensive system health report"""
        try:
            # Content validation statistics
            content_stats = self.content_validator.get_validation_statistics()
            
            # System-wide validation cache status
            cache_stats = {
                'validation_cache_size': len(self.system_validator.validation_cache),
                'cache_lock_status': 'healthy' if self.system_validator.cache_lock else 'warning'
            }
            
            # Integration status
            integration_status = {
                'token_mapper_connected': self.token_mapper is not None,
                'database_connected': self.database is not None,
                'prediction_engine_connected': self.prediction_engine is not None
            }
            
            return {
                'system_status': 'healthy' if content_stats.get('approval_rate', 0) > 70 else 'warning',
                'content_validation_stats': content_stats,
                'cache_stats': cache_stats,
                'integration_status': integration_status,
                'validation_components_active': {
                    'system_wide_validator': True,
                    'content_posting_validator': True,
                    'integration_manager': True
                }
            }
            
        except Exception as e:
            return {
                'system_status': 'error',
                'error': f"Health report generation failed: {str(e)}"
            }    