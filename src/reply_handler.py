#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Union, Tuple
import time
import random
import re
from config import config
from collections import Counter
from datetime import datetime, timedelta, timezone
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
    ElementClickInterceptedException,
    ElementNotInteractableException,
    InvalidElementStateException
)
from selenium.webdriver.common.keys import Keys

from utils.logger import logger
from datetime_utils import strip_timezone, ensure_naive_datetimes, safe_datetime_diff

class ReplyHandler:
    """
    Enhanced handler for generating and posting replies to timeline posts
    with improved diversity and audience targeting.
    Includes support for tech-related content and educational replies.
    """
    
    def __init__(self, browser, config, llm_provider, coingecko=None, db=None):
        """
        Initialize the reply handler with multiple selectors for resilience
    
        Args:
            browser: Browser instance for web interaction
            config: Configuration instance containing settings
            llm_provider: LLM provider for generating replies
            coingecko: CoinGecko handler for market data (optional)
            db: Database instance for storing reply data (optional)
        """
        self.browser = browser
        self.config = config
        self.llm_provider = llm_provider
        self.coingecko = coingecko
        self.db = db
    
        # Configure retry settings
        self.max_retries = 20
        self.retry_delay = 5
        self.max_tokens = 300
    
        # Multiple selectors for each element type to increase resilience
        self.reply_button_selectors = [
            '[data-testid="reply"]',
            'div[role="button"] span:has-text("Reply")',
            'button[aria-label*="Reply"]',
            'div[role="button"][aria-label*="Reply"]',
            'div[aria-label*="Reply"]'
        ]
    
        self.reply_textarea_selectors = [
            '[data-testid="tweetTextarea_0"]',
            '[data-testid*="tweetTextarea"]',
            '[contenteditable="true"]',
            'div[role="textbox"]',
            'div.DraftEditor-root'
        ]
    
        self.reply_send_button_selectors = [
            '[data-testid="tweetButton"]',
            'button[data-testid="tweetButton"]',
            'div[role="button"]:has-text("Reply")',
            'div[role="button"]:has-text("Post")',
            'button[type="submit"]'
        ]
    
        # Track posts we've recently replied to (in-memory cache)
        self.recent_replies = []
        self.max_recent_replies = 100
        
        # Initialize diversity tracking
        self.recent_token_mentions = []
        self.recent_tones = []
        
        # Initialize audience interaction tracking
        self.audience_interaction = {
            'handles': {},        # Track interactions by handle
            'categories': {},     # Track interactions by audience category
            'recent_handles': []  # Recently replied-to handles
        }
        
        # Initialize tech content tracking
        self.tech_interaction = {
            'categories': {},     # Track interactions by tech category
            'recent_topics': [],  # Recently discussed tech topics
            'educational_posts': 0  # Count of educational posts
        }
        
        # Track recent reply types to ensure diversity
        self.reply_type_history = {
            'market_analysis': 0,
            'tech_educational': 0,
            'prediction': 0,
            'sentiment': 0,
            'question_answer': 0
        }
        
        # Load configured tech topics if available
        self.tech_topics = []
        if hasattr(self.config, 'get_tech_topics'):
            self.tech_topics = self.config.get_tech_topics()
        
        logger.logger.info("Enhanced reply handler initialized")

    @ensure_naive_datetimes
    def generate_reply(self, post: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generate a witty market-related or tech-related reply using LLM provider with enhanced diversity
        
        Args:
            post: Post data dictionary
            market_data: Market data from CoinGecko (optional)
            
        Returns:
            Generated reply text or None if generation failed
        """
        try:
            logger.debug(f"Starting reply generation for post: '{post.get('text', '')[:50]}...'")
            
            # Extract post details
            post_text = post.get('text', '')
            author_name = post.get('author_name', 'someone')
            author_handle = post.get('author_handle', '@someone')
            post_timestamp = strip_timezone(post.get('timestamp', datetime.now()))
            
            logger.debug(f"Generating reply to {author_handle}'s post: '{post_text[:50]}...'")
            
            # Detect if this is a tech-related post
            is_tech_post = self._is_tech_related_post(post)
            
            # Analyze content to determine reply approach
            content_analysis = post.get('content_analysis', {})
            if not content_analysis and 'text' in post:
                # No pre-existing analysis, create a basic one
                content_analysis = self._basic_content_analysis(post_text)
            
            # Choose between tech or market reply based on post content
            if is_tech_post:
                return self._generate_tech_reply(post, content_analysis)
            else:
                return self._generate_market_reply(post, market_data, content_analysis)
            
        except Exception as e:
            logger.error(f"Reply Generation: {str(e)}")
            
            # Fallback replies if LLM provider fails
            fallback_replies = [
                "Interesting perspective on the market. The data seems to tell a different story though.",
                "Not sure if I agree with this take. Market fundamentals suggest otherwise.",
                "Classic crypto market analysis - half right, half wishful thinking.",
                "Hmm, that's one way to interpret what's happening. The charts paint a nuanced picture though.",
                "I see your point, but have you considered the broader market implications?"
            ]
            return random.choice(fallback_replies)
        
    def _is_tech_related_post(self, post: Dict[str, Any]) -> bool:
        """
        Determine if a post is related to technology rather than just markets
    
        Args:
            post: Post data dictionary
        
        Returns:
            Boolean indicating if post is tech-related
        """
        # Check if post already has tech_related flag from content analyzer
        if post.get('tech_related', False):
            return True
        
        # Check tech analysis if available
        if 'tech_analysis' in post:
            return post['tech_analysis'].get('has_tech_content', False)
    
        # Basic keyword check if no advanced analysis is available
        post_text = post.get('text', '').lower()
    
        # Tech keywords to check for
        tech_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
            'llm', 'large language model', 'gpt', 'claude', 'chatgpt',
            'quantum', 'computing', 'blockchain technology', 'neural network',
            'transformer', 'computer vision', 'nlp', 'generative ai'
        ]
    
        return any(keyword in post_text for keyword in tech_keywords)

    def _basic_content_analysis(self, text: str) -> Dict[str, Any]:
        """
        Advanced content analysis with TokenMappingManager integration and enhanced detection
        
        Args:
            text: Post text content
            
        Returns:
            Comprehensive content analysis dictionary
        """
        
        analysis = {
            'has_question': '?' in text,
            'sentiment': 'neutral',
            'sentiment_strength': 0.0,  # 0-1 scale
            'topics': [],
            'mentioned_tokens': [],
            'token_sentiment': {},  # sentiment per token
            'market_context': None,  # bull/bear/crab market references
            'urgency_level': 0.0,  # 0-1 scale for FOMO/urgency
            'technical_level': 0.0,  # 0-1 scale for technical complexity
            'engagement_signals': [],  # types of engagement expected
            'topic_categories': {
                'tokens': [],
                'technology': [],
                'market_analysis': [],
                'news_events': [],
                'trading': []
            }
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Enhanced sentiment analysis with strength scoring
        sentiment_words = {
            'extremely_bullish': ['moon', 'lambo', 'explosive', 'rocket', 'parabolic', 'breakout'],
            'bullish': ['bullish', 'up', 'good', 'great', 'excited', 'optimistic', 'pump', 'rally', 'green'],
            'extremely_bearish': ['crash', 'dump', 'rekt', 'liquidated', 'bloodbath', 'capitulation'],
            'bearish': ['bearish', 'down', 'bad', 'poor', 'concerned', 'pessimistic', 'sell', 'red']
        }
        
        sentiment_score = 0
        for category, word_list in sentiment_words.items():
            matches = sum(1 for word in word_list if word in text_lower)
            if category.startswith('extremely_bullish'):
                sentiment_score += matches * 2
            elif category == 'bullish':
                sentiment_score += matches
            elif category.startswith('extremely_bearish'):
                sentiment_score -= matches * 2
            elif category == 'bearish':
                sentiment_score -= matches
        
        # Determine sentiment and strength
        if sentiment_score > 2:
            analysis['sentiment'] = 'extremely_bullish'
            analysis['sentiment_strength'] = min(1.0, abs(sentiment_score) / 5)
        elif sentiment_score > 0:
            analysis['sentiment'] = 'bullish'
            analysis['sentiment_strength'] = min(1.0, abs(sentiment_score) / 3)
        elif sentiment_score < -2:
            analysis['sentiment'] = 'extremely_bearish'
            analysis['sentiment_strength'] = min(1.0, abs(sentiment_score) / 5)
        elif sentiment_score < 0:
            analysis['sentiment'] = 'bearish'
            analysis['sentiment_strength'] = min(1.0, abs(sentiment_score) / 3)
        
        # TokenMappingManager integration - detect all possible tokens
        all_symbols = config.token_mapper.get_all_symbols()
        
        # Build comprehensive token detection patterns
        token_patterns = {}
        full_name_mapping = {}
        
        for symbol in all_symbols:
            token_info = config.token_mapper.get_token_info(symbol)
            if token_info:
                # Add symbol patterns
                token_patterns[symbol] = [
                    rf'\b{symbol.lower()}\b',
                    rf'\${symbol.lower()}\b',
                    rf'\b{symbol.upper()}\b',
                    rf'\${symbol.upper()}\b'
                ]
                
                # Add full name patterns
                display_name = token_info.get('display_name', '')
                if display_name and display_name.lower() != symbol.lower():
                    full_name_mapping[display_name.lower()] = symbol
                    token_patterns[symbol].extend([
                        rf'\b{re.escape(display_name.lower())}\b'
                    ])
        
        # Detect mentioned tokens with context
        for symbol, patterns in token_patterns.items():
            token_mentioned = False
            token_context = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    token_mentioned = True
                    # Capture surrounding context (5 words before/after)
                    start = max(0, match.start() - 50)
                    end = min(len(text_lower), match.end() + 50)
                    context = text_lower[start:end]
                    token_context.append(context)
            
            if token_mentioned:
                analysis['mentioned_tokens'].append(symbol)
                analysis['topic_categories']['tokens'].append(symbol)
                
                # Analyze sentiment specifically for this token
                token_sentiment = 0
                for context in token_context:
                    for category, word_list in sentiment_words.items():
                        context_matches = sum(1 for word in word_list if word in context)
                        if category.startswith('bullish'):
                            token_sentiment += context_matches
                        elif category.startswith('bearish'):
                            token_sentiment -= context_matches
                
                if token_sentiment > 0:
                    analysis['token_sentiment'][symbol] = 'bullish'
                elif token_sentiment < 0:
                    analysis['token_sentiment'][symbol] = 'bearish'
                else:
                    analysis['token_sentiment'][symbol] = 'neutral'
        
        # Market context detection
        market_contexts = {
            'bull_market': ['bull market', 'bullish trend', 'uptrend', 'bull run', 'altseason'],
            'bear_market': ['bear market', 'bearish trend', 'downtrend', 'bear run', 'crypto winter'],
            'crab_market': ['sideways', 'consolidation', 'ranging', 'crab market', 'chop']
        }
        
        for context_type, phrases in market_contexts.items():
            if any(phrase in text_lower for phrase in phrases):
                analysis['market_context'] = context_type
                break
        
        # Technology topics with expanded categories
        tech_categories = {
            'ai_ml': ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'llm', 'chatgpt', 'claude'],
            'blockchain_tech': ['blockchain', 'smart contract', 'consensus', 'proof of stake', 'proof of work', 'layer 2', 'rollup', 'sharding'],
            'defi': ['defi', 'decentralized finance', 'yield farming', 'liquidity', 'amm', 'dex', 'lending', 'staking'],
            'nft_metaverse': ['nft', 'non fungible', 'metaverse', 'virtual reality', 'gaming', 'play to earn'],
            'quantum': ['quantum', 'quantum computing', 'quantum supremacy', 'post-quantum cryptography'],
            'web3': ['web3', 'decentralization', 'dao', 'governance', 'tokenization']
        }
        
        for category, keywords in tech_categories.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                analysis['topic_categories']['technology'].extend(found_keywords)
        
        # Trading and market analysis detection
        trading_terms = ['chart', 'technical analysis', 'support', 'resistance', 'breakout', 'fibonacci', 'rsi', 'macd', 'volume', 'market cap']
        found_trading = [term for term in trading_terms if term in text_lower]
        if found_trading:
            analysis['topic_categories']['trading'] = found_trading
            analysis['technical_level'] = min(1.0, len(found_trading) / 5)
        
        # News and events detection
        news_indicators = ['announcement', 'partnership', 'regulation', 'sec', 'etf', 'adoption', 'hack', 'update', 'release', 'launch']
        found_news = [indicator for indicator in news_indicators if indicator in text_lower]
        if found_news:
            analysis['topic_categories']['news_events'] = found_news
        
        # Urgency and FOMO detection
        urgency_words = ['now', 'urgent', 'quickly', 'immediately', 'last chance', 'limited time', 'fomo', 'dont miss', 'act fast']
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        analysis['urgency_level'] = min(1.0, urgency_count / 3)
        
        # Engagement signals
        if analysis['has_question']:
            analysis['engagement_signals'].append('question')
        if '!' in text:
            analysis['engagement_signals'].append('excitement')
        if any(word in text_lower for word in ['thoughts', 'opinion', 'what do you think']):
            analysis['engagement_signals'].append('opinion_seeking')
        if any(word in text_lower for word in ['help', 'advice', 'recommend']):
            analysis['engagement_signals'].append('help_seeking')
        
        # Flatten topics for backward compatibility
        all_topics = []
        for category_topics in analysis['topic_categories'].values():
            all_topics.extend(category_topics)
        analysis['topics'] = list(set(all_topics))  # Remove duplicates
        
        return analysis

    def _generate_market_reply(self, post: Dict[str, Any], market_data: Optional[Dict[str, Any]], 
                            content_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Generate a market-focused reply
        
        Args:
            post: Post data dictionary
            market_data: Market data for context
            content_analysis: Content analysis of the post
            
        Returns:
            Generated reply text or None if generation failed
        """
        # Extract post details
        post_text = post.get('text', '')
        author_name = post.get('author_name', 'someone')
        author_handle = post.get('author_handle', '@someone')
        
        # Prepare market data context with more variety
        market_context = ""
        mentioned_tokens = []
        trending_tokens = []
        
        # Look for crypto/market symbols in the post (expanded list)
        all_symbols = [
            'btc', 'eth', 'sol', 'xrp', 'bnb', 'avax', 'dot', 'uni', 
            'near', 'aave', 'matic', 'fil', 'pol', 'ada', 'doge', 'shib',
            'link', 'ltc', 'etc', 'atom', 'algo', 'vet', 'xtz', 'xlm',
            'icp', 'sand', 'mana', 'grt', 'one', 'hbar', 'ftm', 'rune'
        ]
        
        # Find tokens mentioned in the post
        mentioned_symbols = [symbol for symbol in all_symbols if symbol in post_text.lower()]
        logger.debug(f"Found market symbols in post: {mentioned_symbols}")
        
        # Select tokens for reply with improved variety
        selected_tokens = self._select_diverse_tokens(mentioned_symbols, market_data)
        logger.debug(f"Selected tokens for reply: {selected_tokens}")
        
        # Build market context for diverse token selection
        if market_data:
            market_context = "Current market data:\n"
            
            # Add data for tokens specifically selected for this reply
            for symbol in selected_tokens:
                if symbol.upper() in market_data:
                    coin_data = market_data[symbol.upper()]
                    price = coin_data.get('current_price', 0)
                    change_24h = coin_data.get('price_change_percentage_24h', 0)
                    
                    # Add human-readable name when available
                    coin_name = self._get_token_full_name(symbol)
                    market_context += f"- {coin_name} ({symbol.upper()}): ${price:,.2f} ({change_24h:+.2f}%)\n"
        
        # Detect potential market topics in the post
        market_topics = self._detect_market_topics(post_text)
        logger.debug(f"Detected market topics: {market_topics}")
        
        # Select a random tone for variety
        tone = self._select_reply_tone(post, market_topics)
        logger.debug(f"Selected tone for reply: {tone}")
        
        # Build the enhanced prompt for the LLM with more variety and instructions
        prompt = f"""You are an intelligent, witty crypto/market commentator replying to posts on social media. 
You specialize in providing informative but humorous replies about cryptocurrency and financial markets.

The post you're replying to:
Author: {author_name} ({author_handle})
Post: "{post_text}"

{market_context}

Your task is to write a brief, intelligent, witty reply with a hint of market knowledge or insight. The reply should:
1. Be conversational and casual in tone
2. Include relevant market insights when appropriate
3. Be humorous but not over-the-top or meme-heavy
4. Be concise (1-3 short sentences, maximum 240 characters)
5. Not use hashtags, emojis, or excessive special characters
6. Sound like a real person, not an automated bot
7. Not appear overly promotional or financial-advice-like
8. Vary your response style to avoid sounding repetitive

Reply tone: {tone}

{f"Consider mentioning these topics: {', '.join(market_topics[:2])}" if market_topics else ""}
{f"Consider referencing these tokens: {', '.join(selected_tokens[:2])}" if selected_tokens else ""}

Your reply (maximum 240 characters):
"""

        # Generate reply using LLM provider
        logger.debug(f"Sending prompt to LLM provider for reply generation")
        reply_text = self.llm_provider.generate_text(prompt, max_tokens=self.max_tokens)
        
        if not reply_text:
            logger.warning("LLM provider returned empty response")
            return None
        
        # Make sure the reply isn't too long for Twitter (240 chars max)
        if len(reply_text) > 240:
            # Truncate with preference to complete sentences
            last_period = reply_text[:240].rfind('.')
            last_question = reply_text[:240].rfind('?')
            last_exclamation = reply_text[:240].rfind('!')
            last_punctuation = max(last_period, last_question, last_exclamation)
            
            if last_punctuation > 180:  # If we can get a substantial reply with complete sentence
                reply_text = reply_text[:last_punctuation+1]
            else:
                # Find last space to avoid cutting words
                last_space = reply_text[:240].rfind(' ')
                if last_space > 180:
                    reply_text = reply_text[:last_space]
                else:
                    # Hard truncate as last resort
                    reply_text = reply_text[:237] + "..."
        
        logger.logger.info(f"Generated market reply ({len(reply_text)} chars): {reply_text}")
        
        # Track this reply type in history
        self.reply_type_history['market_analysis'] += 1
        
        return reply_text

    def _generate_tech_reply(self, post: Dict[str, Any], content_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Generate a technology-focused educational reply
        
        Args:
            post: Post data dictionary
            content_analysis: Content analysis of the post
            
        Returns:
            Generated reply text or None if generation failed
        """
        # Extract post details
        post_text = post.get('text', '')
        author_name = post.get('author_name', 'someone')
        author_handle = post.get('author_handle', '@someone')
        
        # Check if we have tech analysis data
        tech_analysis = post.get('tech_analysis', {})
        
        # If no detailed tech analysis, create basic analysis
        if not tech_analysis:
            tech_analysis = self._analyze_tech_content(post_text)
        
        # Determine tech category
        tech_category = "ai"  # Default to AI if no specific category detected
        
        # Extract category from analysis if available
        if 'categories' in tech_analysis:
            for category, data in tech_analysis['categories'].items():
                if data.get('detected', False):
                    tech_category = category
                    break
        
        # Determine educational approach
        is_educational = tech_analysis.get('educational', {}).get('is_educational', False)
        educational_type = tech_analysis.get('educational', {}).get('educational_type', 'informational')
        audience_level = tech_analysis.get('educational', {}).get('audience_level', 'general')
        
        # Check if post is asking a question
        has_question = content_analysis.get('has_question', '?' in post_text)
        
        # Determine reply tone
        sentiment = tech_analysis.get('tech_sentiment', {}).get('label', 'neutral')
        
        if sentiment in ['positive', 'excited']:
            tone = 'enthusiastic'
        elif sentiment in ['curious', 'interested']:
            tone = 'informative'
        elif sentiment in ['concerned', 'negative']:
            tone = 'balanced'
        else:
            tone = 'educational'
        
        # Extract key technical terms for the prompt
        key_terms = []
        
        # Try to get terms from different sources in the analysis
        if 'categories' in tech_analysis:
            for category, data in tech_analysis['categories'].items():
                if data.get('detected', False) and 'terms' in data:
                    key_terms.extend(data['terms'][:3])  # Take up to 3 terms
        
        # Add key points if available
        if 'key_points' in tech_analysis:
            key_points = tech_analysis['key_points'][:2]  # Take up to 2 key points
            key_terms.extend([f"key point: {point}" for point in key_points])
        
        # Check for crypto integration
        crypto_integration = tech_analysis.get('crypto_integration', {}).get('crypto_tech_integration', False)
        
        # Build prompt based on post type and content
        if has_question:
            prompt_type = "answer tech question"
        elif is_educational:
            prompt_type = "build on educational content"
        elif crypto_integration:
            prompt_type = "discuss crypto-tech integration"
        else:
            prompt_type = "share tech insight"
        
        # Build the prompt for the LLM
        prompt = f"""You're an intelligent technology commentator specializing in AI, blockchain technology, quantum computing, and cutting-edge tech. You're replying to a social media post.

The post you're replying to:
Author: {author_name} ({author_handle})
Post: "{post_text}"

Your task is to write a brief, educational reply about {tech_category}. Your reply should:
1. Be informative and add value to the conversation
2. {f"Address the question about {tech_category}" if has_question else f"Share an insight about {tech_category}"}
3. Target a {audience_level} level audience
4. Be concise and focused (maximum 240 characters)
5. {f"Discuss the intersection of crypto and {tech_category}" if crypto_integration else f"Focus primarily on {tech_category}"}
6. Sound like a knowledgeable but approachable tech enthusiast
7. End with a subtle engagement hook to continue the conversation

Reply tone: {tone}
Reply approach: {prompt_type}

{f"Include these key technical points: {', '.join(key_terms[:3])}" if key_terms else ""}

Your reply (maximum 240 characters):
"""

        # Generate reply using LLM provider
        logger.debug(f"Sending prompt to LLM provider for tech reply generation")
        reply_text = self.llm_provider.generate_text(prompt, max_tokens=self.max_tokens)
        
        if not reply_text:
            logger.warning("LLM provider returned empty response for tech reply")
            return None
        
        # Make sure the reply isn't too long for Twitter (240 chars max)
        if len(reply_text) > 240:
            # Truncate with preference to complete sentences
            last_period = reply_text[:240].rfind('.')
            last_question = reply_text[:240].rfind('?')
            last_exclamation = reply_text[:240].rfind('!')
            last_punctuation = max(last_period, last_question, last_exclamation)
            
            if last_punctuation > 180:  # If we can get a substantial reply with complete sentence
                reply_text = reply_text[:last_punctuation+1]
            else:
                # Find last space to avoid cutting words
                last_space = reply_text[:240].rfind(' ')
                if last_space > 180:
                    reply_text = reply_text[:last_space]
                else:
                    # Hard truncate as last resort
                    reply_text = reply_text[:237] + "..."
        
        logger.logger.info(f"Generated tech reply ({len(reply_text)} chars): {reply_text}")
        
        # Track this reply type in history
        self.reply_type_history['tech_educational'] += 1
        
        # Update tech interaction tracking
        self._update_tech_interaction_history(tech_category, is_educational)
        
        return reply_text

    def _select_diverse_tokens(self, mentioned_symbols: List[str], market_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Advanced token selection with TokenMappingManager integration and intelligent categorization
        
        Args:
            mentioned_symbols: Symbols mentioned in the post
            market_data: Available market data
            
        Returns:
            List of strategically selected token symbols for diverse, engaging replies
        """
        selected_tokens = []
        
        # Build dynamic token categories using TokenMappingManager
        all_symbols = config.token_mapper.get_all_symbols()
        
        # Intelligent categorization based on market cap and token characteristics
        token_categories = {
            "blue_chip": [],
            "layer1": [],
            "defi": [],
            "meme": [],
            "exchange": [],
            "emerging": [],
            "stablecoin": []
        }
        
        # Categorize tokens dynamically using TokenMappingManager data
        for symbol in all_symbols:
            token_info = config.token_mapper.get_token_info(symbol)
            if not token_info:
                continue
                
            display_name = token_info.get('display_name', '').lower()
            symbol_lower = symbol.lower()
            
            # Blue chip classification
            if symbol in ['BTC', 'ETH']:
                token_categories["blue_chip"].append(symbol_lower)
            # Layer 1 blockchain detection
            elif any(term in display_name for term in ['protocol', 'network', 'chain']) or symbol in ['SOL', 'AVAX', 'ADA', 'DOT', 'NEAR', 'ATOM', 'FTM', 'ALGO']:
                token_categories["layer1"].append(symbol_lower)
            # DeFi protocol detection
            elif any(term in display_name for term in ['defi', 'swap', 'exchange', 'lending', 'yield']) or symbol in ['UNI', 'AAVE', 'LINK', 'MATIC', 'GRT', 'SNX', 'CRV', 'COMP', 'MKR']:
                token_categories["defi"].append(symbol_lower)
            # Meme token detection
            elif any(term in display_name for term in ['meme', 'dog', 'shiba']) or symbol in ['DOGE', 'SHIB', 'PEPE']:
                token_categories["meme"].append(symbol_lower)
            # Exchange token detection
            elif any(term in display_name for term in ['exchange', 'binance', 'kucoin']) or symbol in ['BNB', 'KCS', 'CRO']:
                token_categories["exchange"].append(symbol_lower)
            # Stablecoin detection
            elif any(term in display_name for term in ['usd', 'stable', 'tether', 'coin']) or symbol in ['USDT', 'USDC', 'DAI', 'BUSD']:
                token_categories["stablecoin"].append(symbol_lower)
            # Everything else goes to emerging
            else:
                token_categories["emerging"].append(symbol_lower)
        
        # Get frequently mentioned tokens to avoid repetition
        frequently_mentioned = self._get_frequently_mentioned_tokens()
        
        # Priority 1: Include mentioned tokens (but diversify)
        if mentioned_symbols:
            for symbol in mentioned_symbols[:2]:  # Max 2 mentioned tokens
                if symbol.lower() not in frequently_mentioned:
                    selected_tokens.append(symbol.lower())
        
        # Priority 2: Add tokens with significant movement from different categories
        if market_data:
            moving_tokens = self._find_tokens_with_movement(market_data)
            
            # Randomize category selection for diversity
            available_categories = [cat for cat in token_categories.keys() if token_categories[cat]]
            random.shuffle(available_categories)
            
            for category in available_categories:
                if len(selected_tokens) >= 3:  # Max 3 total tokens
                    break
                    
                category_tokens = token_categories[category]
                # Find moving tokens in this category
                category_moving = [t for t in moving_tokens if t.lower() in category_tokens and t.lower() not in frequently_mentioned]
                
                if category_moving:
                    # Pick the token with highest movement in this category
                    best_token = None
                    highest_movement = 0
                    
                    for token in category_moving:
                        token_data = market_data.get(token.upper(), {})
                        if isinstance(token_data, dict):
                            movement = abs(token_data.get('price_change_percentage_24h', 0))
                            if movement > highest_movement and token.lower() not in [t.lower() for t in selected_tokens]:
                                highest_movement = movement
                                best_token = token.lower()
                    
                    if best_token:
                        selected_tokens.append(best_token)
                elif category_tokens:
                    # No moving tokens in category, pick a random available one
                    available = [t for t in category_tokens if t not in frequently_mentioned and t not in [s.lower() for s in selected_tokens]]
                    if available:
                        selected_tokens.append(random.choice(available))
        
        # Priority 3: Ensure minimum diversity if we don't have enough tokens
        if len(selected_tokens) < 2:
            # Fill from different categories
            for category_name, category_tokens in token_categories.items():
                if len(selected_tokens) >= 2:
                    break
                available = [t for t in category_tokens if t not in frequently_mentioned and t not in [s.lower() for s in selected_tokens]]
                if available:
                    selected_tokens.append(random.choice(available))
        
        # Update token mention history
        self._update_token_mention_history(selected_tokens)
        
        return selected_tokens

    def _update_token_mention_history(self, tokens: List[str]) -> None:
        """
        Update history of recently mentioned tokens to ensure variety
        
        Args:
            tokens: List of tokens mentioned in the current reply
        """
        # Add currently mentioned tokens to history
        for token in tokens:
            self.recent_token_mentions.append(token)
        
        # Keep only the most recent mentions (last 20)
        if len(self.recent_token_mentions) > 20:
            self.recent_token_mentions = self.recent_token_mentions[-20:]

    def _get_frequently_mentioned_tokens(self) -> List[str]:
        """
        Get list of tokens that have been frequently mentioned in recent replies
        
        Returns:
            List of frequently mentioned token symbols
        """
        if not self.recent_token_mentions:
            return []
        
        # Count occurrences of each token
        token_counts = {}
        for token in self.recent_token_mentions:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Consider tokens mentioned more than twice in recent history as "frequent"
        frequent_tokens = [token for token, count in token_counts.items() if count > 2]
        
        return frequent_tokens

    def _find_tokens_with_movement(self, market_data: Dict[str, Any]) -> List[str]:
        """
        Find tokens with significant price movement in the last 24h
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            List of token symbols with significant price movement
        """
        moving_tokens = []
        
        for symbol, data in market_data.items():
            if 'price_change_percentage_24h' in data:
                change = data['price_change_percentage_24h']
                # Consider significant movement as > 5% in either direction
                if abs(change) > 5.0:
                    moving_tokens.append(symbol)
        
        # Shuffle to add randomness
        random.shuffle(moving_tokens)
        return moving_tokens

    def _get_token_full_name(self, symbol: str) -> str:
        """
        Get full name for a token symbol
        
        Args:
            symbol: Token symbol (e.g., 'btc')
            
        Returns:
            Full name of the token
        """
        # Map of common symbols to full names
        token_names = {
            'btc': 'Bitcoin',
            'eth': 'Ethereum',
            'sol': 'Solana',
            'xrp': 'Ripple',
            'bnb': 'Binance Coin',
            'avax': 'Avalanche',
            'dot': 'Polkadot',
            'uni': 'Uniswap',
            'near': 'NEAR Protocol',
            'aave': 'Aave',
            'matic': 'Polygon',
            'fil': 'Filecoin',
            'ada': 'Cardano',
            'doge': 'Dogecoin',
            'shib': 'Shiba Inu',
            'link': 'Chainlink',
            'ltc': 'Litecoin',
            'etc': 'Ethereum Classic',
            'atom': 'Cosmos',
            'algo': 'Algorand',
            'vet': 'VeChain',
            'xtz': 'Tezos',
            'xlm': 'Stellar',
            'icp': 'Internet Computer',
            'sand': 'The Sandbox',
            'mana': 'Decentraland',
            'grt': 'The Graph',
            'one': 'Harmony',
            'hbar': 'Hedera',
            'ftm': 'Fantom',
            'rune': 'THORChain',
            'pol': 'Polkadot'
        }
        
        return token_names.get(symbol.lower(), symbol.upper())

    def _select_reply_tone(self, post: Dict[str, Any], market_topics: List[str]) -> str:
        """
        Select an appropriate tone for the reply to add variety
        
        Args:
            post: Post data dictionary
            market_topics: Detected market topics in the post
            
        Returns:
            Selected tone as a string
        """
        # Available tones for variety
        tones = [
            "humorous",       # Funny, light-hearted
            "analytical",     # Data-driven, factual
            "enthusiastic",   # Excited, positive
            "skeptical",      # Questioning, cautious
            "educational",    # Informative, helpful
            "playful",        # Teasing, meme-like
            "contrarian",     # Taking opposite view
            "curious",        # Asking questions
            "impressed",      # Showing appreciation
            "neutral"         # Balanced, moderate
        ]
        
        # Avoid recently used tones
        available_tones = [tone for tone in tones if tone not in self.recent_tones[-3:]]
        
        # If no available tones, reset
        if not available_tones:
            available_tones = tones
        
        # Choose tone with some contextual awareness
        if 'bearish sentiment' in market_topics:
            contextual_tones = [t for t in available_tones if t in ["skeptical", "contrarian", "analytical"]]
            if contextual_tones:
                selected_tone = random.choice(contextual_tones)
            else:
                selected_tone = random.choice(available_tones)
        elif 'bullish sentiment' in market_topics:
            contextual_tones = [t for t in available_tones if t in ["enthusiastic", "impressed", "playful"]]
            if contextual_tones:
                selected_tone = random.choice(contextual_tones)
            else:
                selected_tone = random.choice(available_tones)
        elif 'technical analysis' in market_topics:
            contextual_tones = [t for t in available_tones if t in ["analytical", "educational", "skeptical"]]
            if contextual_tones:
                selected_tone = random.choice(contextual_tones)
            else:
                selected_tone = random.choice(available_tones)
        elif 'ai' in market_topics or 'artificial intelligence' in market_topics:
            contextual_tones = [t for t in available_tones if t in ["educational", "curious", "analytical"]]
            if contextual_tones:
                selected_tone = random.choice(contextual_tones)
            else:
                selected_tone = random.choice(available_tones)
        elif 'quantum' in market_topics:
            contextual_tones = [t for t in available_tones if t in ["educational", "impressed", "analytical"]]
            if contextual_tones:
                selected_tone = random.choice(contextual_tones)
            else:
                selected_tone = random.choice(available_tones)
        else:
            # Random selection for other cases
            selected_tone = random.choice(available_tones)
        
        # Update recent tones
        self.recent_tones.append(selected_tone)
        if len(self.recent_tones) > 10:
            self.recent_tones = self.recent_tones[-10:]
        
        return selected_tone

    def _detect_market_topics(self, text: str) -> List[str]:
        """
        Detect market-related topics in the post text with expanded recognition
        
        Args:
            text: Post text
            
        Returns:
            List of detected market topics
        """
        topics = []
        
        # Enhanced topic patterns with regex
        patterns = [
            (r'\b(bull|bullis(h|m))\b', 'bullish sentiment'),
            (r'\b(bear|bearis(h|m))\b', 'bearish sentiment'),
            (r'\b(crash|dump|collapse)\b', 'market downturn'),
            (r'\b(pump|moon|rally|surge)\b', 'price rally'),
            (r'\b(fed|federal reserve|interest rate|inflation)\b', 'macroeconomic factors'),
            (r'\b(hold|hodl|holding)\b', 'investment strategy'),
            (r'\b(buy|buying|bought)\b', 'buying activity'),
            (r'\b(sell|selling|sold)\b', 'selling pressure'),
            (r'\baltcoin season|alt season\b', 'altcoin performance'),
            (r'\btechnical analysis|TA|support|resistance\b', 'technical analysis'),
            (r'\b(volume|liquidity)\b', 'market liquidity'),
            (r'\b(fund|investor|institutional)\b', 'institutional investment'),
            (r'\bregulat(ion|ory|e)\b', 'regulatory discussion'),
            (r'\b(trade|trading)\b', 'trading activity'),
            # New expanded patterns
            (r'\b(layer\s*1|L1)\b', 'layer 1 blockchain'),
            (r'\b(layer\s*2|L2)\b', 'layer 2 scaling'),
            (r'\b(defi|decentralized finance)\b', 'decentralized finance'),
            (r'\b(nft|non.fungible)\b', 'NFT market'),
            (r'\b(memecoin|meme coin)\b', 'meme coins'),
            (r'\b(airdrop|drop)\b', 'token airdrops'),
            (r'\b(staking|stake)\b', 'staking rewards'),
            (r'\b(yield|farming|pool)\b', 'yield farming'),
            (r'\b(dao|governance)\b', 'governance'),
            (r'\b(bridge|interoperability)\b', 'cross-chain'),
            (r'\b(adoption|mainstream|institutional)\b', 'mainstream adoption'),
            (r'\b(wallet|cold storage|hardware)\b', 'crypto security'),
            (r'\b(private key|seed phrase)\b', 'wallet security'),
            (r'\b(halving|halvening)\b', 'bitcoin halving'),
            (r'\b(metaverse|virtual world)\b', 'metaverse'),
            (r'\b(web3|web 3)\b', 'web3'),
            (r'\b(gas|gwei|fee)\b', 'network fees'),
            (r'\b(block|chain|hash)\b', 'blockchain technology'),
            (r'\b(smart contract|protocol)\b', 'smart contracts'),
            (r'\b(ico|ido|ito)\b', 'token offerings'),
            (r'\b(sec|cftc|regulation)\b', 'regulatory concerns'),
            # Add tech-related patterns
            (r'\b(ai|artificial intelligence|machine learning)\b', 'ai'),
            (r'\b(llm|large language model|gpt|claude|chatgpt)\b', 'large language models'),
            (r'\bquantum\s*(comput|algorithm|bit|supremacy)\b', 'quantum computing'),
            (r'\b(neural network|deep learning|transformer)\b', 'neural networks'),
            (r'\b(generative ai|diffusion model|stable diffusion|midjourney)\b', 'generative ai')
        ]
        
        text_lower = text.lower()
        
        for pattern, topic in patterns:
            if re.search(pattern, text_lower):
                topics.append(topic)
                
        return topics

    def _analyze_tech_content(self, text: str) -> Dict[str, Any]:
        """
        Perform basic tech content analysis
        
        Args:
            text: Post text
            
        Returns:
            Tech analysis result dictionary
        """
        text_lower = text.lower()
        analysis = {
            'has_tech_content': False,
            'categories': {
                'ai': {'detected': False, 'terms': [], 'confidence': 0.0},
                'quantum': {'detected': False, 'terms': [], 'confidence': 0.0},
                'blockchain_tech': {'detected': False, 'terms': [], 'confidence': 0.0},
                'advanced_computing': {'detected': False, 'terms': [], 'confidence': 0.0}
            },
            'educational': {
                'is_educational': False,
                'educational_type': 'informational',
                'audience_level': 'general'
            },
            'tech_sentiment': {
                'label': 'neutral',
                'positivity': 0.5
            },
            'crypto_integration': {
                'crypto_tech_integration': False,
                'integration_topics': []
            }
        }
        
        # AI terms detection
        ai_terms = [
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'llm', 'large language model', 'gpt', 'claude',
            'chatgpt', 'transformer', 'generative ai', 'diffusion model'
        ]
        
        ai_matches = [term for term in ai_terms if term in text_lower]
        if ai_matches:
            analysis['categories']['ai']['detected'] = True
            analysis['categories']['ai']['terms'] = ai_matches
            analysis['categories']['ai']['confidence'] = min(1.0, len(ai_matches) * 0.2)
            analysis['has_tech_content'] = True
        
        # Quantum computing terms
        quantum_terms = [
            'quantum', 'qubit', 'quantum computing', 'quantum computer',
            'quantum supremacy', 'quantum algorithm', 'quantum cryptography'
        ]
        
        quantum_matches = [term for term in quantum_terms if term in text_lower]
        if quantum_matches:
            analysis['categories']['quantum']['detected'] = True
            analysis['categories']['quantum']['terms'] = quantum_matches
            analysis['categories']['quantum']['confidence'] = min(1.0, len(quantum_matches) * 0.25)
            analysis['has_tech_content'] = True
        
        # Blockchain technology terms
        blockchain_terms = [
            'blockchain technology', 'consensus', 'zero-knowledge', 'zk-rollup',
            'layer 2', 'smart contract', 'decentralized identity', 'web3'
        ]
        
        blockchain_matches = [term for term in blockchain_terms if term in text_lower]
        if blockchain_matches:
            analysis['categories']['blockchain_tech']['detected'] = True
            analysis['categories']['blockchain_tech']['terms'] = blockchain_matches
            analysis['categories']['blockchain_tech']['confidence'] = min(1.0, len(blockchain_matches) * 0.2)
            analysis['has_tech_content'] = True
        
        # Advanced computing terms
        computing_terms = [
            'edge computing', 'high performance computing', 'neuromorphic',
            'quantum dot', 'optical computing', 'dna computing', 'exascale'
        ]
        
        computing_matches = [term for term in computing_terms if term in text_lower]
        if computing_matches:
            analysis['categories']['advanced_computing']['detected'] = True
            analysis['categories']['advanced_computing']['terms'] = computing_matches
            analysis['categories']['advanced_computing']['confidence'] = min(1.0, len(computing_matches) * 0.25)
            analysis['has_tech_content'] = True
        
        # Check for crypto terms
        crypto_terms = [
            'crypto', 'bitcoin', 'ethereum', 'blockchain', 'defi', 'nft',
            'token', 'wallet', 'mining', 'staking', 'coin'
        ]
        
        crypto_matches = [term for term in crypto_terms if term in text_lower]
        
        # Check for crypto integration
        if crypto_matches and analysis['has_tech_content']:
            analysis['crypto_integration']['crypto_tech_integration'] = True
            
            # Simple integration topics detection
            if any(term in text_lower for term in ai_terms) and any(term in text_lower for term in crypto_terms):
                analysis['crypto_integration']['integration_topics'].append('ai_crypto')
                
            if any(term in text_lower for term in quantum_terms) and any(term in text_lower for term in crypto_terms):
                analysis['crypto_integration']['integration_topics'].append('quantum_crypto')
        
        # Check for educational content
        educational_indicators = [
            'explained', 'explanation', 'introduction to', 'understanding',
            'learn about', 'basics of', 'how does', 'what is', 'guide to'
        ]
        
        if any(indicator in text_lower for indicator in educational_indicators):
            analysis['educational']['is_educational'] = True
            
            # Determine educational type
            if any(term in text_lower for term in ['how', 'how to', 'how does']):
                analysis['educational']['educational_type'] = 'explanatory'
            elif any(term in text_lower for term in ['introduction', 'basics', 'beginner']):
                analysis['educational']['educational_type'] = 'introductory'
            elif any(term in text_lower for term in ['comparison', 'versus', 'vs']):
                analysis['educational']['educational_type'] = 'comparative'
            
            # Determine audience level based on technical language density
            technical_terms_count = sum(1 for term in text_lower.split() if len(term) > 8)  # Simple heuristic
            if technical_terms_count > 5:
                analysis['educational']['audience_level'] = 'technical'
            elif technical_terms_count > 2:
                analysis['educational']['audience_level'] = 'intermediate'
        
        # Basic sentiment analysis
        positive_terms = [
            'exciting', 'amazing', 'breakthrough', 'revolutionary', 'impressive',
            'innovative', 'powerful', 'advanced', 'cutting-edge', 'promising'
        ]
        
        negative_terms = [
            'concerning', 'worrying', 'overhyped', 'problematic', 'dangerous',
            'risky', 'disappointing', 'limited', 'flawed', 'unreliable'
        ]
        
        positive_count = sum(1 for term in positive_terms if term in text_lower)
        negative_count = sum(1 for term in negative_terms if term in text_lower)
        
        # Calculate sentiment
        if positive_count > negative_count:
            analysis['tech_sentiment']['label'] = 'positive'
            analysis['tech_sentiment']['positivity'] = min(1.0, 0.5 + (positive_count * 0.1))
        elif negative_count > positive_count:
            analysis['tech_sentiment']['label'] = 'negative'
            analysis['tech_sentiment']['positivity'] = max(0.0, 0.5 - (negative_count * 0.1))
        
        # Check for questions (curiosity)
        if '?' in text:
            analysis['tech_sentiment']['label'] = 'curious'
        
        return analysis

    @ensure_naive_datetimes
    def _update_tech_interaction_history(self, tech_category: str, is_educational: bool) -> None:
        """
        Update tech interaction history
        
        Args:
            tech_category: The technology category discussed
            is_educational: Whether the post was educational
        """
        # Update category counts
        if tech_category in self.tech_interaction['categories']:
            self.tech_interaction['categories'][tech_category] += 1
        else:
            self.tech_interaction['categories'][tech_category] = 1
            
        # Update recent topics list
        self.tech_interaction['recent_topics'].append(tech_category)
        if len(self.tech_interaction['recent_topics']) > 20:
            self.tech_interaction['recent_topics'] = self.tech_interaction['recent_topics'][-20:]
            
        # Track educational posts
        if is_educational:
            self.tech_interaction['educational_posts'] += 1
            
        # Store in database if available
        if self.db and hasattr(self.db, 'store_tech_interaction'):
            try:
                timestamp = strip_timezone(datetime.now())
                self.db.store_tech_interaction(
                    category=tech_category,
                    is_educational=is_educational,
                    timestamp=timestamp
                )
            except Exception as e:
                logger.warning(f"Failed to store tech interaction: {str(e)}")

    @ensure_naive_datetimes
    def _diversify_audience_targeting(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Diversify audience targeting to reach broader user segments
        
        Args:
            posts: List of post data dictionaries
            
        Returns:
            Reordered list of posts for diverse targeting
        """
        # Group posts by account type/category
        audience_segments = {
            'influencers': [],      # High follower accounts
            'retail_users': [],     # Regular users
            'projects': [],         # Project/token accounts
            'news': [],             # News/media accounts
            'traders': [],          # Trading accounts
            'developers': [],       # Developer focused accounts
            'tech_educators': [],   # Technology educators
            'other': []             # Uncategorized accounts
        }
        
        # Categorize each post by audience segment
        for post in posts:
            handle = post.get('author_handle', '')
            name = post.get('author_name', '')
            
            # Skip already categorized posts
            if 'audience_category' in post:
                segment = post['audience_category']
                if segment in audience_segments:
                    audience_segments[segment].append(post)
                else:
                    audience_segments['other'].append(post)
                continue
                
            # Categorize based on available signals
            if post.get('author_follower_count', 0) > 10000:
                post['audience_category'] = 'influencers'
                audience_segments['influencers'].append(post)
            elif any(keyword in handle.lower() or keyword in name.lower() for keyword in ['news', 'alert', 'update', 'daily']):
                post['audience_category'] = 'news'
                audience_segments['news'].append(post)
            elif any(keyword in handle.lower() or keyword in name.lower() for keyword in ['trade', 'chart', 'crypto', 'investor']):
                post['audience_category'] = 'traders'
                audience_segments['traders'].append(post)
            elif any(keyword in handle.lower() or keyword in name.lower() for keyword in ['dev', 'code', 'tech', 'build']):
                post['audience_category'] = 'developers'
                audience_segments['developers'].append(post)
            # Check for tech educators
            elif any(keyword in handle.lower() or keyword in name.lower() for keyword in ['teach', 'learn', 'edu', 'professor', 'phd', 'science']):
                post['audience_category'] = 'tech_educators'
                audience_segments['tech_educators'].append(post)
            # Check for project/token names in handle
            elif any(token in handle.lower() or token in name.lower() for token in ['eth', 'btc', 'sol', 'chain', 'protocol', 'token', 'coin']):
                post['audience_category'] = 'projects'
                audience_segments['projects'].append(post)
            else:
                post['audience_category'] = 'retail_users'
                audience_segments['retail_users'].append(post)
        
        # Calculate interaction ratios to identify underserved segments
        total_interactions = sum(len(posts) for posts in audience_segments.values())
        segment_ratios = {}
        
        for segment, segment_posts in audience_segments.items():
            if segment_posts:
                # Calculate current ratio of this segment in our pool
                current_ratio = len(segment_posts) / max(1, total_interactions)
                
                # Calculate historical interaction ratio with this segment
                historical_count = self.audience_interaction.get('categories', {}).get(segment, 0)
                historical_total = sum(self.audience_interaction.get('categories', {}).values()) or 1
                historical_ratio = historical_count / historical_total if historical_total else 0
                
                # Prioritize underserved segments (lower historical ratio)
                if historical_ratio > 0:
                    segment_ratios[segment] = current_ratio / historical_ratio
                else:
                    # Boost segments we haven't interacted with yet
                    segment_ratios[segment] = 2.0
            else:
                segment_ratios[segment] = 0
        
        # Create diversified ordering using the diversity ratios
        diversified_posts = []
        
        # First, include posts from underrepresented segments
        underrepresented = sorted(segment_ratios.items(), key=lambda x: x[1], reverse=True)
        
        for segment, ratio in underrepresented:
            if ratio > 1.0:  # This segment is underrepresented
                # Take the highest priority post from this segment that we haven't replied to recently
                for post in audience_segments[segment]:
                    handle = post.get('author_handle')
                    if handle not in self.audience_interaction['recent_handles'][-10:]:
                        diversified_posts.append(post)
                        # Only take one from each underrepresented segment for now
                        break
        
        # Then add remaining posts sorted by original priority
        remaining_posts = [p for p in posts if p not in diversified_posts]
        
        # Filter out recently interacted handles
        filtered_remaining = [p for p in remaining_posts 
                            if p.get('author_handle') not in self.audience_interaction['recent_handles'][-5:]]
        
        # If we filtered all remaining posts, fall back to original list
        if not filtered_remaining and remaining_posts:
            filtered_remaining = remaining_posts
        
        # Sort by original priority score
        filtered_remaining.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        # Combine lists with diversified posts first
        result = diversified_posts + filtered_remaining
        
        return result

    @ensure_naive_datetimes
    def _update_audience_interaction_history(self, post: Dict[str, Any], reply_text: str, metadata: Dict[str, Any]) -> None:
        """
        Update history of audience interactions to improve targeting
    
        Args:
            post: Post that was replied to
            reply_text: Generated reply text
            metadata: Reply metadata
        """
        handle = post.get('author_handle')
        category = post.get('audience_category', 'other')
    
        # Update handle-specific tracking
        if handle:
            if handle in self.audience_interaction['handles']:
                self.audience_interaction['handles'][handle] += 1
            else:
                self.audience_interaction['handles'][handle] = 1
        
            # Update recent handles list
            self.audience_interaction['recent_handles'].append(handle)
            if len(self.audience_interaction['recent_handles']) > 30:
                self.audience_interaction['recent_handles'] = self.audience_interaction['recent_handles'][-30:]
    
        # Update category tracking
        if category:
            if category in self.audience_interaction['categories']:
                self.audience_interaction['categories'][category] += 1
            else:
                self.audience_interaction['categories'][category] = 1
    
        # Store in database if available
        if self.db and hasattr(self.db, 'store_audience_interaction'):
            try:
                timestamp = strip_timezone(datetime.now())
                self.db.store_audience_interaction(
                    handle=handle,
                    category=category,
                    reply_text=reply_text,
                    sentiment=metadata.get('sentiment', 'neutral'),
                    tokens=metadata.get('tokens', []),
                    timestamp=timestamp
                )
            except Exception as e:
                logger.warning(f"Failed to store audience interaction: {str(e)}")

    @ensure_naive_datetimes
    def _calculate_variable_delay(self, post: Dict[str, Any]) -> float:
        """
        Calculate variable delay between replies based on post engagement
    
        Args:
            post: Post data dictionary
        
        Returns:
            Delay in seconds
        """
        # Base delay
        base_delay = 5.0
    
        # Add randomness
        randomness = random.uniform(0.8, 1.2)
    
        # Adjust based on post priority - higher priority gets less delay
        priority_score = post.get('priority_score', 50)
        # Normalize score to 0-1 range
        normalized_score = min(1.0, max(0.0, priority_score / 100))
        # Higher score = less delay
        priority_factor = 1.5 - normalized_score
    
        # Calculate final delay
        delay = base_delay * randomness * priority_factor
    
        # Ensure minimum reasonable delay
        return max(3.0, min(15.0, delay))

    @ensure_naive_datetimes
    def _capture_reply_metadata(self, post: Dict[str, Any], reply_text: str) -> Dict[str, Any]:
        """
        Capture enhanced metadata about a reply for analysis and tracking
    
        Args:
            post: Original post data
            reply_text: Generated reply text
        
        Returns:
            Dictionary with metadata about the reply
        """
        # Extract mentioned tokens
        tokens = self._extract_mentioned_tokens(reply_text)
    
        # Determine reply sentiment
        sentiment = self._get_reply_sentiment(reply_text)
    
        # Count relevant metrics
        char_count = len(reply_text)
        word_count = len(reply_text.split())
    
        # Track replied topics
        topics = []
        if 'analysis' in post and 'topics' in post['analysis']:
            topics = list(post['analysis']['topics'].keys())
    
        # Analyze reply style for future reference
        reply_style = self._analyze_reply_style(reply_text)
    
        # Add audience targeting data
        audience_data = {
            'author_handle': post.get('author_handle', ''),
            'author_category': post.get('audience_category', 'other'),
            'follower_count': post.get('author_follower_count', 0)
        }
        
        # Check if this was a tech-related reply
        is_tech_reply = False
        tech_category = None
        
        if any(tech_term in reply_text.lower() for tech_term in ['ai', 'artificial intelligence', 'quantum', 'neural']):
            is_tech_reply = True
            
            # Try to determine tech category
            if 'ai' in reply_text.lower() or 'artificial intelligence' in reply_text.lower():
                tech_category = 'ai'
            elif 'quantum' in reply_text.lower():
                tech_category = 'quantum'
            elif 'blockchain' in reply_text.lower():
                tech_category = 'blockchain_tech'
            
        # Add tech data if relevant
        tech_data = None
        if is_tech_reply:
            tech_data = {
                'is_tech_reply': True,
                'tech_category': tech_category,
                'is_educational': 'learn' in reply_text.lower() or 'understand' in reply_text.lower()
            }
    
        return {
            'tokens': tokens,
            'sentiment': sentiment,
            'char_count': char_count,
            'word_count': word_count,
            'topics': topics,
            'timestamp': strip_timezone(datetime.now()),
            'style': reply_style,
            'audience': audience_data,
            'tech_data': tech_data
        }

    def _analyze_reply_style(self, reply_text: str) -> Dict[str, float]:
        """
        Analyze the style characteristics of the generated reply
    
        Args:
            reply_text: Generated reply text
        
        Returns:
            Dictionary with style metrics
        """
        text = reply_text.lower()
    
        # Style characteristics to track
        style = {
            'question': 0.0,       # Is the reply asking questions?
            'informative': 0.0,    # Is the reply providing information?
            'humorous': 0.0,       # Is the reply using humor?
            'technical': 0.0,      # Is the reply technical in nature?
            'casual': 0.0,         # Is the reply casual/conversational?
            'emoji_usage': 0.0,    # Does the reply use emojis?
            'educational': 0.0     # Is the reply educational?
        }
    
        # Check for questions
        if '?' in text:
            style['question'] = 1.0
        
        # Check for informative content
        informative_indicators = ['actually', 'in fact', 'data', 'shows', 'indicates', 'according to', 'research', 'analysis']
        style['informative'] = min(1.0, sum(0.2 for ind in informative_indicators if ind in text))
        
        # Check for humor
        humor_indicators = ['lol', 'haha', 'funny', 'joke', 'hilarious', 'lmao', 'rofl', 'lmfao', '!']
        style['humorous'] = min(1.0, sum(0.2 for ind in humor_indicators if ind in text) + (text.count('!') * 0.1))
        
        # Check for technical content
        technical_indicators = ['chart', 'technical', 'support', 'resistance', 'analysis', 'indicator', 'pattern', 'trend']
        style['technical'] = min(1.0, sum(0.2 for ind in technical_indicators if ind in text))
        
        # Check for casual/conversational tone
        casual_indicators = ['hey', 'haha', 'wow', 'cool', 'nice', 'awesome', 'yeah', 'uh', 'so', 'like', 'just']
        style['casual'] = min(1.0, sum(0.2 for ind in casual_indicators if f" {ind} " in f" {text} "))
        
        # Check for emoji usage
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+')
        emojis = emoji_pattern.findall(reply_text)
        style['emoji_usage'] = min(1.0, len(emojis) * 0.2)
        
        # Check for educational content
        educational_indicators = ['learn', 'understand', 'concept', 'means', 'refers to', 'defined as', 'example']
        style['educational'] = min(1.0, sum(0.2 for ind in educational_indicators if ind in text))
    
        return style

    def _extract_mentioned_tokens(self, text: str) -> List[str]:
        """
        Extract cryptocurrency token symbols mentioned in text with improved detection
        
        Args:
            text: Text content of the post
        
        Returns:
            List of token symbols found in the text
        """
        if not text:
            return []

        from config import config
        
        # Get all available tokens from TokenMappingManager (150+ tokens)
        all_symbols = config.token_mapper.get_all_symbols()
        tokens = [symbol.lower() for symbol in all_symbols]
        
        # Look for both direct mentions and $-prefixed mentions
        mentioned = []
        text_lower = text.lower()

        # Check for $-prefixed tokens (stronger signal)
        for token in tokens:
            if f'${token}' in text_lower:
                mentioned.append(token.upper())

        # Check for word-boundary matches
        for token in tokens:
            # Use regex to find whole-word matches
            if re.search(r'\b' + token + r'\b', text_lower):
                if token.upper() not in mentioned:  # Avoid duplicates
                    mentioned.append(token.upper())

        # Build full name mappings dynamically from TokenMappingManager
        full_names = {}
        for symbol in all_symbols:
            token_info = config.token_mapper.get_token_info(symbol)
            if token_info:
                display_name = token_info.get('display_name', '').lower()
                if display_name and display_name != symbol.lower():
                    full_names[display_name] = symbol.upper()

        # Check for mentions by full name
        for name, symbol in full_names.items():
            if name in text_lower and symbol not in mentioned:
                mentioned.append(symbol)

        return mentioned

    def _get_reply_sentiment(self, reply_text: str) -> str:
        """
        Determine the sentiment of a reply (bullish, bearish, or neutral)
    
        Args:
            reply_text: Reply text content
        
        Returns:
            Sentiment as string ('bullish', 'bearish', or 'neutral')
        """
        # Bullish words
        bullish_words = [
            'bullish', 'bull', 'buy', 'long', 'moon', 'rally', 'pump', 'uptrend',
            'breakout', 'strong', 'growth', 'profit', 'gain', 'higher', 'up',
            'optimistic', 'momentum', 'support', 'bounce', 'surge', 'uptick'
        ]
    
        # Bearish words
        bearish_words = [
            'bearish', 'bear', 'sell', 'short', 'dump', 'crash', 'correction',
            'downtrend', 'weak', 'decline', 'loss', 'lower', 'down', 'pessimistic',
            'resistance', 'fall', 'drop', 'slump', 'collapse', 'cautious'
        ]
    
        # Count sentiment words
        reply_lower = reply_text.lower()
    
        bullish_count = sum(1 for word in bullish_words if re.search(r'\b' + word + r'\b', reply_lower))
        bearish_count = sum(1 for word in bearish_words if re.search(r'\b' + word + r'\b', reply_lower))
    
        # Check for negation that might flip sentiment
        negation_words = ['not', 'no', 'never', 'doubt', 'unlikely', 'against']
        for neg in negation_words:
            for bull in bullish_words:
                if f"{neg} {bull}" in reply_lower or f"{neg} really {bull}" in reply_lower:
                    bullish_count -= 1
                    bearish_count += 1
        
            for bear in bearish_words:
                if f"{neg} {bear}" in reply_lower or f"{neg} really {bear}" in reply_lower:
                    bearish_count -= 1
                    bullish_count += 1
    
        # Determine overall sentiment
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
        
    def _check_reply_permissions(self, post_url: str, post_id: str, author_handle: str) -> bool:
        """
        Check if we have permission to reply to a tweet by looking for reply elements.
        This handles cases where the original poster has restricted replies.
        
        Args:
            post_url: URL of the post to check
            post_id: ID of the post
            author_handle: Handle of the post author
            
        Returns:
            True if reply elements are available (we can reply), False if restricted
        """
        try:
            logger.debug(f"Checking reply permissions for post {post_id} by {author_handle}")
            
            # Navigate to the post if we're not already there
            current_url = self.browser.driver.current_url
            if post_url not in current_url:
                logger.debug(f"Navigating to post to check permissions: {post_url}")
                self.browser.driver.get(post_url)
                time.sleep(3)
                
                # Try both twitter.com and x.com URLs if needed
                if "twitter.com" in post_url and "Page not found" in self.browser.driver.title:
                    x_url = post_url.replace("twitter.com", "x.com")
                    logger.debug(f"Trying alternative URL for permission check: {x_url}")
                    self.browser.driver.get(x_url)
                    time.sleep(3)
            
            # Wait for page to load
            page_loaded = False
            for selector in ['article', '[data-testid="tweet"]', 'div[data-testid="cellInnerDiv"]']:
                try:
                    WebDriverWait(self.browser.driver, 8).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    page_loaded = True
                    logger.debug(f"Page loaded for permission check, found element: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if not page_loaded:
                logger.warning(f"Could not load page for permission check: {post_url}")
                return False
            
            # ENHANCED: First check for explicit restriction indicators in the DOM
            # Look for restriction messages that appear before clicking reply
            restriction_selectors = [
                # Look for "Who can reply?" sections
                '//*[contains(text(), "Who can reply?")]',
                '//*[contains(text(), "mentioned can reply")]',
                '//*[contains(text(), "follows can reply")]',
                '//*[contains(text(), "Replying is not available")]',
                # Look for restriction icons or indicators
                '[data-testid="restrictedReplies"]',
                '[aria-label*="restricted"]',
                '[aria-label*="limited"]'
            ]
            
            # Check for restriction indicators using XPath (more flexible than CSS)
            for xpath_selector in restriction_selectors:
                try:
                    elements = self.browser.driver.find_elements(By.XPATH, xpath_selector)
                    if elements:
                        # Found a restriction indicator
                        restriction_text = elements[0].text if elements[0].text else "restriction_indicator_found"
                        logger.logger.info(f"Reply restricted for post {post_id} by {author_handle} - found restriction indicator: {restriction_text}")
                        
                        # Store restriction
                        if self.db and hasattr(self.db, 'store_reply_restriction'):
                            try:
                                self.db.store_reply_restriction(
                                    post_id=post_id,
                                    post_url=post_url,
                                    author_handle=author_handle,
                                    restriction_reason=f"dom_restriction_indicator: {restriction_text}",
                                    timestamp=strip_timezone(datetime.now())
                                )
                            except Exception as db_error:
                                logger.debug(f"Could not store reply restriction in DB: {str(db_error)}")
                        
                        self._add_to_recent_replies(post_id)
                        return False
                except Exception as e:
                    logger.debug(f"Error checking XPath selector '{xpath_selector}': {str(e)}")
                    continue
            
            # ENHANCED: Check page source for restriction text with more variations
            page_text = self.browser.driver.page_source.lower()
            
            # Updated restriction indicators with more variations
            restriction_indicators = [
                "who can reply?",
                "people the author mentioned can reply",
                "accounts @" + author_handle.replace("@", "").lower() + " mentioned can reply",
                "only people the author follows can reply",
                "replying is not available for this tweet",
                "replying is not available for this post",
                "mentioned can reply",
                "follows can reply",
                "reply settings",
                "restricted replies"
            ]
            
            for indicator in restriction_indicators:
                if indicator in page_text:
                    logger.logger.info(f"Reply restricted for post {post_id} by {author_handle} - found restriction text: {indicator}")
                    
                    # Store restriction
                    if self.db and hasattr(self.db, 'store_reply_restriction'):
                        try:
                            self.db.store_reply_restriction(
                                post_id=post_id,
                                post_url=post_url,
                                author_handle=author_handle,
                                restriction_reason=f"page_text_restriction: {indicator}",
                                timestamp=strip_timezone(datetime.now())
                            )
                        except Exception as db_error:
                            logger.debug(f"Could not store reply restriction in DB: {str(db_error)}")
                    
                    self._add_to_recent_replies(post_id)
                    return False
            
            # Check for reply button using all our selectors
            reply_button_found = False
            accessible_reply_button = None
            
            # Enhanced reply button selectors (removed invalid CSS selectors)
            reply_button_selectors = [
                '[data-testid="reply"]',
                'button[aria-label*="Reply"]',
                'div[role="button"][aria-label*="Reply"]',
                'div[aria-label*="Reply"]',
                '[data-testid="reply"] > div',
                'div[role="button"] svg[aria-label*="Reply"]'
            ]
            
            for selector in reply_button_selectors:
                try:
                    reply_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                    if reply_elements:
                        # Check if any of the found elements are actually visible and clickable
                        for element in reply_elements:
                            try:
                                if element.is_displayed() and element.is_enabled():
                                    # Additional check: ensure it's not disabled or has restriction styling
                                    aria_disabled = element.get_attribute('aria-disabled')
                                    is_disabled = element.get_attribute('disabled')
                                    
                                    if aria_disabled != 'true' and is_disabled != 'true':
                                        reply_button_found = True
                                        accessible_reply_button = element
                                        logger.debug(f"Found accessible reply button with selector: {selector}")
                                        break
                            except Exception:
                                continue
                        if reply_button_found:
                            break
                except Exception as e:
                    logger.debug(f"Error checking selector '{selector}': {str(e)}")
                    continue
            
            if not reply_button_found:
                # Log this as a reply restriction and store it to avoid retrying
                logger.logger.info(f"Reply restricted for post {post_id} by {author_handle} - no accessible reply button found")
                
                # Store this restriction in database to avoid future attempts
                if self.db and hasattr(self.db, 'store_reply_restriction'):
                    try:
                        self.db.store_reply_restriction(
                            post_id=post_id,
                            post_url=post_url,
                            author_handle=author_handle,
                            restriction_reason="no_accessible_reply_button",
                            timestamp=strip_timezone(datetime.now())
                        )
                    except Exception as db_error:
                        logger.debug(f"Could not store reply restriction in DB: {str(db_error)}")
                
                # Add to our in-memory cache to avoid immediate retries
                self._add_to_recent_replies(post_id)
                return False
            
            # ENHANCED: Try clicking the reply button to see if restriction dialog appears
            # This is a more thorough check that catches restrictions that only show after clicking
            if accessible_reply_button:
                try:
                    logger.debug("Performing test click on reply button to check for restrictions")
                    
                    # Save current URL to restore later
                    original_url = self.browser.driver.current_url
                    
                    # Click the reply button
                    accessible_reply_button.click()
                    time.sleep(2)  # Wait for any dialogs to appear
                    
                    # Check if a restriction dialog appeared
                    restriction_dialog_selectors = [
                        '//*[contains(text(), "Who can reply?")]',
                        '//*[contains(text(), "mentioned can reply")]',
                        '//*[contains(text(), "follows can reply")]',
                        '[role="dialog"] *[contains(text(), "reply")]'
                    ]
                    
                    restriction_dialog_found = False
                    for dialog_selector in restriction_dialog_selectors:
                        try:
                            dialog_elements = self.browser.driver.find_elements(By.XPATH, dialog_selector)
                            if dialog_elements and any(elem.is_displayed() for elem in dialog_elements):
                                restriction_dialog_found = True
                                restriction_text = dialog_elements[0].text if dialog_elements[0].text else "restriction_dialog"
                                logger.logger.info(f"Reply restricted for post {post_id} by {author_handle} - restriction dialog appeared: {restriction_text}")
                                
                                # Store restriction
                                if self.db and hasattr(self.db, 'store_reply_restriction'):
                                    try:
                                        self.db.store_reply_restriction(
                                            post_id=post_id,
                                            post_url=post_url,
                                            author_handle=author_handle,
                                            restriction_reason=f"restriction_dialog: {restriction_text}",
                                            timestamp=strip_timezone(datetime.now())
                                        )
                                    except Exception as db_error:
                                        logger.debug(f"Could not store reply restriction in DB: {str(db_error)}")
                                
                                self._add_to_recent_replies(post_id)
                                break
                        except Exception as e:
                            logger.debug(f"Error checking dialog selector '{dialog_selector}': {str(e)}")
                            continue
                    
                    if restriction_dialog_found:
                        # Try to close the dialog and return False
                        try:
                            # Try to press escape or click outside to close dialog
                            self.browser.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
                            time.sleep(1)
                        except Exception:
                            pass
                        
                        # If we can't close dialog, navigate back
                        try:
                            if self.browser.driver.current_url != original_url:
                                self.browser.driver.get(original_url)
                                time.sleep(2)
                        except Exception:
                            pass
                        
                        return False
                    
                    # If no restriction dialog appeared, check if we can find the reply textarea
                    reply_textarea_found = False
                    for textarea_selector in self.reply_textarea_selectors:
                        try:
                            textarea_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, textarea_selector)
                            if textarea_elements and any(elem.is_displayed() for elem in textarea_elements):
                                reply_textarea_found = True
                                logger.debug(f"Found reply textarea after clicking: {textarea_selector}")
                                break
                        except Exception:
                            continue
                    
                    if not reply_textarea_found:
                        # This might be a restriction that only shows after clicking
                        logger.logger.info(f"Reply restricted for post {post_id} by {author_handle} - no textarea after clicking reply")
                        
                        if self.db and hasattr(self.db, 'store_reply_restriction'):
                            try:
                                self.db.store_reply_restriction(
                                    post_id=post_id,
                                    post_url=post_url,
                                    author_handle=author_handle,
                                    restriction_reason="no_textarea_after_click",
                                    timestamp=strip_timezone(datetime.now())
                                )
                            except Exception as db_error:
                                logger.debug(f"Could not store reply restriction in DB: {str(db_error)}")
                        
                        self._add_to_recent_replies(post_id)
                        return False
                    
                    # Clean up - try to close the reply dialog if it opened
                    try:
                        self.browser.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
                        time.sleep(1)
                    except Exception:
                        pass
                    
                    # Navigate back to original URL if needed
                    try:
                        if self.browser.driver.current_url != original_url:
                            self.browser.driver.get(original_url)
                            time.sleep(2)
                    except Exception:
                        pass
                    
                except Exception as click_error:
                    logger.debug(f"Error during test click on reply button: {str(click_error)}")
                    # If we can't test click, assume it's available but be cautious
                    pass
            
            # If we get here, reply elements are available
            logger.debug(f"Reply permissions confirmed for post {post_id} by {author_handle}")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking reply permissions for post {post_id}: {str(e)}")
            # If we can't check permissions, assume restricted to be safe
            return False

    def _is_reply_restricted_post(self, post_id: str) -> bool:
        """
        Check if we've already identified this post as having reply restrictions
        
        Args:
            post_id: ID of the post to check
            
        Returns:
            True if this post is known to have reply restrictions
        """
        # Check in-memory cache first
        if post_id in self.recent_replies:
            return True
        
        # Check database if available
        if self.db and hasattr(self.db, 'check_reply_restriction'):
            try:
                return self.db.check_reply_restriction(post_id)
            except Exception as e:
                logger.debug(f"Could not check reply restriction in DB: {str(e)}")
        
        return False    

    def _generate_fallback_reply(self, post: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generate a fallback reply when normal generation fails
    
        Args:
            post: Post data dictionary
            market_data: Market data for context
        
        Returns:
            Generated reply text or None if generation failed
        """
        try:
            # Extract basic post info
            post_text = post.get('text', '')
        
            # Simple keyword-based reply generation without LLM if possible
        
            # Detect key topics in post
            topics = []
        
            # Check for token mentions
            token_keywords = {
                'btc': ['btc', 'bitcoin', 'satoshi'], 
                'eth': ['eth', 'ethereum', 'vitalik'],
                'sol': ['sol', 'solana'],
                'xrp': ['xrp', 'ripple'],
                'bnb': ['bnb', 'binance coin']
            }
        
            mentioned_tokens = []
            for token, keywords in token_keywords.items():
                if any(kw in post_text.lower() for kw in keywords):
                    mentioned_tokens.append(token)
                
            # Check for market sentiment
            sentiment_keywords = {
                'bullish': ['bull', 'bullish', 'up', 'moon', 'pump', 'rally', 'green'],
                'bearish': ['bear', 'bearish', 'down', 'dump', 'crash', 'red', 'correction'],
                'question': ['?', 'what', 'why', 'how', 'when', 'should']
            }
        
            post_sentiment = 'neutral'
            for sentiment, keywords in sentiment_keywords.items():
                if any(kw in post_text.lower() for kw in keywords):
                    post_sentiment = sentiment
                    break
        
            # Find a trending token from market data
            trending_tokens = []
            if isinstance(market_data, dict):
                for token, data in market_data.items():
                    if isinstance(data, dict) and abs(data.get('price_change_percentage_24h', 0)) > 3.0:
                        trending_tokens.append((token, abs(data.get('price_change_percentage_24h', 0))))
    
                # Sort by price change (highest first)
                trending_tokens.sort(key=lambda x: x[1], reverse=True)
    
                if trending_tokens:
                    reply_token = trending_tokens[0][0]
                else:
                    # Fallback to BTC
                    reply_token = "BTC"
            else:
                # Fallback if market_data is not a dict
                reply_token = "BTC"

            # Generate appropriate reply based on sentiment and token
            token_data = market_data.get(reply_token, {}) if isinstance(market_data, dict) else {}
            token_price = token_data.get('current_price', 0) if isinstance(token_data, dict) else 0
            price_change = token_data.get('price_change_percentage_24h', 0) if isinstance(token_data, dict) else 0
        
            if post_sentiment == 'question':
                # Question templates
                question_replies = [
                    f"Looking at {reply_token}'s recent action (${token_price:,.2f}, {price_change:+.2f}%), I'd say the technical structure is still developing. What's your timeframe?",
                    f"Interesting question. {reply_token} is showing {price_change:+.2f}% movement today. The volume profile suggests institutional interest.",
                    f"If you're looking at {reply_token}, note that it's at ${token_price:,.2f} now with some key resistance levels ahead.",
                    f"On {reply_token}, I'm watching the ${token_price:,.2f} level closely. This price zone has been significant historically.",
                    f"Good question about {reply_token}. At ${token_price:,.2f} ({price_change:+.2f}%), the momentum indicators are giving mixed signals.",
                    f"For {reply_token}, the current ${token_price:,.2f} price point is interesting. The {price_change:+.2f}% move suggests some underlying strength.",
                    f"That's a solid question. {reply_token} is trading at ${token_price:,.2f} with {price_change:+.2f}% movement, but the real story is in the order flow.",
                    f"Regarding {reply_token}, the ${token_price:,.2f} level is worth monitoring. Today's {price_change:+.2f}% change shows some volatility returning.",
                    f"Great question timing. {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is approaching a key decision point on the charts.",
                    f"On {reply_token}, the ${token_price:,.2f} price with {price_change:+.2f}% movement today has me looking at the weekly patterns.",
                    f"Valid question about {reply_token}. The current ${token_price:,.2f} level and {price_change:+.2f}% daily move suggest accumulation patterns.",
                    f"For {reply_token}, I'm seeing ${token_price:,.2f} with {price_change:+.2f}% change. The market structure is setting up for something interesting.",
                    f"That's the right question to ask. {reply_token} is at ${token_price:,.2f} showing {price_change:+.2f}% movement with decent volume backing it.",
                    f"Concerning {reply_token}, the ${token_price:,.2f} price point represents a {price_change:+.2f}% shift that aligns with broader market sentiment.",
                    f"Good timing on this question. {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is testing some important technical levels.",
                    f"With {reply_token} at ${token_price:,.2f} and {price_change:+.2f}% movement, I'd say the risk/reward setup is worth evaluating.",
                    f"That's an important question. {reply_token} showing ${token_price:,.2f} with {price_change:+.2f}% change has caught institutional attention lately.",
                    f"For {reply_token}, the ${token_price:,.2f} level with today's {price_change:+.2f}% move is creating some interesting chart patterns.",
                    f"Solid question about {reply_token}. At ${token_price:,.2f} ({price_change:+.2f}%), the momentum is building but needs confirmation.",
                    f"On {reply_token}, I'm tracking the ${token_price:,.2f} price closely. The {price_change:+.2f}% movement suggests smart money positioning.",
                    f"That's a key question right now. {reply_token} at ${token_price:,.2f} with {price_change:+.2f}% change is showing relative strength.",
                    f"Regarding {reply_token}, the ${token_price:,.2f} level represents a {price_change:+.2f}% move that's worth monitoring for continuation.",
                    f"Interesting question about {reply_token}. The ${token_price:,.2f} price point with {price_change:+.2f}% movement is creating setup opportunities.",
                    f"For {reply_token}, I'm seeing ${token_price:,.2f} ({price_change:+.2f}%) which puts it in an interesting position relative to moving averages.",
                    f"That's the question everyone should be asking. {reply_token} at ${token_price:,.2f} with {price_change:+.2f}% change shows conviction.",
                    f"Good question on {reply_token}. The ${token_price:,.2f} level and {price_change:+.2f}% daily move are creating some buzz in the options market.",
                    f"With {reply_token} at ${token_price:,.2f} showing {price_change:+.2f}% movement, I'm watching how it reacts to key support zones.",
                    f"Valid question about {reply_token}. At ${token_price:,.2f} ({price_change:+.2f}%), the technical picture is getting more interesting.",
                    f"That's a timely question. {reply_token} trading at ${token_price:,.2f} with {price_change:+.2f}% change is approaching critical confluence levels.",
                    f"On {reply_token}, the ${token_price:,.2f} price with today's {price_change:+.2f}% movement is setting up what could be a significant breakout pattern."
                ]
                return random.choice(question_replies)
            
            elif post_sentiment == 'bullish':
                # Bullish templates
                bullish_replies = [
                    f"The volume profile on {reply_token} does support your bullish thesis. Still watching that ${token_price:,.2f} level though.",
                    f"Agreed on the bullish signals, but {reply_token}'s ${token_price:,.2f} price needs to hold above the 21-day EMA for confirmation.",
                    f"Bullish, but cautiously so. {reply_token} needs volume confirmation at these ${token_price:,.2f} levels.",
                    f"The PA looks bullish, though {reply_token} at ${token_price:,.2f} is facing some strong historical resistance.",
                    f"I'm seeing the same bullish divergence on {reply_token}. The ${token_price:,.2f} level with {price_change:+.2f}% confirms momentum building.",
                    f"Bullish setup is forming on {reply_token}. At ${token_price:,.2f} ({price_change:+.2f}%), the institutional flow is definitely turning positive.",
                    f"The bullish case for {reply_token} is solid. ${token_price:,.2f} with {price_change:+.2f}% movement shows conviction from smart money.",
                    f"Agreed on the bullish outlook. {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is breaking above key resistance with authority.",
                    f"The bullish momentum on {reply_token} is undeniable. ${token_price:,.2f} represents a {price_change:+.2f}% move that's attracting serious capital.",
                    f"Bullish sentiment justified here. {reply_token} holding ${token_price:,.2f} with {price_change:+.2f}% strength shows the trend is your friend.",
                    f"I'm aligned with your bullish view on {reply_token}. The ${token_price:,.2f} level with {price_change:+.2f}% movement screams accumulation.",
                    f"Bullish thesis checks out. {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is showing textbook institutional buying patterns.",
                    f"The bullish structure on {reply_token} is clean. ${token_price:,.2f} with {price_change:+.2f}% movement has all the hallmarks of a sustained move.",
                    f"Bullish and rightfully so. {reply_token} breaking ${token_price:,.2f} with {price_change:+.2f}% strength is exactly what we want to see.",
                    f"Your bullish call on {reply_token} is spot on. The ${token_price:,.2f} level representing {price_change:+.2f}% shows real institutional interest.",
                    f"Bullish momentum is building on {reply_token}. At ${token_price:,.2f} ({price_change:+.2f}%), the options flow is heavily skewed to calls.",
                    f"The bullish case for {reply_token} is compelling. ${token_price:,.2f} with {price_change:+.2f}% movement is breaking multi-month consolidation.",
                    f"Bullish and backed by data. {reply_token} at ${token_price:,.2f} showing {price_change:+.2f}% has whale wallets accumulating aggressively.",
                    f"I'm bullish on {reply_token} too. The ${token_price:,.2f} price point with {price_change:+.2f}% movement has technical and fundamental alignment.",
                    f"Bullish setup is textbook perfect. {reply_token} holding ${token_price:,.2f} ({price_change:+.2f}%) above the 50-day with increasing volume.",
                    f"The bullish narrative on {reply_token} is gaining traction. ${token_price:,.2f} representing {price_change:+.2f}% shows smart money positioning.",
                    f"Bullish momentum is accelerating on {reply_token}. At ${token_price:,.2f} with {price_change:+.2f}% movement, the trend is undeniable.",
                    f"Your bullish outlook on {reply_token} is validated. The ${token_price:,.2f} level with {price_change:+.2f}% strength shows conviction buying.",
                    f"Bullish and the charts confirm it. {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is in a clear uptrend with volume support.",
                    f"The bullish sentiment on {reply_token} is infectious. ${token_price:,.2f} with {price_change:+.2f}% movement has momentum players piling in.",
                    f"Bullish thesis is rock solid. {reply_token} breaking ${token_price:,.2f} with {price_change:+.2f}% strength has institutional fingerprints all over it.",
                    f"I'm bullish on {reply_token} for the same reasons. The ${token_price:,.2f} level showing {price_change:+.2f}% has perfect risk/reward setup.",
                    f"Bullish momentum is undeniable here. {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is attracting serious institutional flow.",
                    f"The bullish case for {reply_token} is ironclad. ${token_price:,.2f} representing {price_change:+.2f}% movement has all technical boxes checked.",
                    f"Bullish and the market agrees. {reply_token} holding ${token_price:,.2f} with {price_change:+.2f}% strength is showing textbook accumulation patterns."
                ]
                return random.choice(bullish_replies)
            
            elif post_sentiment == 'bearish':
                # Bearish templates - completing this section
                bearish_replies = [
                    f"I see your bearish case, but {reply_token}'s holding strong at ${token_price:,.2f}. Keep an eye on the stoch RSI for possible divergence.",
                    f"The bears might be getting exhausted here. {reply_token} at ${token_price:,.2f} is showing some accumulation patterns.",
                    f"Bearish sentiment is high, but {reply_token}'s technicals at ${token_price:,.2f} don't fully support more downside yet.",
                    f"While the market looks weak, {reply_token} at ${token_price:,.2f} is approaching key support levels that could provide a bounce.",
                    f"The bearish pressure on {reply_token} is noted, but ${token_price:,.2f} with {price_change:+.2f}% is holding critical support surprisingly well.",
                    f"Bearish thesis has merit, though {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is showing hidden bullish divergence on the RSI.",
                    f"I understand the bearish sentiment, but {reply_token}'s ${token_price:,.2f} level with {price_change:+.2f}% movement suggests smart money is stepping in.",
                    f"Bearish calls are everywhere, yet {reply_token} holding ${token_price:,.2f} ({price_change:+.2f}%) indicates institutional support at these levels.",
                    f"The bearish narrative on {reply_token} is loud, but ${token_price:,.2f} with {price_change:+.2f}% shows surprising resilience against selling pressure.",
                    f"Bearish sentiment is justified, though {reply_token} at ${token_price:,.2f} representing {price_change:+.2f}% is finding buyers at key fibonacci levels.",
                    f"I hear the bearish concerns, but {reply_token}'s ${token_price:,.2f} price with {price_change:+.2f}% movement is creating a potential bear trap setup.",
                    f"Bearish outlook is understandable, yet {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is showing signs of capitulation exhaustion.",
                    f"The bearish case for {reply_token} is noted, but ${token_price:,.2f} with {price_change:+.2f}% suggests we might be near a significant bottom.",
                    f"Bearish sentiment is thick, though {reply_token} holding ${token_price:,.2f} with {price_change:+.2f}% strength shows contrarian opportunity emerging.",
                    f"I see the bearish pressure, but {reply_token} at ${token_price:,.2f} representing {price_change:+.2f}% is approaching oversold territory on multiple timeframes.",
                    f"Bearish calls are valid, yet {reply_token}'s ${token_price:,.2f} level with {price_change:+.2f}% movement is creating textbook reversal patterns.",
                    f"The bearish momentum on {reply_token} is real, but ${token_price:,.2f} ({price_change:+.2f}%) suggests we're hitting major institutional buy zones.",
                    f"Bearish sentiment is palpable, though {reply_token} at ${token_price:,.2f} with {price_change:+.2f}% shows volume climax that often marks bottoms.",
                    f"I understand the bearish view, but {reply_token}'s ${token_price:,.2f} price showing {price_change:+.2f}% has whale accumulation signatures all over it.",
                    f"Bearish pressure is mounting, yet {reply_token} holding ${token_price:,.2f} ({price_change:+.2f}%) is displaying classic double bottom characteristics.",
                    f"The bearish thesis on {reply_token} has merit, but ${token_price:,.2f} with {price_change:+.2f}% movement suggests sellers are getting exhausted.",
                    f"Bearish sentiment is widespread, though {reply_token} at ${token_price:,.2f} representing {price_change:+.2f}% shows hidden strength in the order book.",
                    f"I acknowledge the bearish concerns, but {reply_token}'s ${token_price:,.2f} level with {price_change:+.2f}% is creating massive divergence on momentum indicators.",
                    f"Bearish outlook is reasonable, yet {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is showing institutional buying at every dip.",
                    f"The bearish case for {reply_token} is heard, but ${token_price:,.2f} with {price_change:+.2f}% suggests we're in prime contrarian territory.",
                    f"Bearish sentiment is justified, though {reply_token} holding ${token_price:,.2f} with {price_change:+.2f}% strength indicates major support confluence.",
                    f"I see the bearish pressure, but {reply_token} at ${token_price:,.2f} representing {price_change:+.2f}% is approaching key psychological support levels.",
                    f"Bearish momentum is strong, yet {reply_token}'s ${token_price:,.2f} price with {price_change:+.2f}% movement shows classic washout characteristics.",
                    f"The bearish narrative on {reply_token} is compelling, but ${token_price:,.2f} ({price_change:+.2f}%) suggests smart money is quietly accumulating.",
                    f"Bearish sentiment is extreme, though {reply_token} at ${token_price:,.2f} with {price_change:+.2f}% is showing all the hallmarks of a major reversal setup."
                ]
                return random.choice(bearish_replies)
            
            else:  # neutral sentiment
                # Neutral templates
                neutral_replies = [
                    f"The market's indecisive right now. {reply_token} at ${token_price:,.2f} is consolidating within a tight range.",
                    f"Watching {reply_token} closely at ${token_price:,.2f}. The volume profile is suggesting accumulation but needs confirmation.",
                    f"{reply_token} is in a critical zone at ${token_price:,.2f}. The next 24 hours could determine the mid-term trend.",
                    f"Interesting view. {reply_token}'s current price action at ${token_price:,.2f} suggests smart money might be positioning quietly.",
                    f"The sideways action on {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is creating a coiling pattern worth monitoring.",
                    f"Mixed signals from {reply_token} right now. At ${token_price:,.2f} with {price_change:+.2f}% movement, it's in a decision zone.",
                    f"Neutral territory for {reply_token}. The ${token_price:,.2f} level showing {price_change:+.2f}% has equal probability of breaking either way.",
                    f"The consolidation on {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is textbook - waiting for a catalyst to show direction.",
                    f"Balanced view on {reply_token}. At ${token_price:,.2f} with {price_change:+.2f}% movement, the risk/reward is symmetrical here.",
                    f"The market's speaking in whispers about {reply_token}. ${token_price:,.2f} representing {price_change:+.2f}% shows institutional indecision.",
                    f"Neutral stance makes sense for {reply_token}. The ${token_price:,.2f} level with {price_change:+.2f}% is right at the crossroads.",
                    f"The price action on {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is creating a perfect symmetrical triangle formation.",
                    f"Watching {reply_token} with interest. At ${token_price:,.2f} showing {price_change:+.2f}%, it's in that no-man's land between major levels.",
                    f"The neutrality on {reply_token} is warranted. ${token_price:,.2f} with {price_change:+.2f}% movement sits right at the 50-day moving average.",
                    f"Balanced perspective on {reply_token}. The ${token_price:,.2f} price point with {price_change:+.2f}% shows equal buying and selling pressure.",
                    f"The consolidation phase for {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is healthy - building energy for the next move.",
                    f"Neutral ground for {reply_token}. At ${token_price:,.2f} with {price_change:+.2f}% movement, the market is waiting for fresh information.",
                    f"The sideways grind on {reply_token} at ${token_price:,.2f} representing {price_change:+.2f}% is creating option sellers' paradise.",
                    f"Measured approach to {reply_token}. The ${token_price:,.2f} level with {price_change:+.2f}% sits perfectly between key support and resistance.",
                    f"The neutrality on {reply_token} is tactical. At ${token_price:,.2f} ({price_change:+.2f}%), it's in the eye of the storm before the next move.",
                    f"Balanced view warranted for {reply_token}. The ${token_price:,.2f} price with {price_change:+.2f}% movement shows institutional patience.",
                    f"The consolidation on {reply_token} at ${token_price:,.2f} with {price_change:+.2f}% is creating a powder keg - waiting for the spark.",
                    f"Neutral territory makes sense for {reply_token}. At ${token_price:,.2f} representing {price_change:+.2f}%, the technicals are perfectly balanced.",
                    f"The sideways action on {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is building the foundation for the next major move.",
                    f"Watching {reply_token} with keen interest. The ${token_price:,.2f} level with {price_change:+.2f}% is at a critical inflection point.",
                    f"The neutrality on {reply_token} is strategic. At ${token_price:,.2f} showing {price_change:+.2f}%, both bulls and bears are regrouping.",
                    f"Balanced outlook for {reply_token}. The ${token_price:,.2f} price point with {price_change:+.2f}% movement is right at the pivot zone.",
                    f"The consolidation pattern on {reply_token} at ${token_price:,.2f} ({price_change:+.2f}%) is creating a high-probability setup for patient traders.",
                    f"Neutral stance is wise for {reply_token}. At ${token_price:,.2f} with {price_change:+.2f}% movement, the market is coiled and ready to spring.",
                    f"The sideways chop on {reply_token} at ${token_price:,.2f} representing {price_change:+.2f}% is the calm before the storm - direction TBD."
                ]
                return random.choice(neutral_replies)
        
        except Exception as e:
            logger.error(f"Fallback Reply Generation: {str(e)}")
        
            # Ultimate fallback replies if everything else fails
            emergency_replies = [
                "Interesting perspective on the market. The chart patterns tell a different story though.",
                "Not sure I agree with that take. The technicals suggest a more nuanced picture.",
                "The market structure is hinting at something else entirely. Worth watching closely.",
                "That's one way to interpret it. I'm seeing conflicting signals in the order flow data.",
                "Charts suggest we might be at a critical inflection point. Could go either way from here.",
                "The market's telling two different stories right now. Price action vs. volume flow creates interesting dynamics.",
                "I'm reading the tape differently. The institutional footprints suggest a more complex narrative is developing.",
                "That's a valid viewpoint, though the options flow is painting a completely different picture from here.",
                "The momentum indicators are giving mixed signals. This setup requires patience and discipline.",
                "Interesting analysis. The market microstructure data I'm seeing suggests we're in for some volatility.",
                "The technical picture is more layered than it appears. Multiple timeframes are showing divergent signals.",
                "That perspective has merit, but the smart money seems to be positioning for something entirely different.",
                "The market's language is subtle right now. What looks obvious on the surface might be misdirection.",
                "I'm tracking several key levels that could change this entire narrative in the next few sessions.",
                "The institutional flow tells a story that contradicts the obvious price action we're seeing.",
                "That's one interpretation. The volume profile analysis suggests the real move hasn't started yet.",
                "The market structure is evolving in ways that make traditional analysis more challenging right now.",
                "I'm seeing some unusual patterns in the order book that don't align with conventional wisdom.",
                "The technical confluence at these levels creates multiple scenarios worth monitoring closely.",
                "That viewpoint makes sense on paper, but the actual execution data shows institutional disagreement.",
                "The market's giving us a masterclass in complexity. Simple narratives rarely capture the full picture.",
                "I'm tracking some divergences that suggest the obvious trade might be the wrong trade here.",
                "The algorithmic activity I'm monitoring suggests we're in a period of strategic repositioning.",
                "That's a reasonable take, though the cross-asset correlations are hinting at broader themes developing.",
                "The market timing indicators are flashing yellow. This environment rewards patience over aggression.",
                "I'm seeing institutional fingerprints that don't match the retail sentiment we're observing.",
                "The technical setup is more intricate than it appears. Multiple scenarios remain in play.",
                "That analysis has validity, but the derivative markets are suggesting a different probability matrix.",
                "The market's current behavior suggests we're in a transition phase that defies easy categorization.",
                "I'm monitoring several key catalysts that could completely reshape the narrative from these levels."
            ]
            return random.choice(emergency_replies)

    @ensure_naive_datetimes
    def post_reply(self, post: Dict[str, Any], reply_text: str) -> bool:
        """
        Navigate to the post and submit a reply with improved resilience
        
        Args:
            post: Post data dictionary
            reply_text: Text to reply with
        
        Returns:
            True if reply was successfully posted, False otherwise
        """
        if not reply_text:
            logger.error("Cannot post empty reply")
            return False
        
        # Check if post has a URL to navigate to
        post_url = post.get('post_url')
        if not post_url:
            logger.error("No URL available for the post")
            return False
        
        post_id = post.get('post_id', 'unknown')
        author_handle = post.get('author_handle', '@unknown')

        # Check if we've already replied to this post (memory cache)
        if self._already_replied(post_id):
            logger.logger.info(f"Already replied to post {post_id} by {author_handle}")
            return False
        
        # Check if this post is known to have reply restrictions
        if self._is_reply_restricted_post(post_id):
            logger.logger.info(f"Skipping post {post_id} by {author_handle} - known reply restrictions")
            return False
        
        # Check if we've already replied to this post (database)
        if self.db and hasattr(self.db, 'check_if_post_replied') and self.db.check_if_post_replied(post_id, post_url):
            logger.logger.info(f"Already replied to post {post_id} by {author_handle} (DB)")
            return False
        
        # NEW: Check reply permissions before attempting to reply
        if not self._check_reply_permissions(post_url, post_id, author_handle):
            logger.logger.info(f"Skipping post {post_id} by {author_handle} - reply restrictions detected")
            return False
        
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                # Navigate to the post (only if not already there from permission check)
                current_url = self.browser.driver.current_url
                if post_url not in current_url:
                    logger.debug(f"Navigating to post: {post_url}")
                    self.browser.driver.get(post_url)
                    time.sleep(3)  # Allow time for page to load
                
                    # Try both twitter.com and x.com URLs if needed
                    if "twitter.com" in post_url and "Page not found" in self.browser.driver.title:
                        x_url = post_url.replace("twitter.com", "x.com")
                        logger.debug(f"Trying alternative URL: {x_url}")
                        self.browser.driver.get(x_url)
                        time.sleep(3)
            
                # Wait for page to load - try multiple selectors
                page_loaded = False
                for selector in ['article', '[data-testid="tweet"]', 'div[data-testid="cellInnerDiv"]']:
                    try:
                        WebDriverWait(self.browser.driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        page_loaded = True
                        logger.debug(f"Page loaded, found element with selector: {selector}")
                        break
                    except TimeoutException:
                        continue
            
                if not page_loaded:
                    logger.warning("Could not confirm page loaded with any selector")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
            
                # Take screenshot for debugging
                try:
                    debug_screenshot = f"reply_debug_{strip_timezone(datetime.now()).strftime('%Y%m%d_%H%M%S')}.png"
                    self.browser.driver.save_screenshot(debug_screenshot)
                    logger.debug(f"Saved reply debugging screenshot to {debug_screenshot}")
                except Exception as ss_error:
                    logger.debug(f"Failed to save debugging screenshot: {str(ss_error)}")
            
                # Find and click the reply button using multiple selectors
                reply_button = None
                for selector in self.reply_button_selectors:
                    try:
                        logger.debug(f"Trying to find reply button with selector: {selector}")
                        reply_button = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        if reply_button:
                            logger.debug(f"Found reply button using selector: {selector}")
                            break
                    except Exception as e:
                        logger.debug(f"Could not find reply button with selector '{selector}': {str(e)}")
                        continue
            
                if not reply_button:
                    logger.warning("Could not find reply button with any selector")
                    # NEW: Check if this is a reply restriction and store it
                    logger.logger.info(f"Reply restricted for post {post_id} by {author_handle} - no reply button found during attempt")
                    if self.db and hasattr(self.db, 'store_reply_restriction'):
                        try:
                            self.db.store_reply_restriction(
                                post_id=post_id,
                                post_url=post_url,
                                author_handle=author_handle,
                                restriction_reason="no_reply_button_during_attempt",
                                timestamp=strip_timezone(datetime.now())
                            )
                        except Exception as db_error:
                            logger.debug(f"Could not store reply restriction in DB: {str(db_error)}")
                    self._add_to_recent_replies(post_id)
                    return False
            
                # Try multiple methods to click the reply button
                click_success = False
            
                # Method 1: JavaScript click
                try:
                    logger.debug("Attempting to click reply button using JavaScript")
                    self.browser.driver.execute_script("arguments[0].click();", reply_button)
                    click_success = True
                    logger.debug("Clicked reply button with JavaScript")
                except Exception as js_error:
                    logger.debug(f"JavaScript click failed: {str(js_error)}")
            
                # Method 2: Standard click if JavaScript failed
                if not click_success:
                    try:
                        logger.debug("Attempting to click reply button using standard click")
                        reply_button.click()
                        click_success = True
                        logger.debug("Clicked reply button with standard click")
                    except Exception as click_error:
                        logger.debug(f"Standard click failed: {str(click_error)}")
            
                # Method 3: Action chains if both methods failed
                if not click_success:
                    try:
                        logger.debug("Attempting to click reply button using ActionChains")
                        ActionChains(self.browser.driver).move_to_element(reply_button).click().perform()
                        click_success = True
                        logger.debug("Clicked reply button with ActionChains")
                    except Exception as action_error:
                        logger.debug(f"ActionChains click failed: {str(action_error)}")
            
                if not click_success:
                    logger.warning("Could not click reply button with any method")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
            
                # Wait for reply area to appear
                time.sleep(2)
            
                # Find the reply textarea using multiple selectors
                reply_textarea = None
                for selector in self.reply_textarea_selectors:
                    try:
                        logger.debug(f"Trying to find reply textarea with selector: {selector}")
                        reply_textarea = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        if reply_textarea:
                            logger.debug(f"Found reply textarea using selector: {selector}")
                            break
                    except Exception as e:
                        logger.debug(f"Could not find textarea with selector '{selector}': {str(e)}")
                        continue
            
                if not reply_textarea:
                    logger.warning("Could not find reply textarea with any selector")
                    # NEW: Check if this is a reply restriction and store it
                    logger.logger.info(f"Reply restricted for post {post_id} by {author_handle} - no textarea found after clicking reply")
                    if self.db and hasattr(self.db, 'store_reply_restriction'):
                        try:
                            self.db.store_reply_restriction(
                                post_id=post_id,
                                post_url=post_url,
                                author_handle=author_handle,
                                restriction_reason="no_textarea_after_reply_click",
                                timestamp=strip_timezone(datetime.now())
                            )
                        except Exception as db_error:
                            logger.debug(f"Could not store reply restriction in DB: {str(db_error)}")
                    self._add_to_recent_replies(post_id)
                    return False
            
                # Focus the textarea
                try:
                    reply_textarea.click()
                    time.sleep(1)
                except Exception as click_error:
                    logger.debug(f"Could not click textarea: {str(click_error)}")
                    # Try scrolling to it first
                    try:
                        self.browser.driver.execute_script("arguments[0].scrollIntoView(true);", reply_textarea)
                        time.sleep(1)
                        reply_textarea.click()
                    except Exception as e:
                        logger.debug(f"Could not focus textarea after scroll: {str(e)}")
                        retry_count += 1
                        time.sleep(self.retry_delay)
                        continue
            
                # Clear any existing text
                try:
                    reply_textarea.clear()
                except Exception as clear_error:
                    logger.debug(f"Could not clear textarea: {str(clear_error)}")
            
                # Try multiple methods to enter text
                text_entry_success = False
            
                # Method 1: Direct send_keys
                try:
                    logger.debug("Attempting to enter reply text using send_keys")
                    reply_textarea.send_keys(reply_text)
                    text_entry_success = True
                    logger.debug("Entered text using send_keys")
                except Exception as keys_error:
                    logger.debug(f"send_keys failed: {str(keys_error)}")
            
                # Method 2: JavaScript to set value
                if not text_entry_success:
                    try:
                        logger.debug("Attempting to enter reply text using JavaScript")
                        self.browser.driver.execute_script(
                            "arguments[0].textContent = arguments[1]; arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", 
                            reply_textarea, 
                            reply_text
                        )
                        text_entry_success = True
                        logger.debug("Entered text using JavaScript")
                    except Exception as js_error:
                        logger.debug(f"JavaScript text entry failed: {str(js_error)}")
            
                # Method 3: ActionChains
                if not text_entry_success:
                    try:
                        logger.debug("Attempting to enter reply text using ActionChains")
                        ActionChains(self.browser.driver).move_to_element(reply_textarea).click().send_keys(reply_text).perform()
                        text_entry_success = True
                        logger.debug("Entered text using ActionChains")
                    except Exception as action_error:
                        logger.debug(f"ActionChains text entry failed: {str(action_error)}")
            
                # Method 4: Char by char
                if not text_entry_success:
                    try:
                        logger.debug("Attempting to enter reply text character by character")
                        for char in reply_text:
                            reply_textarea.send_keys(char)
                            time.sleep(0.05)
                        text_entry_success = True
                        logger.debug("Entered text character by character")
                    except Exception as char_error:
                        logger.debug(f"Character by character entry failed: {str(char_error)}")
            
                if not text_entry_success:
                    logger.warning("Could not enter reply text with any method")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
            
                # Wait for the text to be processed
                time.sleep(2)
            
                # Find the send button using multiple selectors
                send_button = None
                for selector in self.reply_send_button_selectors:
                    try:
                        logger.debug(f"Trying to find send button with selector: {selector}")
                        send_button = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        if send_button:
                            logger.debug(f"Found send button using selector: {selector}")
                            break
                    except Exception as e:
                        logger.debug(f"Could not find send button with selector '{selector}': {str(e)}")
                        continue
            
                if not send_button:
                    logger.warning("Could not find send button with any selector")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
            
                # Try multiple methods to click the send button
                send_success = False
            
                # Method 1: JavaScript click
                try:
                    logger.debug("Attempting to click send button using JavaScript")
                    self.browser.driver.execute_script("arguments[0].click();", send_button)
                    send_success = True
                    logger.debug("Clicked send button with JavaScript")
                except Exception as js_error:
                    logger.debug(f"JavaScript click failed: {str(js_error)}")
            
                # Method 2: Standard click if JavaScript failed
                if not send_success:
                    try:
                        logger.debug("Attempting to click send button using standard click")
                        send_button.click()
                        send_success = True
                        logger.debug("Clicked send button with standard click")
                    except Exception as click_error:
                        logger.debug(f"Standard click failed: {str(click_error)}")
            
                # Method 3: Action chains if both methods failed
                if not send_success:
                    try:
                        logger.debug("Attempting to click send button using ActionChains")
                        ActionChains(self.browser.driver).move_to_element(send_button).click().perform()
                        send_success = True
                        logger.debug("Clicked send button with ActionChains")
                    except Exception as action_error:
                        logger.debug(f"ActionChains click failed: {str(action_error)}")
            
                if not send_success:
                    logger.warning("Could not click send button with any method")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
            
                # Wait for reply to be sent
                time.sleep(5)
            
                # Verify that the reply was posted successfully
                if self._verify_reply_posted():
                    logger.logger.info(f"Successfully replied to post by {author_handle}")
                
                    # Add to recent replies list (memory cache)
                    self._add_to_recent_replies(post_id)
                
                    # Store in database if available
                    if self.db and hasattr(self.db, 'store_reply'):
                        self.db.store_reply(
                            post_id=post_id,
                            post_url=post_url,
                            post_author=author_handle,
                            post_text=post.get('text', ''),
                            reply_text=reply_text,
                            reply_time=strip_timezone(datetime.now())
                        )
                
                    return True
                else:
                    logger.warning(f"Reply verification failed, may not have been posted")
                    retry_count += 1
                    time.sleep(self.retry_delay)
                
            except TimeoutException as te:
                logger.warning(f"Timeout while trying to reply to post: {str(te)} (attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(self.retry_delay)
            
            except (ElementClickInterceptedException, NoSuchElementException) as e:
                logger.warning(f"Element interaction error: {str(e)} (attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(self.retry_delay)
            
            except Exception as e:
                logger.error(f"Reply Posting: {str(e)}")
                retry_count += 1
                time.sleep(self.retry_delay)

        logger.error(f"Failed to post reply to {author_handle} after {self.max_retries} attempts")
        return False

    def _verify_reply_posted(self) -> bool:
        """
        Verify that a reply was successfully posted using multiple methods
    
        Returns:
            True if verification succeeded, False otherwise
        """
        try:
            # Method 1: Check if reply compose area is no longer visible
            textarea_gone = False
            for selector in self.reply_textarea_selectors:
                try:
                    WebDriverWait(self.browser.driver, 5).until_not(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    textarea_gone = True
                    logger.debug(f"Verified reply posted - textarea {selector} no longer visible")
                    break
                except TimeoutException:
                    continue
        
            if textarea_gone:
                return True
        
            # Method 2: Check if success indicators are present
            success_indicators = [
                '[data-testid="toast"]',  # Success toast notification
                '[role="alert"]',         # Alert role that might indicate success
                '.css-1dbjc4n[style*="background-color: rgba(0, 0, 0, 0)"]'  # Modal closed
            ]
        
            for indicator in success_indicators:
                try:
                    WebDriverWait(self.browser.driver, 3).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, indicator))
                    )
                    logger.debug(f"Verified reply posted - found success indicator: {indicator}")
                    return True
                except TimeoutException:
                    continue
        
            # Method 3: Check if URL changed
            current_url = self.browser.driver.current_url
            if '/compose/' not in current_url:
                logger.debug("Verified reply posted - no longer on compose URL")
                return True
        
            # Method 4: Check if send button is disabled or gone
            for selector in self.reply_send_button_selectors:
                try:
                    # Check for disabled state
                    send_buttons = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                    if not send_buttons:
                        logger.debug(f"Verified reply posted - send button {selector} no longer present")
                        return True
                
                    # If button exists, check if it's disabled
                    for button in send_buttons:
                        aria_disabled = button.get_attribute('aria-disabled')
                        is_disabled = button.get_attribute('disabled')
                        if aria_disabled == 'true' or is_disabled == 'true':
                            logger.debug(f"Verified reply posted - send button is now disabled")
                            return True
                except Exception:
                    continue
        
            # If we get here, no verification method succeeded
            logger.warning("Could not verify if reply was posted with any method")
        
            # Take screenshot for debugging
            try:
                debug_screenshot = f"reply_verification_{strip_timezone(datetime.now()).strftime('%Y%m%d_%H%M%S')}.png"
                self.browser.driver.save_screenshot(debug_screenshot)
                logger.debug(f"Saved verification debugging screenshot to {debug_screenshot}")
            except Exception as e:
                logger.debug(f"Failed to save verification screenshot: {str(e)}")
        
            return False
        
        except Exception as e:
            logger.warning(f"Reply verification error: {str(e)}")
            return False

    def _already_replied(self, post_id: str) -> bool:
        """
        Check if we've already replied to a post (memory cache)
    
        Args:
            post_id: Unique identifier for the post
        
        Returns:
            True if we've already replied, False otherwise
        """
        return post_id in self.recent_replies

    def _add_to_recent_replies(self, post_id: str) -> None:
        """
        Add a post ID to the recent replies list
    
        Args:
            post_id: Unique identifier for the post
        """
        if post_id in self.recent_replies:
            return
        
        self.recent_replies.append(post_id)
    
        # Keep the list at a reasonable size
        if len(self.recent_replies) > self.max_recent_replies:
            self.recent_replies.pop(0)  # Remove oldest entry

    @ensure_naive_datetimes
    def reply_to_posts(self, posts: List[Dict[str, Any]], market_data: Dict[str, Any], max_replies: int = 5) -> int:
        """
        Generate and post replies to a list of posts with intelligent prioritization
    
        Args:
            posts: List of post data dictionaries
            market_data: Current market data dictionary
            max_replies: Maximum number of replies to post
        
        Returns:
            Number of successfully posted replies
        """
        if not posts:
            logger.logger.info("No posts to reply to")
            return 0
        
        # Limit to maximum replies
        posts_to_reply = posts[:max_replies]
        logger.logger.info(f"Attempting to reply to {len(posts_to_reply)} posts")
    
        # Track successful replies
        successful_replies = 0
    
        # Process each post
        for post in posts_to_reply:
            try:
                # Generate reply
                reply_text = self.generate_reply(post, market_data)
            
                if not reply_text:
                    logger.logger.warning(f"Failed to generate reply for post by {post.get('author_handle', 'unknown')}")
                    continue
                
                # Post the reply
                if self.post_reply(post, reply_text):
                    successful_replies += 1
                    logger.logger.info(f"Successfully replied to post by {post.get('author_handle', 'unknown')}")
                
                    # Capture metadata about the reply
                    metadata = self._capture_reply_metadata(post, reply_text)
                
                    # Update audience interaction history
                    self._update_audience_interaction_history(post, reply_text, metadata)
                
                    # Mark post as replied in database
                    if self.db and hasattr(self.db, 'mark_post_as_replied'):
                        post_id = post.get('post_id', '')
                        post_url = post.get('post_url', '')
                        self.db.mark_post_as_replied(post_id, post_url, reply_text)
                
                    # Allow delay between posts
                    if successful_replies < len(posts_to_reply):
                        delay = self._calculate_variable_delay(post)
                        time.sleep(delay)
                else:
                    logger.logger.warning(f"Failed to post reply to {post.get('author_handle', 'unknown')}")
                
            except Exception as e:
                logger.log_error("Reply To Post", str(e))
                continue
            
        logger.logger.info(f"Successfully posted {successful_replies} replies of {len(posts_to_reply)} attempted")
        return successful_replies

    @ensure_naive_datetimes
    def handle_tech_educational_posts(self, tech_posts: List[Dict[str, Any]], market_data: Optional[Dict[str, Any]] = None, max_replies: int = 3) -> int:
        """
        Handle tech-focused educational posts with special consideration
        
        Args:
            tech_posts: List of tech-related post dictionaries
            market_data: Market data for context (optional)
            max_replies: Maximum number of replies to post
            
        Returns:
            Number of successful replies
        """
        if not tech_posts:
            logger.logger.info("No tech posts to reply to")
            return 0
            
        # Filter for specifically educational content
        educational_posts = []
        for post in tech_posts:
            # Check if post has tech_analysis
            tech_analysis = post.get('tech_analysis', {})
            is_educational = tech_analysis.get('educational', {}).get('is_educational', False)
            
            # If it has explicit educational marking, use it
            if is_educational:
                educational_posts.append(post)
            # Otherwise do a simple text check
            elif 'text' in post and any(term in post['text'].lower() for term in 
                                     ['explain', 'learn', 'understand', 'guide', 'tutorial']):
                educational_posts.append(post)
                
        if not educational_posts:
            logger.logger.info("No educational tech posts found")
            return 0
            
        # Sort by educational value if available
        educational_posts.sort(
            key=lambda x: x.get('tech_analysis', {}).get('educational', {}).get('educational_value', 0.5),
            reverse=True
        )
        
        # Prioritize posts we haven't replied to recently
        prioritized_posts = []
        for post in educational_posts:
            post_id = post.get('post_id', '')
            if post_id and not self._already_replied(post_id):
                prioritized_posts.append(post)
                
        # Limit to max replies
        prioritized_posts = prioritized_posts[:max_replies]
        
        # Generate and post replies
        successful_replies = 0
        for post in prioritized_posts:
            try:
                # Generate tech-specific educational reply
                reply_text = self._generate_tech_reply(post, post.get('content_analysis', {}))
                
                if not reply_text:
                    logger.warning(f"Failed to generate tech reply for post by {post.get('author_handle', 'unknown')}")
                    continue
                    
                # Post the reply
                if self.post_reply(post, reply_text):
                    successful_replies += 1
                    
                    # Track this reply type in history
                    self.reply_type_history['tech_educational'] += 1
                    
                    # Allow delay between posts
                    if successful_replies < len(prioritized_posts):
                        delay = self._calculate_variable_delay(post)
                        time.sleep(delay)
                else:
                    logger.warning(f"Failed to post tech reply to {post.get('author_handle', 'unknown')}")
                    
            except Exception as e:
                logger.error(f"Tech Educational Reply: {str(e)}")
                continue
                
        return successful_replies

    @ensure_naive_datetimes
    def get_reply_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent replies
        
        Returns:
            Dictionary of reply statistics
        """
        stats = {
            'total_count': sum(self.reply_type_history.values()),
            'type_distribution': {k: v for k, v in self.reply_type_history.items()},
            'audience_stats': dict(self.audience_interaction['categories']),
            'recent_replies_count': len(self.recent_replies),
            'tech_stats': {
                'categories': dict(self.tech_interaction['categories']),
                'educational_count': self.tech_interaction['educational_posts']
            },
            'timestamp': strip_timezone(datetime.now())
        }
        
        # Calculate percentages if there are any replies
        if stats['total_count'] > 0:
            stats['type_percentages'] = {
                k: (v / stats['total_count']) * 100 
                for k, v in self.reply_type_history.items()
            }
            
        return stats