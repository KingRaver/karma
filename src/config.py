#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, List, TypedDict, Optional, Any
import os
from dotenv import load_dotenv
from utils.logger import logger
from database import CryptoDatabase
import threading

class CryptoInfo(TypedDict):
    id: str
    symbol: str
    name: str

class MarketAnalysisConfig(TypedDict):
    correlation_sensitivity: float
    volatility_threshold: float
    volume_significance: int
    historical_periods: List[int]

class TweetConstraints(TypedDict):
    MIN_LENGTH: int
    MAX_LENGTH: int
    HARD_STOP_LENGTH: int

class CoinGeckoParams(TypedDict):
    vs_currency: str
    ids: str
    order: str
    per_page: int
    page: int
    sparkline: bool
    price_change_percentage: str

class PredictionConfig(TypedDict):
    """Configuration for market predictions"""
    enabled_timeframes: List[str]
    confidence_threshold: float
    model_weights: Dict[str, float]
    prediction_frequency: int
    fomo_factor: float
    range_factor: float
    accuracy_threshold: float

class ProviderConfig(TypedDict):
    """LLM provider-specific configuration"""
    anthropic: Dict[str, str]
    openai: Dict[str, str]
    mistral: Dict[str, str]
    groq: Dict[str, str]

# New TypedDict definitions for Tech content
class TechCategory(TypedDict):
    enabled: bool
    priority: int
    min_interval_minutes: int
    keywords: List[str]

class TechContentConfig(TypedDict):
    """Configuration for technology content"""
    enabled: bool
    categories: Dict[str, TechCategory]
    integration_weight: float
    educational_boost: float
    post_frequency: int
    reply_frequency: int
    max_daily_tech_posts: int

class TechPromptConfig(TypedDict):
    """Configuration for technology content prompts"""
    educational_template: str
    integration_template: str
    reply_template: str
    complexity_levels: Dict[str, float]

# NEW: Multi-API Data Format Configuration TypedDicts
class DataSourceDetection(TypedDict):
    """Configuration for detecting data source format"""
    database: List[str]
    coingecko: List[str] 
    coinmarketcap: List[str]
    priority_order: List[str]

class TokenMappingManager:
    """
    Industry best practice implementation with proper dependency management
    """
    
    def __init__(self):
        # Initialize token mappings
        self.token_mappings = {
            'BTC': {
                'database_name': 'BITCOIN',
                'coingecko_id': 'bitcoin',
                'cmc_slug': 'bitcoin',
                'display_name': 'Bitcoin'
            },
            'ETH': {
                'database_name': 'ETHEREUM',
                'coingecko_id': 'ethereum',
                'cmc_slug': 'ethereum',
                'display_name': 'Ethereum'
            },
            'USDT': {
                'database_name': 'TETHER',
                'coingecko_id': 'tether',
                'cmc_slug': 'tether',
                'display_name': 'Tether'
            },
            'BNB': {
                'database_name': 'BINANCECOIN',
                'coingecko_id': 'binancecoin',
                'cmc_slug': 'bnb',
                'display_name': 'BNB'
            },
            'POL': {
                'database_name': 'POLYGON-ECOSYSTEM-TOKEN',
                'coingecko_id': 'polygon-ecosystem-token',
                'cmc_slug': 'polygon-ecosystem-token',
                'display_name': 'Polygon'
            },
            'SOL': {
                'database_name': 'SOLANA',
                'coingecko_id': 'solana',
                'cmc_slug': 'solana',
                'display_name': 'Solana'
            },
            'USDC': {
                'database_name': 'USD-COIN',
                'coingecko_id': 'usd-coin',
                'cmc_slug': 'usd-coin',
                'display_name': 'USD Coin'
            },
            'XRP': {
                'database_name': 'RIPPLE',
                'coingecko_id': 'ripple',
                'cmc_slug': 'xrp',
                'display_name': 'XRP'
            },
            'STETH': {
                'database_name': 'STAKED-ETHER',
                'coingecko_id': 'staked-ether',
                'cmc_slug': 'lido-staked-ether',
                'display_name': 'Lido Staked Ether'
            },
            'DOGE': {
                'database_name': 'DOGECOIN',
                'coingecko_id': 'dogecoin',
                'cmc_slug': 'dogecoin',
                'display_name': 'Dogecoin'
            },
            'ADA': {
                'database_name': 'CARDANO',
                'coingecko_id': 'cardano',
                'cmc_slug': 'cardano',
                'display_name': 'Cardano'
            },
            'TRX': {
                'database_name': 'TRON',
                'coingecko_id': 'tron',
                'cmc_slug': 'tron',
                'display_name': 'TRON'
            },
            'AVAX': {
                'database_name': 'AVALANCHE-2',
                'coingecko_id': 'avalanche-2',
                'cmc_slug': 'avalanche',
                'display_name': 'Avalanche'
            },
            'SHIB': {
                'database_name': 'SHIBA-INU',
                'coingecko_id': 'shiba-inu',
                'cmc_slug': 'shiba-inu',
                'display_name': 'Shiba Inu'
            },
            'WBTC': {
                'database_name': 'WRAPPED-BITCOIN',
                'coingecko_id': 'wrapped-bitcoin',
                'cmc_slug': 'wrapped-bitcoin',
                'display_name': 'Wrapped Bitcoin'
            },
            'TON': {
                'database_name': 'THE-OPEN-NETWORK',
                'coingecko_id': 'the-open-network',
                'cmc_slug': 'toncoin',
                'display_name': 'Toncoin'
            },
            'LINK': {
                'database_name': 'CHAINLINK',
                'coingecko_id': 'chainlink',
                'cmc_slug': 'chainlink',
                'display_name': 'Chainlink'
            },
            'WETH': {
                'database_name': 'WETH',
                'coingecko_id': 'weth',
                'cmc_slug': 'weth',
                'display_name': 'Wrapped Ether'
            },
            'DOT': {
                'database_name': 'POLKADOT',
                'coingecko_id': 'polkadot',
                'cmc_slug': 'polkadot-new',
                'display_name': 'Polkadot'
            },
            'BCH': {
                'database_name': 'BITCOIN-CASH',
                'coingecko_id': 'bitcoin-cash',
                'cmc_slug': 'bitcoin-cash',
                'display_name': 'Bitcoin Cash'
            },
            'MATIC': {
                'database_name': 'MATIC-NETWORK',
                'coingecko_id': 'matic-network',
                'cmc_slug': 'polygon',
                'display_name': 'Polygon'
            },
            'UNI': {
                'database_name': 'UNISWAP',
                'coingecko_id': 'uniswap',
                'cmc_slug': 'uniswap',
                'display_name': 'Uniswap'
            },
            'LTC': {
                'database_name': 'LITECOIN',
                'coingecko_id': 'litecoin',
                'cmc_slug': 'litecoin',
                'display_name': 'Litecoin'
            },
            'PEPE': {
                'database_name': 'PEPE',
                'coingecko_id': 'pepe',
                'cmc_slug': 'pepe',
                'display_name': 'Pepe'
            },
            'NEAR': {
                'database_name': 'NEAR',
                'coingecko_id': 'near',
                'cmc_slug': 'near-protocol',
                'display_name': 'NEAR Protocol'
            },
            'XLM': {
                'database_name': 'STELLAR',
                'coingecko_id': 'stellar',
                'cmc_slug': 'stellar',
                'display_name': 'Stellar'
            },
            'DAI': {
                'database_name': 'DAI',
                'coingecko_id': 'dai',
                'cmc_slug': 'multi-collateral-dai',
                'display_name': 'Dai'
            },
            'ETC': {
                'database_name': 'ETHEREUM-CLASSIC',
                'coingecko_id': 'ethereum-classic',
                'cmc_slug': 'ethereum-classic',
                'display_name': 'Ethereum Classic'
            },
            'BUSD': {
                'database_name': 'BINANCE-USD',
                'coingecko_id': 'binance-usd',
                'cmc_slug': 'binance-usd',
                'display_name': 'Binance USD'
            },
            'APT': {
                'database_name': 'APTOS',
                'coingecko_id': 'aptos',
                'cmc_slug': 'aptos',
                'display_name': 'Aptos'
            },
            'CRO': {
                'database_name': 'CRYPTO-COM-CHAIN',
                'coingecko_id': 'crypto-com-chain',
                'cmc_slug': 'cronos',
                'display_name': 'Cronos'
            },
            'XMR': {
                'database_name': 'MONERO',
                'coingecko_id': 'monero',
                'cmc_slug': 'monero',
                'display_name': 'Monero'
            },
            'ARB': {
                'database_name': 'ARBITRUM',
                'coingecko_id': 'arbitrum',
                'cmc_slug': 'arbitrum',
                'display_name': 'Arbitrum'
            },
            'OP': {
                'database_name': 'OPTIMISM',
                'coingecko_id': 'optimism',
                'cmc_slug': 'optimism-ethereum',
                'display_name': 'Optimism'
            },
            'ATOM': {
                'database_name': 'COSMOS',
                'coingecko_id': 'cosmos',
                'cmc_slug': 'cosmos',
                'display_name': 'Cosmos Hub'
            },
            'LDO': {
                'database_name': 'LIDO-DAO',
                'coingecko_id': 'lido-dao',
                'cmc_slug': 'lido-dao',
                'display_name': 'Lido DAO'
            },
            'HBAR': {
                'database_name': 'HEDERA-HASHGRAPH',
                'coingecko_id': 'hedera-hashgraph',
                'cmc_slug': 'hedera',
                'display_name': 'Hedera'
            },
            'FIL': {
                'database_name': 'FILECOIN',
                'coingecko_id': 'filecoin',
                'cmc_slug': 'filecoin',
                'display_name': 'Filecoin'
            },
            'IMX': {
                'database_name': 'IMMUTABLE-X',
                'coingecko_id': 'immutable-x',
                'cmc_slug': 'immutable-x',
                'display_name': 'Immutable'
            },
            'VET': {
                'database_name': 'VECHAIN',
                'coingecko_id': 'vechain',
                'cmc_slug': 'vechain',
                'display_name': 'VeChain'
            },
            'GRT': {
                'database_name': 'THE-GRAPH',
                'coingecko_id': 'the-graph',
                'cmc_slug': 'the-graph',
                'display_name': 'The Graph'
            },
            'MANA': {
                'database_name': 'DECENTRALAND',
                'coingecko_id': 'decentraland',
                'cmc_slug': 'decentraland',
                'display_name': 'Decentraland'
            },
            'SAND': {
                'database_name': 'THE-SANDBOX',
                'coingecko_id': 'the-sandbox',
                'cmc_slug': 'the-sandbox',
                'display_name': 'The Sandbox'
            },
            'APE': {
                'database_name': 'APECOIN',
                'coingecko_id': 'apecoin',
                'cmc_slug': 'apecoin-ape',
                'display_name': 'ApeCoin'
            },
            'ALGO': {
                'database_name': 'ALGORAND',
                'coingecko_id': 'algorand',
                'cmc_slug': 'algorand',
                'display_name': 'Algorand'
            },
            'AAVE': {
                'database_name': 'AAVE',
                'coingecko_id': 'aave',
                'cmc_slug': 'aave',
                'display_name': 'Aave'
            },
            'MKR': {
                'database_name': 'MAKER',
                'coingecko_id': 'maker',
                'cmc_slug': 'maker',
                'display_name': 'Maker'
            },
            'THETA': {
                'database_name': 'THETA-TOKEN',
                'coingecko_id': 'theta-token',
                'cmc_slug': 'theta',
                'display_name': 'Theta Network'
            },
            'QNT': {
                'database_name': 'QUANT-NETWORK',
                'coingecko_id': 'quant-network',
                'cmc_slug': 'quant',
                'display_name': 'Quant'
            },
            'ICP': {
                'database_name': 'INTERNET-COMPUTER',
                'coingecko_id': 'internet-computer',
                'cmc_slug': 'internet-computer',
                'display_name': 'Internet Computer'
            },
            'CRV': {
                'database_name': 'CURVE-DAO-TOKEN',
                'coingecko_id': 'curve-dao-token',
                'cmc_slug': 'curve-dao-token',
                'display_name': 'Curve DAO'
            },
            'FTM': {
                'database_name': 'FANTOM',
                'coingecko_id': 'fantom',
                'cmc_slug': 'fantom',
                'display_name': 'Fantom'
            },
            'FLOW': {
                'database_name': 'FLOW',
                'coingecko_id': 'flow',
                'cmc_slug': 'flow',
                'display_name': 'Flow'
            },
            'XTZ': {
                'database_name': 'TEZOS',
                'coingecko_id': 'tezos',
                'cmc_slug': 'tezos',
                'display_name': 'Tezos'
            },
            'EGLD': {
                'database_name': 'ELROND-EGULD',
                'coingecko_id': 'elrond-eguld',
                'cmc_slug': 'multiversx-egld',
                'display_name': 'MultiversX'
            },
            'SNX': {
                'database_name': 'HAVVEN',
                'coingecko_id': 'havven',
                'cmc_slug': 'synthetix-network-token',
                'display_name': 'Synthetix'
            },
            'EOS': {
                'database_name': 'EOS',
                'coingecko_id': 'eos',
                'cmc_slug': 'eos',
                'display_name': 'EOS'
            },
            'AXS': {
                'database_name': 'AXIE-INFINITY',
                'coingecko_id': 'axie-infinity',
                'cmc_slug': 'axie-infinity',
                'display_name': 'Axie Infinity'
            },
            'MINA': {
                'database_name': 'MINA-PROTOCOL',
                'coingecko_id': 'mina-protocol',
                'cmc_slug': 'mina',
                'display_name': 'Mina Protocol'
            },
            'RUNE': {
                'database_name': 'THORCHAIN',
                'coingecko_id': 'thorchain',
                'cmc_slug': 'thorchain',
                'display_name': 'THORChain'
            },
            'ZEC': {
                'database_name': 'ZCASH',
                'coingecko_id': 'zcash',
                'cmc_slug': 'zcash',
                'display_name': 'Zcash'
            },
            'NEO': {
                'database_name': 'NEO',
                'coingecko_id': 'neo',
                'cmc_slug': 'neo',
                'display_name': 'Neo'
            },
            'IOTA': {
                'database_name': 'IOTA',
                'coingecko_id': 'iota',
                'cmc_slug': 'iota',
                'display_name': 'IOTA'
            },
            'FET': {
                'database_name': 'FETCH-AI',
                'coingecko_id': 'fetch-ai',
                'cmc_slug': 'fetch-ai',
                'display_name': 'Fetch.ai'
            },
            'CAKE': {
                'database_name': 'PANCAKESWAP-TOKEN',
                'coingecko_id': 'pancakeswap-token',
                'cmc_slug': 'pancakeswap',
                'display_name': 'PancakeSwap'
            },
            'SUSHI': {
                'database_name': 'SUSHI',
                'coingecko_id': 'sushi',
                'cmc_slug': 'sushiswap',
                'display_name': 'SushiSwap'
            },
            'KCS': {
                'database_name': 'KUCOIN-SHARES',
                'coingecko_id': 'kucoin-shares',
                'cmc_slug': 'kucoin-token',
                'display_name': 'KuCoin'
            },
            'HT': {
                'database_name': 'HUOBI-TOKEN',
                'coingecko_id': 'huobi-token',
                'cmc_slug': 'huobi-token',
                'display_name': 'Huobi'
            },
            'COMP': {
                'database_name': 'COMPOUND-GOVERNANCE-TOKEN',
                'coingecko_id': 'compound-governance-token',
                'cmc_slug': 'compound',
                'display_name': 'Compound'
            },
            'DASH': {
                'database_name': 'DASH',
                'coingecko_id': 'dash',
                'cmc_slug': 'dash',
                'display_name': 'Dash'
            },
            'ZIL': {
                'database_name': 'ZILLIQA',
                'coingecko_id': 'zilliqa',
                'cmc_slug': 'zilliqa',
                'display_name': 'Zilliqa'
            },
            'ENJ': {
                'database_name': 'ENJINCOIN',
                'coingecko_id': 'enjincoin',
                'cmc_slug': 'enjin-coin',
                'display_name': 'Enjin Coin'
            },
            'GALA': {
                'database_name': 'GALA',
                'coingecko_id': 'gala',
                'cmc_slug': 'gala',
                'display_name': 'Gala'
            },
            'CHZ': {
                'database_name': 'CHILIZ',
                'coingecko_id': 'chiliz',
                'cmc_slug': 'chiliz',
                'display_name': 'Chiliz'
            },
            'LRC': {
                'database_name': 'LOOPRING',
                'coingecko_id': 'loopring',
                'cmc_slug': 'loopring',
                'display_name': 'Loopring'
            },
            'BAT': {
                'database_name': 'BASIC-ATTENTION-TOKEN',
                'coingecko_id': 'basic-attention-token',
                'cmc_slug': 'basic-attention-token',
                'display_name': 'Basic Attention'
            },
            'ENS': {
                'database_name': 'ETHEREUM-NAME-SERVICE',
                'coingecko_id': 'ethereum-name-service',
                'cmc_slug': 'ethereum-name-service',
                'display_name': 'Ethereum Name Service'
            },
            'GMT': {
                'database_name': 'STEPN',
                'coingecko_id': 'stepn',
                'cmc_slug': 'stepn',
                'display_name': 'STEPN'
            },
            'XEM': {
                'database_name': 'NEM',
                'coingecko_id': 'nem',
                'cmc_slug': 'nem',
                'display_name': 'NEM'
            },
            'HOT': {
                'database_name': 'HOLO',
                'coingecko_id': 'holo',
                'cmc_slug': 'holotoken',
                'display_name': 'Holo'
            },
            'ZRX': {
                'database_name': '0X',
                'coingecko_id': '0x',
                'cmc_slug': '0x',
                'display_name': '0x Protocol'
            },
            'ONT': {
                'database_name': 'ONTOLOGY',
                'coingecko_id': 'ontology',
                'cmc_slug': 'ontology',
                'display_name': 'Ontology'
            },
            'ICX': {
                'database_name': 'ICON',
                'coingecko_id': 'icon',
                'cmc_slug': 'icon',
                'display_name': 'ICON'
            },
            'QTUM': {
                'database_name': 'QTUM',
                'coingecko_id': 'qtum',
                'cmc_slug': 'qtum',
                'display_name': 'Qtum'
            },
            'OMG': {
                'database_name': 'OMISEGO',
                'coingecko_id': 'omisego',
                'cmc_slug': 'omg',
                'display_name': 'OMG Network'
            },
            'ZEN': {
                'database_name': 'HORIZEN',
                'coingecko_id': 'horizen',
                'cmc_slug': 'horizen',
                'display_name': 'Horizen'
            },
            'DGB': {
                'database_name': 'DIGIBYTE',
                'coingecko_id': 'digibyte',
                'cmc_slug': 'digibyte',
                'display_name': 'DigiByte'
            },
            'BNT': {
                'database_name': 'BANCOR',
                'coingecko_id': 'bancor',
                'cmc_slug': 'bancor',
                'display_name': 'Bancor Network'
            },
            'NANO': {
                'database_name': 'NANO',
                'coingecko_id': 'nano',
                'cmc_slug': 'nano',
                'display_name': 'Nano'
            },
            'REP': {
                'database_name': 'AUGUR',
                'coingecko_id': 'augur',
                'cmc_slug': 'augur',
                'display_name': 'Augur'
            },
            'SC': {
                'database_name': 'SIACOIN',
                'coingecko_id': 'siacoin',
                'cmc_slug': 'siacoin',
                'display_name': 'Siacoin'
            },
            'WAVES': {
                'database_name': 'WAVES',
                'coingecko_id': 'waves',
                'cmc_slug': 'waves',
                'display_name': 'Waves'
            },
            'XDC': {
                'database_name': 'XDC-NETWORK',
                'coingecko_id': 'xdce-crowd-sale',
                'cmc_slug': 'xdc-network',
                'display_name': 'XDC Network'
            },
            'KAVA': {
                'database_name': 'KAVA',
                'coingecko_id': 'kava',
                'cmc_slug': 'kava',
                'display_name': 'Kava'
            },
            'CELO': {
                'database_name': 'CELO',
                'coingecko_id': 'celo',
                'cmc_slug': 'celo',
                'display_name': 'Celo'
            },
            'ONE': {
                'database_name': 'HARMONY',
                'coingecko_id': 'harmony',
                'cmc_slug': 'harmony',
                'display_name': 'Harmony'
            },
            'IOTX': {
                'database_name': 'IOTEX',
                'coingecko_id': 'iotex',
                'cmc_slug': 'iotex',
                'display_name': 'IoTeX'
            },
            'ANKR': {
                'database_name': 'ANKR',
                'coingecko_id': 'ankr',
                'cmc_slug': 'ankr',
                'display_name': 'Ankr'
            },
            'AUDIO': {
                'database_name': 'AUDIUS',
                'coingecko_id': 'audius',
                'cmc_slug': 'audius',
                'display_name': 'Audius'
            },
            'STORJ': {
                'database_name': 'STORJ',
                'coingecko_id': 'storj',
                'cmc_slug': 'storj',
                'display_name': 'Storj'
            },
            'SRM': {
                'database_name': 'SERUM',
                'coingecko_id': 'serum',
                'cmc_slug': 'serum',
                'display_name': 'Serum'
            },
            'REEF': {
                'database_name': 'REEF',
                'coingecko_id': 'reef',
                'cmc_slug': 'reef',
                'display_name': 'Reef'
            },
            'SXP': {
                'database_name': 'SWIPE',
                'coingecko_id': 'swipe',
                'cmc_slug': 'solar',
                'display_name': 'Solar'
            },
            'YFI': {
                'database_name': 'YEARN-FINANCE',
                'coingecko_id': 'yearn-finance',
                'cmc_slug': 'yearn-finance',
                'display_name': 'yearn.finance'
            },
            'UMA': {
                'database_name': 'UMA',
                'coingecko_id': 'uma',
                'cmc_slug': 'uma',
                'display_name': 'UMA'
            },
            'BAND': {
                'database_name': 'BAND-PROTOCOL',
                'coingecko_id': 'band-protocol',
                'cmc_slug': 'band-protocol',
                'display_name': 'Band Protocol'
            },
            'ALPHA': {
                'database_name': 'ALPHA-FINANCE-LAB',
                'coingecko_id': 'alpha-finance-lab',
                'cmc_slug': 'alpha-finance-lab',
                'display_name': 'Alpha Venture DAO'
            },
            'OCEAN': {
                'database_name': 'OCEAN-PROTOCOL',
                'coingecko_id': 'ocean-protocol',
                'cmc_slug': 'ocean-protocol',
                'display_name': 'Ocean Protocol'
            },
            'RSR': {
                'database_name': 'RESERVE-RIGHTS',
                'coingecko_id': 'reserve-rights',
                'cmc_slug': 'reserve-rights',
                'display_name': 'Reserve Rights'
            },
            'INJ': {
                'database_name': 'INJECTIVE-PROTOCOL',
                'coingecko_id': 'injective-protocol',
                'cmc_slug': 'injective-protocol',
                'display_name': 'Injective'
            },
            'DYDX': {
                'database_name': 'DYDX',
                'coingecko_id': 'dydx',
                'cmc_slug': 'dydx',
                'display_name': 'dYdX'
            },
            'PERP': {
                'database_name': 'PERPETUAL-PROTOCOL',
                'coingecko_id': 'perpetual-protocol',
                'cmc_slug': 'perpetual-protocol',
                'display_name': 'Perpetual Protocol'
            },
            'REN': {
                'database_name': 'REPUBLIC-PROTOCOL',
                'coingecko_id': 'republic-protocol',
                'cmc_slug': 'ren',
                'display_name': 'Ren'
            },
            'LUNA': {
                'database_name': 'TERRA-LUNA-2',
                'coingecko_id': 'terra-luna-2',
                'cmc_slug': 'terra-luna',
                'display_name': 'Terra Luna Classic'
            },
            'LUNC': {
                'database_name': 'TERRA-LUNA',
                'coingecko_id': 'terra-luna',
                'cmc_slug': 'terra-luna-classic',
                'display_name': 'Terra Luna Classic'
            },
            'SUI': {
                'database_name': 'SUI',
                'coingecko_id': 'sui',
                'cmc_slug': 'sui',
                'display_name': 'Sui'
            },
            'BLUR': {
                'database_name': 'BLUR',
                'coingecko_id': 'blur',
                'cmc_slug': 'blur',
                'display_name': 'Blur'
            },
            'DODO': {
                'database_name': 'DODO',
                'coingecko_id': 'dodo',
                'cmc_slug': 'dodo',
                'display_name': 'DODO'
            },
            '1INCH': {
                'database_name': '1INCH',
                'coingecko_id': '1inch',
                'cmc_slug': '1inch',
                'display_name': '1inch Network'
            },
            'BAL': {
                'database_name': 'BALANCER',
                'coingecko_id': 'balancer',
                'cmc_slug': 'balancer',
                'display_name': 'Balancer'
            },
            'FTT': {
                'database_name': 'FTX-TOKEN',
                'coingecko_id': 'ftx-token',
                'cmc_slug': 'ftx-token',
                'display_name': 'FTX Token'
            },
            'LSK': {
                'database_name': 'LISK',
                'coingecko_id': 'lisk',
                'cmc_slug': 'lisk',
                'display_name': 'Lisk'
            },
            'DCR': {
                'database_name': 'DECRED',
                'coingecko_id': 'decred',
                'cmc_slug': 'decred',
                'display_name': 'Decred'
            },
            'IOST': {
                'database_name': 'IOST',
                'coingecko_id': 'iostoken',
                'cmc_slug': 'iost',
                'display_name': 'IOST'
            },
            'RVN': {
                'database_name': 'RAVENCOIN',
                'coingecko_id': 'ravencoin',
                'cmc_slug': 'ravencoin',
                'display_name': 'Ravencoin'
            },
            'STX': {
                'database_name': 'STACKS',
                'coingecko_id': 'stacks',
                'cmc_slug': 'stacks',
                'display_name': 'Stacks'
            },
            'KNC': {
                'database_name': 'KYBER-NETWORK-CRYSTAL',
                'coingecko_id': 'kyber-network-crystal',
                'cmc_slug': 'kyber-network-crystal',
                'display_name': 'Kyber Network Crystal'
            },
            'SKALE': {
                'database_name': 'SKALE-NETWORK',
                'coingecko_id': 'skale',
                'cmc_slug': 'skale-network',
                'display_name': 'SKALE Network'
            },
            'NMR': {
                'database_name': 'NUMERAIRE',
                'coingecko_id': 'numeraire',
                'cmc_slug': 'numeraire',
                'display_name': 'Numeraire'
            },
            'DENT': {
                'database_name': 'DENT',
                'coingecko_id': 'dent',
                'cmc_slug': 'dent',
                'display_name': 'Dent'
            },
            'WIN': {
                'database_name': 'WINKLINK',
                'coingecko_id': 'wink',
                'cmc_slug': 'winklink',
                'display_name': 'WINkLink'
            },
            'BTT': {
                'database_name': 'BITTORRENT-NEW',
                'coingecko_id': 'bittorrent',
                'cmc_slug': 'bittorrent-new',
                'display_name': 'BitTorrent'
            },
            'SUN': {
                'database_name': 'SUN-NEW',
                'coingecko_id': 'sun-token',
                'cmc_slug': 'sun-new',
                'display_name': 'SUN'
            },
            'JST': {
                'database_name': 'JUST',
                'coingecko_id': 'just',
                'cmc_slug': 'just',
                'display_name': 'JUST'
            },
            'BAKE': {
                'database_name': 'BAKERYTOKEN',
                'coingecko_id': 'bakerytoken',
                'cmc_slug': 'bakerytoken',
                'display_name': 'BakeryToken'
            },
            'AUTO': {
                'database_name': 'AUTO',
                'coingecko_id': 'auto',
                'cmc_slug': 'auto',
                'display_name': 'Auto'
            },
            'BURGER': {
                'database_name': 'BURGER-SWAP',
                'coingecko_id': 'burger-swap',
                'cmc_slug': 'burger-swap',
                'display_name': 'Burger Swap'
            },
            'XVS': {
                'database_name': 'VENUS',
                'coingecko_id': 'venus',
                'cmc_slug': 'venus',
                'display_name': 'Venus'
            },
            'HARD': {
                'database_name': 'KAVA-LEND',
                'coingecko_id': 'kava-lend',
                'cmc_slug': 'hard-protocol',
                'display_name': 'Hard Protocol'
            },
            'RLC': {
                'database_name': 'IEXEC-RLC',
                'coingecko_id': 'iexec-rlc',
                'cmc_slug': 'iexec-rlc',
                'display_name': 'iExec RLC'
            },
            'CTSI': {
                'database_name': 'CARTESI',
                'coingecko_id': 'cartesi',
                'cmc_slug': 'cartesi',
                'display_name': 'Cartesi'
            },
            'DATA': {
                'database_name': 'STREAMR',
                'coingecko_id': 'streamr',
                'cmc_slug': 'streamr',
                'display_name': 'Streamr'
            },
            'MLN': {
                'database_name': 'MELON',
                'coingecko_id': 'melon',
                'cmc_slug': 'enzyme',
                'display_name': 'Enzyme'
            },
            'POLY': {
                'database_name': 'POLYMATH',
                'coingecko_id': 'polymath',
                'cmc_slug': 'polymath',
                'display_name': 'Polymath'
            },
            'LPT': {
                'database_name': 'LIVEPEER',
                'coingecko_id': 'livepeer',
                'cmc_slug': 'livepeer',
                'display_name': 'Livepeer'
            },
            'ARK': {
                'database_name': 'ARK',
                'coingecko_id': 'ark',
                'cmc_slug': 'ark',
                'display_name': 'Ark'
            },
            'KMD': {
                'database_name': 'KOMODO',
                'coingecko_id': 'komodo',
                'cmc_slug': 'komodo',
                'display_name': 'Komodo'
            },
            'STRAX': {
                'database_name': 'STRAX',
                'coingecko_id': 'strax',
                'cmc_slug': 'strax',
                'display_name': 'Strax'
            },
            'STEEM': {
                'database_name': 'STEEM',
                'coingecko_id': 'steem',
                'cmc_slug': 'steem',
                'display_name': 'Steem'
            },
            'HIVE': {
                'database_name': 'HIVE',
                'coingecko_id': 'hive',
                'cmc_slug': 'hive',
                'display_name': 'Hive'
            },
            'ARDR': {
                'database_name': 'ARDOR',
                'coingecko_id': 'ardor',
                'cmc_slug': 'ardor',
                'display_name': 'Ardor'
            },
            'NXT': {
                'database_name': 'NXT',
                'coingecko_id': 'nxt',
                'cmc_slug': 'nxt',
                'display_name': 'Nxt'
            },
            'BTS': {
                'database_name': 'BITSHARES',
                'coingecko_id': 'bitshares',
                'cmc_slug': 'bitshares',
                'display_name': 'BitShares'
            },
            'MAID': {
                'database_name': 'MAIDSAFECOIN',
                'coingecko_id': 'maidsafecoin',
                'cmc_slug': 'maidsafecoin',
                'display_name': 'MaidSafeCoin'
            },
            'FCT': {
                'database_name': 'FACTOM',
                'coingecko_id': 'factom',
                'cmc_slug': 'factom',
                'display_name': 'Factom'
            },
            'MONA': {
                'database_name': 'MONACOIN',
                'coingecko_id': 'monacoin',
                'cmc_slug': 'monacoin',
                'display_name': 'MonaCoin'
            },
            'SYS': {
                'database_name': 'SYSCOIN',
                'coingecko_id': 'syscoin',
                'cmc_slug': 'syscoin',
                'display_name': 'Syscoin'
            },
            'VTC': {
                'database_name': 'VERTCOIN',
                'coingecko_id': 'vertcoin',
                'cmc_slug': 'vertcoin',
                'display_name': 'Vertcoin'
            },
            'GAS': {
                'database_name': 'GAS',
                'coingecko_id': 'gas',
                'cmc_slug': 'gas',
                'display_name': 'Gas'
            },
            'WAN': {
                'database_name': 'WANCHAIN',
                'coingecko_id': 'wanchain',
                'cmc_slug': 'wanchain',
                'display_name': 'Wanchain'
            },
            'WICC': {
                'database_name': 'WAYKICHAIN',
                'coingecko_id': 'waykichain',
                'cmc_slug': 'waykichain',
                'display_name': 'WaykiChain'
            },
            'ELA': {
                'database_name': 'ELASTOS',
                'coingecko_id': 'elastos',
                'cmc_slug': 'elastos',
                'display_name': 'Elastos'
            },
            'BEAM': {
                'database_name': 'BEAM',
                'coingecko_id': 'beam',
                'cmc_slug': 'beam',
                'display_name': 'Beam'
            },
            'GRIN': {
                'database_name': 'GRIN',
                'coingecko_id': 'grin',
                'cmc_slug': 'grin',
                'display_name': 'Grin'
            },
            'ERG': {
                'database_name': 'ERGO',
                'coingecko_id': 'ergo',
                'cmc_slug': 'ergo',
                'display_name': 'Ergo'
            },
            'FLUX': {
                'database_name': 'ZELCASH',
                'coingecko_id': 'zelcash',
                'cmc_slug': 'flux',
                'display_name': 'Flux'
            },
            'RDD': {
                'database_name': 'REDDCOIN',
                'coingecko_id': 'reddcoin',
                'cmc_slug': 'reddcoin',
                'display_name': 'ReddCoin'
            },
            'PPC': {
                'database_name': 'PEERCOIN',
                'coingecko_id': 'peercoin',
                'cmc_slug': 'peercoin',
                'display_name': 'Peercoin'
            },
            'NMC': {
                'database_name': 'NAMECOIN',
                'coingecko_id': 'namecoin',
                'cmc_slug': 'namecoin',
                'display_name': 'Namecoin'
            },
            'DOGE2': {
                'database_name': 'DOGECHAIN',
                'coingecko_id': 'dogechain',
                'cmc_slug': 'dogechain',
                'display_name': 'Dogechain'
            },
            'TFUEL': {
                'database_name': 'THETA-FUEL',
                'coingecko_id': 'theta-fuel',
                'cmc_slug': 'theta-fuel',
                'display_name': 'Theta Fuel'
            },
            'WOO': {
                'database_name': 'WOO-NETWORK',
                'coingecko_id': 'woo-network',
                'cmc_slug': 'woo-network',
                'display_name': 'WOO Network'
            },
            'RNDR': {
                'database_name': 'RENDER-TOKEN',
                'coingecko_id': 'render-token',
                'cmc_slug': 'render',
                'display_name': 'Render'
            },
            'LOKA': {
                'database_name': 'LEAGUE-OF-KINGDOMS',
                'coingecko_id': 'league-of-kingdoms',
                'cmc_slug': 'league-of-kingdoms',
                'display_name': 'League of Kingdoms'
            },
            'JASMY': {
                'database_name': 'JASMYCOIN',
                'coingecko_id': 'jasmycoin',
                'cmc_slug': 'jasmycoin',
                'display_name': 'JasmyCoin'
            },
            'RAY': {
                'database_name': 'RAYDIUM',
                'coingecko_id': 'raydium',
                'cmc_slug': 'raydium',
                'display_name': 'Raydium'
            },
            'SANTOS': {
                'database_name': 'SANTOS-FC-FAN-TOKEN',
                'coingecko_id': 'santos-fc-fan-token',
                'cmc_slug': 'santos-fc-fan-token',
                'display_name': 'Santos FC Fan Token'
            },
            'PEOPLE': {
                'database_name': 'CONSTITUTIONDAO',
                'coingecko_id': 'constitutiondao',
                'cmc_slug': 'constitutiondao',
                'display_name': 'ConstitutionDAO'
            },
            'C98': {
                'database_name': 'COIN98',
                'coingecko_id': 'coin98',
                'cmc_slug': 'coin98',
                'display_name': 'Coin98'
            },
            'BICO': {
                'database_name': 'BICONOMY',
                'coingecko_id': 'biconomy',
                'cmc_slug': 'biconomy',
                'display_name': 'Biconomy'
            },
            'LOOKS': {
                'database_name': 'LOOKSRARE',
                'coingecko_id': 'looksrare',
                'cmc_slug': 'looksrare',
                'display_name': 'LooksRare'
            },
            'FLM': {
                'database_name': 'FLAMINGO-FINANCE',
                'coingecko_id': 'flamingo-finance',
                'cmc_slug': 'flamingo-finance',
                'display_name': 'Flamingo Finance'
            },
            'ILV': {
                'database_name': 'ILLUVIUM',
                'coingecko_id': 'illuvium',
                'cmc_slug': 'illuvium',
                'display_name': 'Illuvium'
            },
            'SLP': {
                'database_name': 'SMOOTH-LOVE-POTION',
                'coingecko_id': 'smooth-love-potion',
                'cmc_slug': 'smooth-love-potion',
                'display_name': 'Smooth Love Potion'
            },
            'GODS': {
                'database_name': 'GODS-UNCHAINED',
                'coingecko_id': 'gods-unchained',
                'cmc_slug': 'gods-unchained',
                'display_name': 'Gods Unchained'
            },
            'ALICE': {
                'database_name': 'MY-NEIGHBOR-ALICE',
                'coingecko_id': 'my-neighbor-alice',
                'cmc_slug': 'my-neighbor-alice',
                'display_name': 'My Neighbor Alice'
            },
            'TLM': {
                'database_name': 'ALIEN-WORLDS',
                'coingecko_id': 'alien-worlds',
                'cmc_slug': 'alien-worlds',
                'display_name': 'Alien Worlds'
            },
            'SUPER': {
                'database_name': 'SUPERFARM',
                'coingecko_id': 'superfarm',
                'cmc_slug': 'superfarm',
                'display_name': 'SuperFarm'
            },
            'RARE': {
                'database_name': 'SUPERRARE',
                'coingecko_id': 'superrare',
                'cmc_slug': 'superrare',
                'display_name': 'SuperRare'
            },
            'PSG': {
                'database_name': 'PARIS-SAINT-GERMAIN-FAN-TOKEN',
                'coingecko_id': 'paris-saint-germain-fan-token',
                'cmc_slug': 'paris-saint-germain-fan-token',
                'display_name': 'Paris Saint-Germain Fan Token'
            },
            'JUV': {
                'database_name': 'JUVENTUS-FAN-TOKEN',
                'coingecko_id': 'juventus-fan-token',
                'cmc_slug': 'juventus-fan-token',
                'display_name': 'Juventus Fan Token'
            },
            'BAR': {
                'database_name': 'FC-BARCELONA-FAN-TOKEN',
                'coingecko_id': 'fc-barcelona-fan-token',
                'cmc_slug': 'fc-barcelona-fan-token',
                'display_name': 'FC Barcelona Fan Token'
            },
            'ATM': {
                'database_name': 'ATLETICO-DE-MADRID-FAN-TOKEN',
                'coingecko_id': 'atletico-de-madrid',
                'cmc_slug': 'atletico-madrid',
                'display_name': 'Atletico Madrid Fan Token'
            },
            'ASR': {
                'database_name': 'AS-ROMA-FAN-TOKEN',
                'coingecko_id': 'as-roma-fan-token',
                'cmc_slug': 'as-roma-fan-token',
                'display_name': 'AS Roma Fan Token'
            },
            'ACM': {
                'database_name': 'AC-MILAN-FAN-TOKEN',
                'coingecko_id': 'ac-milan-fan-token',
                'cmc_slug': 'ac-milan-fan-token',
                'display_name': 'AC Milan Fan Token'
            },
            'CITY': {
                'database_name': 'MANCHESTER-CITY-FAN-TOKEN',
                'coingecko_id': 'manchester-city-fan-token',
                'cmc_slug': 'manchester-city-fan-token',
                'display_name': 'Manchester City Fan Token'
            },
            'LAZIO': {
                'database_name': 'LAZIO-FAN-TOKEN',
                'coingecko_id': 'lazio-fan-token',
                'cmc_slug': 'lazio-fan-token',
                'display_name': 'Lazio Fan Token'
            },
            'INTER': {
                'database_name': 'INTER-MILAN-FAN-TOKEN',
                'coingecko_id': 'inter-milan-fan-token',
                'cmc_slug': 'inter-milan-fan-token',
                'display_name': 'Inter Milan Fan Token'
            },
            'GAL': {
                'database_name': 'GALATASARAY-FAN-TOKEN',
                'coingecko_id': 'galatasaray-fan-token',
                'cmc_slug': 'galatasaray-fan-token',
                'display_name': 'Galatasaray Fan Token'
            },
            'ALPINE': {
                'database_name': 'ALPINE-F1-TEAM-FAN-TOKEN',
                'coingecko_id': 'alpine-f1-team-fan-token',
                'cmc_slug': 'alpine-f1-team-fan-token',
                'display_name': 'Alpine F1 Team Fan Token'
            },
            'SANTOS2': {
                'database_name': 'SANTOS-FC-FAN-TOKEN-2',
                'coingecko_id': 'santos-fc-fan-token-2',
                'cmc_slug': 'santos-fc-fan-token-2',
                'display_name': 'Santos FC Fan Token 2'
            },
            'VRA': {
                'database_name': 'VERASITY',
                'coingecko_id': 'verasity',
                'cmc_slug': 'verasity',
                'display_name': 'Verasity'
            },
            'CEEK': {
                'database_name': 'CEEK-VR',
                'coingecko_id': 'ceek',
                'cmc_slug': 'ceek-smart-vr-token',
                'display_name': 'CEEK VR'
            },
            'HIFI': {
                'database_name': 'HIFI-FINANCE',
                'coingecko_id': 'hifi-finance',
                'cmc_slug': 'hifi-finance',
                'display_name': 'Hifi Finance'
            },
            'TRIBE': {
                'database_name': 'TRIBE',
                'coingecko_id': 'tribe-2',
                'cmc_slug': 'tribe',
                'display_name': 'Tribe'
            },
            'FEI': {
                'database_name': 'FEI-USD',
                'coingecko_id': 'fei-usd',
                'cmc_slug': 'fei-protocol',
                'display_name': 'Fei USD'
            },
            'TORN': {
                'database_name': 'TORNADO-CASH',
                'coingecko_id': 'tornado-cash',
                'cmc_slug': 'tornado-cash',
                'display_name': 'Tornado Cash'
            },
            'BADGER': {
                'database_name': 'BADGER-DAO',
                'coingecko_id': 'badger-dao',
                'cmc_slug': 'badger-dao',
                'display_name': 'Badger DAO'
            },
            'DIA': {
                'database_name': 'DIA-DATA',
                'coingecko_id': 'dia-data',
                'cmc_slug': 'dia',
                'display_name': 'DIA'
            },
            'MIR': {
                'database_name': 'MIRROR-PROTOCOL',
                'coingecko_id': 'mirror-protocol',
                'cmc_slug': 'mirror-protocol',
                'display_name': 'Mirror Protocol'
            },
            'ANC': {
                'database_name': 'ANCHOR-PROTOCOL',
                'coingecko_id': 'anchor-protocol',
                'cmc_slug': 'anchorusd',
                'display_name': 'Anchor Protocol'
            },
            'CKB': {
                'database_name': 'NERVOS-NETWORK',
                'coingecko_id': 'nervos-network',
                'cmc_slug': 'nervos-network',
                'display_name': 'Nervos Network'
            },
            'CELR': {
                'database_name': 'CELER-NETWORK',
                'coingecko_id': 'celer-network',
                'cmc_slug': 'celer-network',
                'display_name': 'Celer Network'
            },
            'KDA': {
                'database_name': 'KADENA',
                'coingecko_id': 'kadena',
                'cmc_slug': 'kadena',
                'display_name': 'Kadena'
            },
            'ROSE': {
                'database_name': 'OASIS-NETWORK',
                'coingecko_id': 'oasis-network',
                'cmc_slug': 'oasis-network',
                'display_name': 'Oasis Network'
            },
            'AR': {
                'database_name': 'ARWEAVE',
                'coingecko_id': 'arweave',
                'cmc_slug': 'arweave',
                'display_name': 'Arweave'
            },
            'SAITAMA': {
                'database_name': 'SAITAMAINU',
                'coingecko_id': 'saitama-inu',
                'cmc_slug': 'saitama-inu',
                'display_name': 'Saitama'
            },
            'FLOKI': {
                'database_name': 'FLOKI',
                'coingecko_id': 'floki',
                'cmc_slug': 'floki',
                'display_name': 'FLOKI'
            },
            'SAFEMOON': {
                'database_name': 'SAFEMOON',
                'coingecko_id': 'safemoon',
                'cmc_slug': 'safemoon',
                'display_name': 'SafeMoon'
            },
            'BABYDOGE': {
                'database_name': 'BABY-DOGE-COIN',
                'coingecko_id': 'baby-doge-coin',
                'cmc_slug': 'baby-doge-coin',
                'display_name': 'Baby Doge Coin'
            },
            'ELON': {
                'database_name': 'DOGELON-MARS',
                'coingecko_id': 'dogelon-mars',
                'cmc_slug': 'dogelon',
                'display_name': 'Dogelon Mars'
            },
            'KISHU': {
                'database_name': 'KISHU-INU',
                'coingecko_id': 'kishu-inu',
                'cmc_slug': 'kishu-inu',
                'display_name': 'Kishu Inu'
            },
            'AKITA': {
                'database_name': 'AKITA-INU',
                'coingecko_id': 'akita-inu',
                'cmc_slug': 'akita-inu',
                'display_name': 'Akita Inu'
            },
            'HOGE': {
                'database_name': 'HOGE-FINANCE',
                'coingecko_id': 'hoge-finance',
                'cmc_slug': 'hoge-finance',
                'display_name': 'Hoge Finance'
            },
            'SPELL': {
                'database_name': 'SPELL-TOKEN',
                'coingecko_id': 'spell-token',
                'cmc_slug': 'spell-token',
                'display_name': 'Spell Token'
            },
            'CVX': {
                'database_name': 'CONVEX-FINANCE',
                'coingecko_id': 'convex-finance',
                'cmc_slug': 'convex-finance',
                'display_name': 'Convex Finance'
            },
            'FXS': {
                'database_name': 'FRAX-SHARE',
                'coingecko_id': 'frax-share',
                'cmc_slug': 'frax-share',
                'display_name': 'Frax Share'
            },
            'FRAX': {
                'database_name': 'FRAX',
                'coingecko_id': 'frax',
                'cmc_slug': 'frax',
                'display_name': 'Frax'
            },
            'ALCX': {
                'database_name': 'ALCHEMIX',
                'coingecko_id': 'alchemix',
                'cmc_slug': 'alchemix',
                'display_name': 'Alchemix'
            },
            'OHM': {
                'database_name': 'OLYMPUS',
                'coingecko_id': 'olympus',
                'cmc_slug': 'olympus',
                'display_name': 'Olympus'
            },
            'KLIMA': {
                'database_name': 'KLIMA-DAO',
                'coingecko_id': 'klimadao',
                'cmc_slug': 'klimadao',
                'display_name': 'KlimaDAO'
            },
            'TIME': {
                'database_name': 'WONDERLAND',
                'coingecko_id': 'wonderland',
                'cmc_slug': 'wonderland',
                'display_name': 'Wonderland'
            },
            'MEMO': {
                'database_name': 'WONDERLAND-MEMO',
                'coingecko_id': 'wonderland-memo',
                'cmc_slug': 'wonderland-memo',
                'display_name': 'Wonderland MEMO'
            },
            'GOHM': {
                'database_name': 'GOVERNANCE-OHM',
                'coingecko_id': 'governance-ohm',
                'cmc_slug': 'governance-ohm',
                'display_name': 'Governance OHM'
            },
            'BOND': {
                'database_name': 'BARNBRIDGE',
                'coingecko_id': 'barnbridge',
                'cmc_slug': 'barnbridge',
                'display_name': 'BarnBridge'
            },
            'CREAM': {
                'database_name': 'CREAM-2',
                'coingecko_id': 'cream-2',
                'cmc_slug': 'cream-finance',
                'display_name': 'Cream Finance'
            },
            'PICKLE': {
                'database_name': 'PICKLE-FINANCE',
                'coingecko_id': 'pickle-finance',
                'cmc_slug': 'pickle-finance',
                'display_name': 'Pickle Finance'
            },
            'BOBA': {
                'database_name': 'BOBA-NETWORK',
                'coingecko_id': 'boba-network',
                'cmc_slug': 'boba-network',
                'display_name': 'Boba Network'
            },
            'METIS': {
                'database_name': 'METIS-TOKEN',
                'coingecko_id': 'metis-token',
                'cmc_slug': 'metis',
                'display_name': 'Metis'
            },
            'MOVR': {
                'database_name': 'MOONRIVER',
                'coingecko_id': 'moonriver',
                'cmc_slug': 'moonriver',
                'display_name': 'Moonriver'
            },
            'GLMR': {
                'database_name': 'MOONBEAM',
                'coingecko_id': 'moonbeam',
                'cmc_slug': 'moonbeam',
                'display_name': 'Moonbeam'
            },
            'ASTR': {
                'database_name': 'ASTAR',
                'coingecko_id': 'astar',
                'cmc_slug': 'astar',
                'display_name': 'Astar'
            },
            'SDN': {
                'database_name': 'SHIDEN',
                'coingecko_id': 'shiden',
                'cmc_slug': 'shiden',
                'display_name': 'Shiden Network'
            },
            'ACALA': {
                'database_name': 'ACALA-TOKEN',
                'coingecko_id': 'acala',
                'cmc_slug': 'acala',
                'display_name': 'Acala Token'
            },
            'KARURA': {
                'database_name': 'KARURA',
                'coingecko_id': 'karura',
                'cmc_slug': 'karura',
                'display_name': 'Karura'
            },
            'PHB': {
                'database_name': 'PHOENIX-GLOBAL',
                'coingecko_id': 'phoenix-global',
                'cmc_slug': 'phoenix-global-new',
                'display_name': 'Phoenix Global'
            },
            'FOR': {
                'database_name': 'FORTUBE',
                'coingecko_id': 'for',
                'cmc_slug': 'fortube',
                'display_name': 'ForTube'
            },
            'AUCTION': {
                'database_name': 'BOUNCE-TOKEN',
                'coingecko_id': 'bounce-token',
                'cmc_slug': 'bounce',
                'display_name': 'Bounce Token'
            },
            'SFP': {
                'database_name': 'SAFEPAL',
                'coingecko_id': 'safepal',
                'cmc_slug': 'safepal',
                'display_name': 'SafePal'
            },
            'PUNDIX': {
                'database_name': 'PUNDI-X-NEW',
                'coingecko_id': 'pundi-x-2',
                'cmc_slug': 'pundi-x-new',
                'display_name': 'Pundi X'
            },
            'UTK': {
                'database_name': 'UTRUST',
                'coingecko_id': 'utrust',
                'cmc_slug': 'utrust',
                'display_name': 'xMoney'
            },
            'TKO': {
                'database_name': 'TOKOCRYPTO',
                'coingecko_id': 'tokocrypto',
                'cmc_slug': 'tokocrypto',
                'display_name': 'Tokocrypto'
            },
            'PROS': {
                'database_name': 'PROSPER',
                'coingecko_id': 'prosper',
                'cmc_slug': 'prosper',
                'display_name': 'Prosper'
            },
            'POLS': {
                'database_name': 'POLKASTARTER',
                'coingecko_id': 'polkastarter',
                'cmc_slug': 'polkastarter',
                'display_name': 'Polkastarter'
            },
            'DF': {
                'database_name': 'DFORCE-TOKEN',
                'coingecko_id': 'dforce-token',
                'cmc_slug': 'dforce',
                'display_name': 'dForce'
            },
            'EPS': {
                'database_name': 'ELLIPSIS',
                'coingecko_id': 'ellipsis',
                'cmc_slug': 'ellipsis',
                'display_name': 'Ellipsis'
            },
            'CHESS': {
                'database_name': 'TRANCHESS',
                'coingecko_id': 'tranchess',
                'cmc_slug': 'tranchess',
                'display_name': 'Tranchess'
            },
            'FIDA': {
                'database_name': 'BONFIDA',
                'coingecko_id': 'bonfida',
                'cmc_slug': 'bonfida',
                'display_name': 'Bonfida'
            },
            'OXY': {
                'database_name': 'OXYGEN',
                'coingecko_id': 'oxygen',
                'cmc_slug': 'oxygen',
                'display_name': 'Oxygen'
            },
            'MAPS': {
                'database_name': 'MAPS',
                'coingecko_id': 'maps',
                'cmc_slug': 'maps',
                'display_name': 'MAPS'
            },
            'STEP': {
                'database_name': 'STEP-FINANCE',
                'coingecko_id': 'step-finance',
                'cmc_slug': 'step-finance',
                'display_name': 'Step Finance'
            },
            'MEDIA': {
                'database_name': 'MEDIA-NETWORK',
                'coingecko_id': 'media-network',
                'cmc_slug': 'media-network',
                'display_name': 'Media Network'
            },
            'ROPE': {
                'database_name': 'ROPE-TOKEN',
                'coingecko_id': 'rope-token',
                'cmc_slug': 'rope',
                'display_name': 'ROPE'
            },
            'COPE': {
                'database_name': 'COPE',
                'coingecko_id': 'cope',
                'cmc_slug': 'cope',
                'display_name': 'COPE'
            },
            'STAR': {
                'database_name': 'STARNAME',
                'coingecko_id': 'starname',
                'cmc_slug': 'starname',
                'display_name': 'Starname'
            },
            'PORT': {
                'database_name': 'PORTO',
                'coingecko_id': 'fc-porto',
                'cmc_slug': 'fc-porto',
                'display_name': 'FC Porto Fan Token'
            },
            'LEET': {
                'database_name': 'LEETSWAP',
                'coingecko_id': 'leetswap',
                'cmc_slug': 'leetswap',
                'display_name': 'Leetswap'
            },
            'COCOS': {
                'database_name': 'COCOS-BCX',
                'coingecko_id': 'cocos-bcx',
                'cmc_slug': 'cocos-bcx',
                'display_name': 'Cocos-BCX'
            },
            'NKN': {
                'database_name': 'NKN',
                'coingecko_id': 'nkn',
                'cmc_slug': 'nkn',
                'display_name': 'NKN'
            },
            'LTO': {
                'database_name': 'LTO-NETWORK',
                'coingecko_id': 'lto-network',
                'cmc_slug': 'lto-network',
                'display_name': 'LTO Network'
            },
            'KEY': {
                'database_name': 'SELFKEY',
                'coingecko_id': 'selfkey',
                'cmc_slug': 'selfkey',
                'display_name': 'SelfKey'
            },
            'ONG': {
                'database_name': 'ONTOLOGY-GAS',
                'coingecko_id': 'ontology-gas',
                'cmc_slug': 'ontology-gas',
                'display_name': 'Ontology Gas'
            },
            'VTHO': {
                'database_name': 'VETHOR-TOKEN',
                'coingecko_id': 'vethor-token',
                'cmc_slug': 'vethor-token',
                'display_name': 'VeThor Token'
            },
            'FUN': {
                'database_name': 'FUNFAIR',
                'coingecko_id': 'funfair',
                'cmc_slug': 'funfair',
                'display_name': 'FunFair'
            },
            'DUSK': {
                'database_name': 'DUSK-NETWORK',
                'coingecko_id': 'dusk-network',
                'cmc_slug': 'dusk-network',
                'display_name': 'Dusk Network'
            },
            'IRIS': {
                'database_name': 'IRISNET',
                'coingecko_id': 'iris-network',
                'cmc_slug': 'irisnet',
                'display_name': 'IRISnet'
            },
            'BLZ': {
                'database_name': 'BLUZELLE',
                'coingecko_id': 'bluzelle',
                'cmc_slug': 'bluzelle',
                'display_name': 'Bluzelle'
            },
            'KLAY': {
                'database_name': 'KLAYTN',
                'coingecko_id': 'klay-token',
                'cmc_slug': 'klaytn',
                'display_name': 'Klaytn'
            },
            'CTC': {
                'database_name': 'CREDITCOIN',
                'coingecko_id': 'creditcoin-2',
                'cmc_slug': 'creditcoin',
                'display_name': 'Creditcoin'
            },
            'WAXP': {
                'database_name': 'WAX',
                'coingecko_id': 'wax',
                'cmc_slug': 'wax',
                'display_name': 'WAX'
            },
            'HNT': {
                'database_name': 'HELIUM',
                'coingecko_id': 'helium',
                'cmc_slug': 'helium',
                'display_name': 'Helium'
            },
            'MOBILE': {
                'database_name': 'HELIUM-MOBILE',
                'coingecko_id': 'helium-mobile',
                'cmc_slug': 'helium-mobile',
                'display_name': 'Helium Mobile'
            },
            'IOT': {
                'database_name': 'HELIUM-IOT',
                'coingecko_id': 'helium-iot',
                'cmc_slug': 'helium-iot',
                'display_name': 'Helium IOT'
            },
            'MIOTA': {
                'database_name': 'IOTA-2',
                'coingecko_id': 'miota',
                'cmc_slug': 'miota',
                'display_name': 'MIOTA'
            },
            'SMR': {
                'database_name': 'SHIMMER',
                'coingecko_id': 'shimmer',
                'cmc_slug': 'shimmer',
                'display_name': 'Shimmer'
            },
            'AION': {
                'database_name': 'AION',
                'coingecko_id': 'aion',
                'cmc_slug': 'aion',
                'display_name': 'Aion'
            },
            # Add more tokens to reach 150+ as needed...
        }
        self._create_reverse_mappings()
            
        # Lazy initialization for database - don't create during __init__
        self._database_instance = None
        self._database_lock = threading.Lock()

    def _get_database_instance(self):
        """
        Lazy database initialization with thread-safe singleton pattern
        Industry best practice: Don't initialize heavy resources during object creation
        Uses DatabaseManager to ensure only one database instance across entire application
        """
        if self._database_instance is None:
            with self._database_lock:
                # Double-check locking pattern
                if self._database_instance is None:
                    try:
                        from database import get_database_instance
                        # Use DatabaseManager singleton to get database instance
                        self._database_instance = get_database_instance()
                        logger.logger.debug("Database instance obtained via DatabaseManager singleton")
                    except Exception as e:
                        logger.logger.warning(f"Database lazy initialization failed: {e}")
                        # Return None to trigger fallback behavior
                        return None
        
        return self._database_instance 
    
    def _create_reverse_mappings(self):
        """Create reverse mapping dictionaries for fast lookups"""
        self.database_to_symbol = {}
        self.coingecko_to_symbol = {}
        self.cmc_to_symbol = {}
        
        for symbol, mapping in self.token_mappings.items():
            # Database name to symbol
            if mapping['database_name']:
                self.database_to_symbol[mapping['database_name']] = symbol
            
            # CoinGecko ID to symbol  
            if mapping['coingecko_id']:
                self.coingecko_to_symbol[mapping['coingecko_id']] = symbol
                
            # CoinMarketCap slug to symbol
            if mapping['cmc_slug']:
                self.cmc_to_symbol[mapping['cmc_slug']] = symbol
    
    # ========================================================================
    # CORE MAPPING METHODS
    # ========================================================================
    
    def database_name_to_symbol(self, database_name: str) -> str:
        """Convert database name to symbol (BITCOIN -> BTC)"""
        return self.database_to_symbol.get(database_name.upper(), database_name.upper())
    
    def symbol_to_coingecko_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko ID (BTC -> bitcoin)"""
        mapping = self.token_mappings.get(symbol.upper())
        return mapping['coingecko_id'] if mapping else symbol.lower()
    
    def symbol_to_cmc_slug(self, symbol: str) -> str:
        """Convert symbol to CoinMarketCap slug (BTC -> bitcoin)"""
        mapping = self.token_mappings.get(symbol.upper())
        return mapping['cmc_slug'] if mapping else symbol.lower()
    
    def coingecko_id_to_symbol(self, coingecko_id: str) -> str:
        """Convert CoinGecko ID to symbol (bitcoin -> BTC)"""
        return self.coingecko_to_symbol.get(coingecko_id.lower(), coingecko_id.upper())
    
    def cmc_slug_to_symbol(self, cmc_slug: str) -> str:
        """Convert CoinMarketCap slug to symbol (bitcoin -> BTC)"""
        return self.cmc_to_symbol.get(cmc_slug.lower(), cmc_slug.upper())
    
    def get_display_name(self, symbol: str) -> str:
        """Get display name for symbol (BTC -> Bitcoin)"""
        mapping = self.token_mappings.get(symbol.upper())
        return mapping['display_name'] if mapping else symbol.upper()
    
    def database_lookup_symbol_to_coingecko_id(self, symbol: str) -> Optional[str]:
        """
        Query market_data table to find CoinGecko ID for a symbol
        """
        # Input validation
        if not symbol or not isinstance(symbol, str):
            logger.logger.warning(f"Invalid symbol provided: {symbol}")
            return self.symbol_to_coingecko_id(symbol) if symbol else None
        
        # Get database instance via lazy initialization
        db = self._get_database_instance()
        if db is None:
            logger.logger.debug(f"Database unavailable for {symbol}, using TokenMappingManager mapping")
            return self.symbol_to_coingecko_id(symbol)
        
        try:
            conn, cursor = db._get_connection()
            
            cursor.execute("""
                SELECT coin_id FROM market_data 
                WHERE UPPER(symbol) = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol.upper(),))
            
            result = cursor.fetchone()
            if result and result.get('coin_id'):
                return result['coin_id']
            return None
                
        except Exception as e:
            logger.logger.warning(f"Database lookup failed for {symbol}: {str(e)}")
            return self.symbol_to_coingecko_id(symbol)
    
    def database_lookup_symbol_to_cmc_slug(self, symbol: str) -> Optional[str]:
        """
        Query coinmarketcap_market_data table to find CoinMarketCap slug for a symbol
        """
        if not symbol or not isinstance(symbol, str):
            return self.symbol_to_cmc_slug(symbol) if symbol else None
        
        db = self._get_database_instance()
        if db is None:
            return self.symbol_to_cmc_slug(symbol)
        
    #    try:
    #        conn, cursor = db._get_connection()
    #        
    #        cursor.execute("""
    #            SELECT slug FROM coinmarketcap_market_data 
    #            WHERE UPPER(symbol) = ?
    #            ORDER BY timestamp DESC
    #            LIMIT 1
    #        """, (symbol.upper(),))
    #        
    #        result = cursor.fetchone()
    #        if result and result.get('slug'):
    #            return result['slug']
    #        return None
    #            
    #    except Exception as e:
    #       return self.symbol_to_cmc_slug(symbol)
        return self.symbol_to_cmc_slug(symbol)
        
    def database_lookup_coingecko_id_to_symbol(self, coingecko_id: str) -> Optional[str]:
        """
        Query market_data table to find symbol for a CoinGecko ID
        """
        if not coingecko_id or not isinstance(coingecko_id, str):
            return self.coingecko_id_to_symbol(coingecko_id) if coingecko_id else None
        
        db = self._get_database_instance()
        if db is None:
            return self.coingecko_id_to_symbol(coingecko_id)
        
        try:
            conn, cursor = db._get_connection()
            
            cursor.execute("""
                SELECT symbol FROM market_data 
                WHERE LOWER(coin_id) = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (coingecko_id.lower(),))
            
            result = cursor.fetchone()
            if result and result.get('symbol'):
                return result['symbol'].upper()
            return None
                
        except Exception as e:
            return self.coingecko_id_to_symbol(coingecko_id)
    
    def database_lookup_cmc_slug_to_symbol(self, cmc_slug: str) -> Optional[str]:
        """
        Query coinmarketcap_market_data table to find symbol for a CoinMarketCap slug
        """
        if not cmc_slug or not isinstance(cmc_slug, str):
            return self.cmc_slug_to_symbol(cmc_slug) if cmc_slug else None
        
        db = self._get_database_instance()
        if db is None:
            return self.cmc_slug_to_symbol(cmc_slug)
        
    #    try:
    #        conn, cursor = db._get_connection()
    #        
    #        cursor.execute("""
    #            SELECT symbol FROM coinmarketcap_market_data 
    #            WHERE LOWER(slug) = ?
    #            ORDER BY timestamp DESC
    #            LIMIT 1
    #        """, (cmc_slug.lower(),))
    #        
    #        result = cursor.fetchone()
    #        if result and result.get('symbol'):
    #            return result['symbol'].upper()
    #        return None
    #            
    #    except Exception as e:
    #        return self.cmc_slug_to_symbol(cmc_slug)
        return self.cmc_slug_to_symbol(cmc_slug)    

    def get_token_from_all_sources(self, identifier: str, identifier_type: str = 'symbol') -> Dict[str, Any]:
        """
        Check both database tables and hardcoded mappings for token information
        
        Args:
            identifier: Token identifier (symbol, coingecko_id, or cmc_slug)
            identifier_type: Type of identifier ('symbol', 'coingecko_id', 'cmc_slug')
            
        Returns:
            Dictionary with token information from all available sources
        """
        result = {
            'identifier': identifier,
            'identifier_type': identifier_type,
            'found_in_sources': [],
            'symbol': None,
            'coingecko_id': None,
            'cmc_slug': None,
            'database_name': None
        }
        
        try:
            if identifier_type == 'symbol':
                # Try database lookups first
                coingecko_id = self.database_lookup_symbol_to_coingecko_id(identifier)
                if coingecko_id:
                    result['coingecko_id'] = coingecko_id
                    result['found_in_sources'].append('coingecko_database')
                
                cmc_slug = self.database_lookup_symbol_to_cmc_slug(identifier)
                if cmc_slug:
                    result['cmc_slug'] = cmc_slug
                    result['found_in_sources'].append('coinmarketcap_database')
                
                # Get hardcoded mapping info
                token_info = self.get_token_info(identifier)
                if token_info:
                    result['symbol'] = token_info.get('symbol')
                    result['database_name'] = token_info.get('database_name')
                    if not result['coingecko_id']:
                        result['coingecko_id'] = token_info.get('coingecko_id')
                    if not result['cmc_slug']:
                        result['cmc_slug'] = token_info.get('cmc_slug')
                    result['found_in_sources'].append('hardcoded_mapping')
                    
            elif identifier_type == 'coingecko_id':
                # Try database lookup first
                symbol = self.database_lookup_coingecko_id_to_symbol(identifier)
                if symbol:
                    result['symbol'] = symbol
                    result['found_in_sources'].append('coingecko_database')
                    # Get additional info using the symbol
                    additional_info = self.get_token_from_all_sources(symbol, 'symbol')
                    result.update({k: v for k, v in additional_info.items() if v and k != 'found_in_sources'})
                    result['found_in_sources'].extend(additional_info['found_in_sources'])
                else:
                    # Fallback to hardcoded mapping
                    symbol = self.coingecko_id_to_symbol(identifier)
                    if symbol != identifier.upper():  # Found a mapping
                        result['symbol'] = symbol
                        result['found_in_sources'].append('hardcoded_mapping')
                        
            elif identifier_type == 'cmc_slug':
                # Try database lookup first
                symbol = self.database_lookup_cmc_slug_to_symbol(identifier)
                if symbol:
                    result['symbol'] = symbol
                    result['found_in_sources'].append('coinmarketcap_database')
                    # Get additional info using the symbol
                    additional_info = self.get_token_from_all_sources(symbol, 'symbol')
                    result.update({k: v for k, v in additional_info.items() if v and k != 'found_in_sources'})
                    result['found_in_sources'].extend(additional_info['found_in_sources'])
                else:
                    # Fallback to hardcoded mapping
                    symbol = self.cmc_slug_to_symbol(identifier)
                    if symbol != identifier.upper():  # Found a mapping
                        result['symbol'] = symbol
                        result['found_in_sources'].append('hardcoded_mapping')
            
            # Remove duplicates from sources
            result['found_in_sources'] = list(set(result['found_in_sources']))
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_all_available_tokens(self, include_database: bool = True) -> Dict[str, List[str]]:
        """
        Get combined token list from both database tables and hardcoded mappings
        """
        result = {
            'hardcoded_symbols': [],
            'coingecko_database_symbols': [],
            'coinmarketcap_database_symbols': [],
            'all_unique_symbols': []
        }
        
        try:
            # Get hardcoded symbols (always available)
            result['hardcoded_symbols'] = self.get_all_symbols()
            
            if include_database:
                db = self._get_database_instance()
                if db is not None:
                    try:
                        conn, cursor = db._get_connection()
                        
                        # CoinGecko symbols
                        cursor.execute("""
                            SELECT DISTINCT UPPER(symbol) as symbol 
                            FROM market_data 
                            WHERE symbol IS NOT NULL 
                            AND symbol != ''
                            ORDER BY symbol
                        """)
                        result['coingecko_database_symbols'] = [row['symbol'] for row in cursor.fetchall()]
                        
                        # CoinMarketCap symbols
                        cursor.execute("""
                            SELECT DISTINCT UPPER(symbol) as symbol 
                            FROM coinmarketcap_market_data 
                            WHERE symbol IS NOT NULL 
                            AND symbol != ''
                            ORDER BY symbol
                        """)
                        result['coinmarketcap_database_symbols'] = [row['symbol'] for row in cursor.fetchall()]
                        
                    except Exception as e:
                        logger.logger.warning(f"Database token lookup failed: {e}")
            
            # Combine all unique symbols
            all_symbols = set()
            all_symbols.update(result['hardcoded_symbols'])
            all_symbols.update(result['coingecko_database_symbols'])
            all_symbols.update(result['coinmarketcap_database_symbols'])
            result['all_unique_symbols'] = sorted(all_symbols)
            
        except Exception as e:
            logger.logger.error(f"Token enumeration failed: {e}")
            result['all_unique_symbols'] = result['hardcoded_symbols']  # Fallback to hardcoded only
        
        return result
    
    def analyze_token_coverage(self) -> Dict[str, Any]:
        """
        Analyze token coverage across hardcoded mappings and database tables
        
        Returns:
            Dictionary with coverage analysis
        """
        try:
            all_tokens = self.get_all_available_tokens()
            
            analysis = {
                'total_unique_tokens': len(all_tokens['all_unique_symbols']),
                'hardcoded_count': len(all_tokens['hardcoded_symbols']),
                'coingecko_db_count': len(all_tokens['coingecko_database_symbols']),
                'coinmarketcap_db_count': len(all_tokens['coinmarketcap_database_symbols']),
                'coverage_gaps': {},
                'overlaps': {}
            }
            
            hardcoded_set = set(all_tokens['hardcoded_symbols'])
            coingecko_set = set(all_tokens['coingecko_database_symbols'])
            coinmarketcap_set = set(all_tokens['coinmarketcap_database_symbols'])
            
            # Find gaps (tokens in one source but not others)
            analysis['coverage_gaps'] = {
                'hardcoded_only': list(hardcoded_set - coingecko_set - coinmarketcap_set),
                'coingecko_only': list(coingecko_set - hardcoded_set - coinmarketcap_set),
                'coinmarketcap_only': list(coinmarketcap_set - hardcoded_set - coingecko_set),
                'missing_from_hardcoded': list((coingecko_set | coinmarketcap_set) - hardcoded_set)
            }
            
            # Find overlaps (tokens in multiple sources)
            analysis['overlaps'] = {
                'hardcoded_and_coingecko': list(hardcoded_set & coingecko_set),
                'hardcoded_and_coinmarketcap': list(hardcoded_set & coinmarketcap_set),
                'coingecko_and_coinmarketcap': list(coingecko_set & coinmarketcap_set),
                'all_three_sources': list(hardcoded_set & coingecko_set & coinmarketcap_set)
            }
            
            # Calculate coverage percentages
            total_tokens = analysis['total_unique_tokens']
            if total_tokens > 0:
                analysis['coverage_percentages'] = {
                    'hardcoded': round((analysis['hardcoded_count'] / total_tokens) * 100, 1),
                    'coingecko_db': round((analysis['coingecko_db_count'] / total_tokens) * 100, 1),
                    'coinmarketcap_db': round((analysis['coinmarketcap_db_count'] / total_tokens) * 100, 1),
                    'all_three': round((len(analysis['overlaps']['all_three_sources']) / total_tokens) * 100, 1)
                }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e), 'analysis_failed': True}
        
    def get_missing_tokens(self, comparison_type: str = 'all') -> Dict[str, Any]:
        """
        Identify tokens that are missing from specific sources compared to others
        
        Args:
            comparison_type: Type of comparison to perform
                - 'all': Compare each source against union of all other sources
                - 'pairwise': Compare sources against each other in pairs
                - 'database_only': Focus only on CoinGecko vs CoinMarketCap database tables
                
        Returns:
            Dictionary with lists of missing tokens organized by source and comparison type
        """
        try:
            all_tokens = self.get_all_available_tokens()
            
            if not all_tokens or 'all_unique_symbols' not in all_tokens:
                return {'error': 'Failed to retrieve token data', 'analysis_failed': True}
            
            hardcoded_set = set(all_tokens['hardcoded_symbols'])
            coingecko_set = set(all_tokens['coingecko_database_symbols'])
            coinmarketcap_set = set(all_tokens['coinmarketcap_database_symbols'])
            
            missing_analysis = {
                'comparison_type': comparison_type,
                'missing_tokens': {},
                'summary': {}
            }
            
            if comparison_type == 'all':
                missing_analysis['missing_tokens'] = {
                    'missing_from_hardcoded': list((coingecko_set | coinmarketcap_set) - hardcoded_set),
                    'missing_from_coingecko': list((hardcoded_set | coinmarketcap_set) - coingecko_set),
                    'missing_from_coinmarketcap': list((hardcoded_set | coingecko_set) - coinmarketcap_set)
                }
                
                missing_counts = {
                    'hardcoded': len(missing_analysis['missing_tokens']['missing_from_hardcoded']),
                    'coingecko': len(missing_analysis['missing_tokens']['missing_from_coingecko']),
                    'coinmarketcap': len(missing_analysis['missing_tokens']['missing_from_coinmarketcap'])
                }
                
                missing_analysis['summary'] = {
                    'total_missing': sum(missing_counts.values()),
                    'missing_counts': missing_counts,
                    'biggest_gap_source': max(missing_counts.items(), key=lambda x: x[1])[0] if missing_counts else None,
                    'biggest_gap_count': max(missing_counts.values()) if missing_counts else 0
                }
                
            elif comparison_type == 'pairwise':
                missing_analysis['missing_tokens'] = {
                    'hardcoded_vs_coingecko': {
                        'missing_from_hardcoded': list(coingecko_set - hardcoded_set),
                        'missing_from_coingecko': list(hardcoded_set - coingecko_set)
                    },
                    'hardcoded_vs_coinmarketcap': {
                        'missing_from_hardcoded': list(coinmarketcap_set - hardcoded_set),
                        'missing_from_coinmarketcap': list(hardcoded_set - coinmarketcap_set)
                    },
                    'coingecko_vs_coinmarketcap': {
                        'missing_from_coingecko': list(coinmarketcap_set - coingecko_set),
                        'missing_from_coinmarketcap': list(coingecko_set - coinmarketcap_set)
                    }
                }
                
            elif comparison_type == 'database_only':
                missing_analysis['missing_tokens'] = {
                    'missing_from_coingecko_db': list(coinmarketcap_set - coingecko_set),
                    'missing_from_coinmarketcap_db': list(coingecko_set - coinmarketcap_set)
                }
                
                missing_analysis['summary'] = {
                    'coingecko_missing_count': len(missing_analysis['missing_tokens']['missing_from_coingecko_db']),
                    'coinmarketcap_missing_count': len(missing_analysis['missing_tokens']['missing_from_coinmarketcap_db']),
                    'total_synchronization_gap': len(missing_analysis['missing_tokens']['missing_from_coingecko_db']) + len(missing_analysis['missing_tokens']['missing_from_coinmarketcap_db'])
                }
            else:
                return {'error': f'Invalid comparison_type: {comparison_type}', 'valid_types': ['all', 'pairwise', 'database_only']}
            
            return missing_analysis
            
        except Exception as e:
            return {'error': str(e), 'analysis_failed': True}  

    def get_overlapping_tokens(self, comparison_type: str = 'all') -> Dict[str, Any]:
        """
        Find tokens that exist in multiple sources (overlapping tokens between sources)
        
        Args:
            comparison_type: Type of overlap analysis to perform
                - 'all': Show overlaps between all source combinations
                - 'pairwise': Show overlaps between specific pairs only
                - 'database_only': Focus only on CoinGecko vs CoinMarketCap database overlaps
                
        Returns:
            Dictionary with lists of overlapping tokens organized by source combinations
        """
        try:
            all_tokens = self.get_all_available_tokens()
            
            if not all_tokens or 'all_unique_symbols' not in all_tokens:
                return {'error': 'Failed to retrieve token data', 'analysis_failed': True}
            
            hardcoded_set = set(all_tokens['hardcoded_symbols'])
            coingecko_set = set(all_tokens['coingecko_database_symbols'])
            coinmarketcap_set = set(all_tokens['coinmarketcap_database_symbols'])
            
            overlap_analysis = {
                'comparison_type': comparison_type,
                'overlapping_tokens': {},
                'summary': {}
            }
            
            if comparison_type == 'all':
                overlap_analysis['overlapping_tokens'] = {
                    'hardcoded_and_coingecko': list(hardcoded_set & coingecko_set),
                    'hardcoded_and_coinmarketcap': list(hardcoded_set & coinmarketcap_set),
                    'coingecko_and_coinmarketcap': list(coingecko_set & coinmarketcap_set),
                    'all_three_sources': list(hardcoded_set & coingecko_set & coinmarketcap_set)
                }
                
                overlap_counts = {
                    'hardcoded_coingecko': len(overlap_analysis['overlapping_tokens']['hardcoded_and_coingecko']),
                    'hardcoded_coinmarketcap': len(overlap_analysis['overlapping_tokens']['hardcoded_and_coinmarketcap']),
                    'coingecko_coinmarketcap': len(overlap_analysis['overlapping_tokens']['coingecko_and_coinmarketcap']),
                    'all_three': len(overlap_analysis['overlapping_tokens']['all_three_sources'])
                }
                
                overlap_analysis['summary'] = {
                    'total_overlaps': sum(overlap_counts.values()),
                    'overlap_counts': overlap_counts,
                    'highest_overlap_pair': max(overlap_counts.items(), key=lambda x: x[1])[0] if overlap_counts else None,
                    'highest_overlap_count': max(overlap_counts.values()) if overlap_counts else 0,
                    'complete_coverage_tokens': len(overlap_analysis['overlapping_tokens']['all_three_sources'])
                }
                
            elif comparison_type == 'pairwise':
                overlap_analysis['overlapping_tokens'] = {
                    'hardcoded_vs_coingecko': list(hardcoded_set & coingecko_set),
                    'hardcoded_vs_coinmarketcap': list(hardcoded_set & coinmarketcap_set),
                    'coingecko_vs_coinmarketcap': list(coingecko_set & coinmarketcap_set)
                }
                
                pairwise_counts = {
                    'hardcoded_coingecko': len(overlap_analysis['overlapping_tokens']['hardcoded_vs_coingecko']),
                    'hardcoded_coinmarketcap': len(overlap_analysis['overlapping_tokens']['hardcoded_vs_coinmarketcap']),
                    'coingecko_coinmarketcap': len(overlap_analysis['overlapping_tokens']['coingecko_vs_coinmarketcap'])
                }
                
                overlap_analysis['summary'] = {
                    'pairwise_overlap_counts': pairwise_counts,
                    'best_synchronized_pair': max(pairwise_counts.items(), key=lambda x: x[1])[0] if pairwise_counts else None,
                    'worst_synchronized_pair': min(pairwise_counts.items(), key=lambda x: x[1])[0] if pairwise_counts else None
                }
                
            elif comparison_type == 'database_only':
                overlap_analysis['overlapping_tokens'] = {
                    'coingecko_coinmarketcap_overlap': list(coingecko_set & coinmarketcap_set)
                }
                
                database_overlap_count = len(overlap_analysis['overlapping_tokens']['coingecko_coinmarketcap_overlap'])
                total_database_tokens = len(coingecko_set | coinmarketcap_set)
                
                overlap_analysis['summary'] = {
                    'database_overlap_count': database_overlap_count,
                    'total_database_tokens': total_database_tokens,
                    'database_overlap_percentage': round((database_overlap_count / total_database_tokens) * 100, 1) if total_database_tokens > 0 else 0,
                    'coingecko_unique_count': len(coingecko_set - coinmarketcap_set),
                    'coinmarketcap_unique_count': len(coinmarketcap_set - coingecko_set)
                }
            else:
                return {'error': f'Invalid comparison_type: {comparison_type}', 'valid_types': ['all', 'pairwise', 'database_only']}
            
            return overlap_analysis
            
        except Exception as e:
            return {'error': str(e), 'analysis_failed': True}      
    
    def get_all_symbols(self) -> list:
        """Get list of all supported symbols"""
        return list(self.token_mappings.keys())
    
    def get_all_coingecko_ids(self) -> list:
        """Get list of all CoinGecko IDs"""
        return [mapping['coingecko_id'] for mapping in self.token_mappings.values()]
    
    def is_supported_token(self, identifier: str, source: str = 'symbol') -> bool:
        """Check if token is supported by any identifier"""
        identifier = identifier.upper() if source == 'symbol' else identifier.lower()
        
        if source == 'symbol':
            return identifier in self.token_mappings
        elif source == 'database':
            return identifier in self.database_to_symbol
        elif source == 'coingecko':
            return identifier in self.coingecko_to_symbol
        elif source == 'cmc':
            return identifier in self.cmc_to_symbol
        return False
    
    def get_token_info(self, symbol: str) -> dict:
        """Get complete token information"""
        mapping = self.token_mappings.get(symbol.upper())
        if mapping:
            return {
                'symbol': symbol.upper(),
                'database_name': mapping['database_name'],
                'coingecko_id': mapping['coingecko_id'],
                'cmc_slug': mapping['cmc_slug'],
                'display_name': mapping['display_name']
            }
        return {}
    
    def get_legacy_tracked_crypto_format(self) -> dict:
        """
        Return data in the legacy TRACKED_CRYPTO format for backward compatibility
        This allows existing code to continue working while we transition
        """
        legacy_format = {}
        for symbol, mapping in self.token_mappings.items():
            coingecko_id = mapping['coingecko_id']
            legacy_format[coingecko_id] = {
                'id': coingecko_id,
                'symbol': symbol.lower(),
                'name': mapping['display_name']
            }
        return legacy_format
    
    # ========================================================================
    # STANDARDIZATION METHODS FOR DATA PROCESSING
    # ========================================================================
    
    def standardize_market_data_item(self, item: dict, source: str) -> dict:
        """
        Enhanced standardize a single market data item from any source using database-aware TokenMappingManager
        
        Args:
            item: Raw market data item
            source: Data source ('database', 'coingecko', 'coinmarketcap')
            
        Returns:
            Standardized market data item with database-first fallback hierarchy
        """
        try:
            # ENHANCED: Get the standardized symbol using database-aware methods
            symbol = None
            
            if source == 'database':
                chain_value = item.get('chain', '')
                if chain_value:
                    # ENHANCED: Database-first approach with fallback
                    try:
                        # First: Try database lookup for validation/cross-reference
                        symbol = self.database_lookup_symbol_to_coingecko_id(chain_value)
                        if symbol:
                            # Convert back to symbol if we got a coingecko_id
                            symbol = self.database_lookup_coingecko_id_to_symbol(symbol) or chain_value.upper()
                            logger.logger.debug(f"✅ Database cross-reference: {chain_value} -> {symbol}")
                        else:
                            # Fallback: Use hardcoded mapping
                            symbol = self.database_name_to_symbol(chain_value)
                            logger.logger.debug(f"✅ Hardcoded mapping: {chain_value} -> {symbol}")
                    except Exception as e:
                        # Final fallback: Use chain value directly
                        symbol = chain_value.upper()
                        logger.logger.debug(f"⚠️ Direct fallback: {chain_value} -> {symbol}")
                        
            elif source == 'coingecko':
                coingecko_id = item.get('id', '')
                if coingecko_id:
                    # ENHANCED: Database-first approach with fallback
                    try:
                        # First: Try database lookup (most current mapping)
                        symbol = self.database_lookup_coingecko_id_to_symbol(coingecko_id)
                        if symbol:
                            logger.logger.debug(f"✅ Database lookup: {coingecko_id} -> {symbol}")
                        else:
                            # Fallback: Use hardcoded mapping
                            symbol = self.coingecko_id_to_symbol(coingecko_id)
                            logger.logger.debug(f"✅ Hardcoded mapping: {coingecko_id} -> {symbol}")
                    except Exception as e:
                        # Final fallback: Use ID directly
                        symbol = coingecko_id.upper()
                        logger.logger.debug(f"⚠️ Direct fallback: {coingecko_id} -> {symbol}")
                        
            elif source == 'coinmarketcap':
                cmc_slug = item.get('slug', '')
                if cmc_slug:
                    # ENHANCED: Database-first approach with fallback
                    try:
                        # First: Try database lookup (most current mapping)
                        symbol = self.database_lookup_cmc_slug_to_symbol(cmc_slug)
                        if symbol:
                            logger.logger.debug(f"✅ Database lookup: {cmc_slug} -> {symbol}")
                        else:
                            # Fallback: Use hardcoded mapping
                            symbol = self.cmc_slug_to_symbol(cmc_slug)
                            logger.logger.debug(f"✅ Hardcoded mapping: {cmc_slug} -> {symbol}")
                    except Exception as e:
                        # Final fallback: Use slug directly
                        symbol = cmc_slug.upper()
                        logger.logger.debug(f"⚠️ Direct fallback: {cmc_slug} -> {symbol}")
            else:
                # Unknown source - try to get symbol directly with validation
                symbol = item.get('symbol', 'UNKNOWN').upper()
                if symbol and symbol != 'UNKNOWN':
                    # ENHANCED: Try to validate through database lookups
                    try:
                        token_info = self.get_token_from_all_sources(symbol, 'symbol')
                        if token_info and token_info.get('found_in_sources'):
                            logger.logger.debug(f"✅ Symbol validated: {symbol} (sources: {token_info['found_in_sources']})")
                        else:
                            logger.logger.debug(f"⚠️ Unknown symbol: {symbol}")
                    except Exception as e:
                        logger.logger.debug(f"⚠️ Symbol validation failed: {symbol}")
            
            # Create standardized item structure (preserving all existing fields)
            standardized = {
                'symbol': symbol or 'UNKNOWN',
                'current_price': self._safe_float(item.get('current_price') or item.get('price', 0)),
                'price_change_percentage_24h': self._safe_float(item.get('price_change_percentage_24h') or item.get('price_change_24h', 0)),
                'total_volume': self._safe_float(item.get('total_volume') or item.get('volume', 0)),
                'market_cap': self._safe_float(item.get('market_cap', 0)),
                'last_updated': item.get('last_updated') or item.get('timestamp'),
                
                # Preserve original source data
                'source': source,
                'original_data': item.copy()
            }
            
            return standardized
            
        except Exception as e:
            logger.logger.warning(f"Error standardizing {source} data item: {e}")
            return {
                'symbol': 'UNKNOWN',
                'current_price': 0.0,
                'price_change_percentage_24h': 0.0,
                'total_volume': 0.0,
                'market_cap': 0.0,
                'last_updated': None,
                'source': source,
                'original_data': item.copy()
            }
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float"""
        try:
            if value is None or value == '':
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0     

class FieldMappings(TypedDict):
    """Field mapping configurations for different data sources"""
    database_to_standard: Dict[str, str]
    coingecko_to_standard: Dict[str, str]
    coinmarketcap_to_standard: Dict[str, str]
    standard_fields: List[str]

class DataFormatConfig(TypedDict):
    """Complete data format standardization configuration"""
    source_detection: DataSourceDetection
    field_mappings: FieldMappings
    token_mappings: TokenMappingManager
    fallback_values: Dict[str, Any]
    validation_rules: Dict[str, Dict[str, Any]]

@dataclass
class Config:
    def __init__(self) -> None:
        # Get the absolute path to the .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        logger.logger.info(f"Loading .env from: {env_path}")

        # Load environment variables
        if not os.getenv('LOADED_DOTENV'):
            load_dotenv(env_path)
            os.environ['LOADED_DOTENV'] = 'true'
        logger.logger.info("Environment variables loaded")
        
        # Initialize LLM provider configuration
        self.LLM_PROVIDER: str = os.getenv('LLM_PROVIDER', 'anthropic')
        # Use client-agnostic naming for API keys and models
        self.client_API_KEY: str = self._get_provider_api_key()
        self.client_MODEL: str = self._get_provider_model()

        # Timeline scraper configuration
        self.max_concurrent_requests = 10  # Max concurrent requests
        self.circuit_breaker_threshold = 5  # Max consecutive errors before circuit breaker
        self.recovery_wait_time = 10  # Seconds to wait during recovery
        self.max_consecutive_failures = 3  # Max failures before giving up
        self.cache_extracted_data = True  # Enable extraction caching
        self.skip_honeypot_detection = True  # Skip honeypot detection for speed
        self.enable_debug_screenshots = False  # Disable debug screenshots by default
        self.recovery_mode_duration = 300  # Recovery mode duration in seconds (5 minutes)
        self.anti_detection_delay_range = (0.5, 2.0)  # Min and max delay range in seconds for anti-detection
        self.max_scroll_attempts = 15  # Maximum number of scroll attempts during timeline scraping
        self.element_wait_timeout = 10

        # Debug loaded variables
        logger.logger.info(f"CHROME_DRIVER_PATH: {os.getenv('CHROME_DRIVER_PATH')}")
        logger.logger.info(f"LLM Provider: {self.LLM_PROVIDER}")
        logger.logger.info(f"Client API Key Present: {bool(self.client_API_KEY)}")
        logger.logger.info(f"Client Model: {self.client_MODEL}")
        
        # Provider-specific configurations
        self.PROVIDER_CONFIG: ProviderConfig = {
            "anthropic": {
                "api_key": os.getenv('CLAUDE_API_KEY', ''),
                "model": os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
            },
            "openai": {
                "api_key": os.getenv('OPENAI_API_KEY', ''),
                "model": os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
            },
            "mistral": {
                "api_key": os.getenv('MISTRAL_API_KEY', ''),
                "model": os.getenv('MISTRAL_MODEL', 'mistral-medium')
            },
            "groq": {
                "api_key": os.getenv('GROQ_API_KEY', ''),
                "model": os.getenv('GROQ_MODEL', 'llama2-70b-4096')
            }
        }
        
        # Initialize database
        from database import get_database_instance
        self.db = get_database_instance(db_path=self.get_database_path())
        logger.logger.info(f"Database initialized at: {self.get_database_path()}")
        
        # Analysis Intervals and Thresholds
        self.BASE_INTERVAL: int = 300  # 5 minutes in seconds
        self.PRICE_CHANGE_THRESHOLD: float = 5.0  # 5% change triggers post
        self.VOLUME_CHANGE_THRESHOLD: float = 10.0  # 10% change triggers post
        self.VOLUME_WINDOW_MINUTES: int = 60  # Track volume over 1 hour
        self.VOLUME_TREND_THRESHOLD: float = 15.0  # 15% change over rolling window triggers post
        
        # Google Sheets Configuration (maintained for compatibility)
        self.GOOGLE_SHEETS_PROJECT_ID: str = os.getenv('GOOGLE_SHEETS_PROJECT_ID', '')
        self.GOOGLE_SHEETS_PRIVATE_KEY: str = os.getenv('GOOGLE_SHEETS_PRIVATE_KEY', '')
        self.GOOGLE_SHEETS_CLIENT_EMAIL: str = os.getenv('GOOGLE_SHEETS_CLIENT_EMAIL', '')
        self.GOOGLE_SHEET_ID: str = os.getenv('GOOGLE_SHEET_ID', '')
        self.GOOGLE_SHEETS_RANGE: str = 'Market Analysis!A:F'

        # Twitter Configuration
        self.TWITTER_USERNAME: str = os.getenv('TWITTER_USERNAME', '')
        self.TWITTER_PASSWORD: str = os.getenv('TWITTER_PASSWORD', '')
        self.CHROME_DRIVER_PATH: str = os.getenv('CHROME_DRIVER_PATH', '/usr/local/bin/chromedriver')
        
        # Analysis Configuration
        self.CORRELATION_INTERVAL: int = 5  # minutes (for testing)
        self.MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', '3'))
        
        # Market Analysis Parameters
        self.MARKET_ANALYSIS_CONFIG: MarketAnalysisConfig = {
            'correlation_sensitivity': 0.7,
            'volatility_threshold': 2.0,
            'volume_significance': 100000,
            'historical_periods': [1, 4, 24]
        }
        
        # Tweet Length Constraints
        self.TWEET_CONSTRAINTS: TweetConstraints = {
            'MIN_LENGTH': 220,
            'MAX_LENGTH': 270,
            'HARD_STOP_LENGTH': 280
        }
        
        # API Endpoints
        self.COINMARKETCAP_API_KEY = os.getenv('CoinMarketCap_API', '')
        self.COINGECKO_BASE_URL: str = "https://api.coingecko.com/api/v3"
        
        # ================================================================
        # 🎯 DYNAMIC TOKEN SELECTION FOR COINGECKO_PARAMS 🎯
        # ================================================================
        
        # Get dynamic token IDs using TokenMappingManager (SAFETY-FIRST APPROACH)
        dynamic_token_ids = None
        
        try:
            # Initialize TokenMappingManager first
            temp_token_mapper = TokenMappingManager()
            
            # PRIMARY: TokenMappingManager database + hardcoded lookup
            try:
                all_tokens_info = temp_token_mapper.get_all_available_tokens(include_database=True)
                if all_tokens_info and all_tokens_info.get('all_unique_symbols'):
                    # Get top 30 symbols for trading
                    priority_symbols = all_tokens_info['all_unique_symbols'][:30]
                    
                    # Convert symbols to CoinGecko IDs
                    api_ids = []
                    for symbol in priority_symbols:
                        try:
                            coingecko_id = temp_token_mapper.symbol_to_coingecko_id(symbol)
                            if coingecko_id:
                                api_ids.append(coingecko_id)
                        except Exception:
                            continue  # Skip problematic tokens
                    
                    if api_ids and len(api_ids) >= 5:  # Minimum 5 tokens for safe trading
                        dynamic_token_ids = ",".join(api_ids)
                        logger.logger.info(f"✅ SAFE INIT: Loaded {len(api_ids)} tokens via TokenMappingManager")
                    else:
                        logger.logger.warning(f"⚠️ TokenMappingManager returned insufficient tokens: {len(api_ids)}")
                        
            except Exception as tmm_error:
                logger.logger.warning(f"⚠️ TokenMappingManager failed during init: {str(tmm_error)}")
            
            # SECONDARY: Direct database lookup if TokenMappingManager insufficient
            if not dynamic_token_ids:
                try:
                    # Use existing database instance that was already initialized
                    conn, cursor = self.db._get_connection()
                    
                    # Get top tokens by market cap with recent data
                    cursor.execute("""
                        SELECT DISTINCT coin_id, MAX(market_cap) as max_market_cap
                        FROM market_data 
                        WHERE market_cap > 1000000
                        AND coin_id IS NOT NULL
                        AND timestamp >= datetime('now', '-48 hours')
                        GROUP BY coin_id
                        ORDER BY max_market_cap DESC 
                        LIMIT 30
                    """)
                    
                    db_results = cursor.fetchall()
                    if db_results:
                        db_token_ids = [row['coin_id'] for row in db_results if row['coin_id']]
                        if db_token_ids and len(db_token_ids) >= 5:  # Minimum 5 tokens for safe trading
                            dynamic_token_ids = ",".join(db_token_ids)
                            logger.logger.info(f"✅ SAFE INIT: Loaded {len(db_token_ids)} tokens via database fallback")
                        else:
                            logger.logger.warning(f"⚠️ Database returned insufficient tokens: {len(db_token_ids)}")
                            
                except Exception as db_error:
                    logger.logger.error(f"❌ Database fallback failed during init: {str(db_error)}")
            
            # CRITICAL SAFETY CHECK: Ensure we have valid tokens before proceeding
            if not dynamic_token_ids:
                raise RuntimeError(
                    "CRITICAL TRADING BOT INITIALIZATION FAILURE: "
                    "Unable to load token list for trading operations. "
                    "Both TokenMappingManager and database lookups failed. "
                    "Cannot proceed with wallet operations for safety. "
                    "Please check database connection and TokenMappingManager configuration."
                )
            
            # Validate token format
            token_list = [token.strip() for token in dynamic_token_ids.split(",") if token.strip()]
            if len(token_list) < 5:
                raise RuntimeError(
                    f"CRITICAL TRADING BOT INITIALIZATION FAILURE: "
                    f"Insufficient tokens loaded for safe trading ({len(token_list)} tokens). "
                    f"Minimum 5 tokens required for risk management. "
                    f"Cannot proceed with wallet operations for safety."
                )
            
            logger.logger.info(f"🔒 SAFETY VERIFIED: Trading bot initialized with {len(token_list)} valid tokens")
            
        except Exception as critical_error:
            # Re-raise critical errors to prevent unsafe bot startup
            logger.logger.error(f"💥 CRITICAL INITIALIZATION ERROR: {str(critical_error)}")
            raise critical_error
        
        # CoinGecko API Request Settings - NOW WITH DYNAMIC TOKENS
        self.COINGECKO_PARAMS: CoinGeckoParams = {
            "vs_currency": "usd",
            "ids": dynamic_token_ids,  # 🎯 DYNAMIC TOKEN LIST (NO MORE HARDCODED)
            "order": "market_cap_desc",
            "per_page": 100,  # Ensure we get all tokens
            "page": 1,
            "sparkline": True,
            "price_change_percentage": "1h,24h,7d"
        }
        
        # Tracked Cryptocurrencies - Ordered consistently by market cap
        self.token_mapper = TokenMappingManager()
        
        # Enhanced Client Analysis Prompt Template - Token-agnostic
        self.client_ANALYSIS_PROMPT: str = """Analyze {token} Market Dynamics:

Current Market Data:
{token}:
- Price: ${price:,.2f}
- 24h Change: {change:.2f}%
- Volume: ${volume:,.0f}

Please provide a concise but detailed market analysis:
1. Short-term Movement: 
   - Price action in last few minutes
   - Volume profile significance
   - Immediate support/resistance levels

2. Market Microstructure:
   - Order flow analysis
   - Volume weighted price trends
   - Market depth indicators

3. Cross-Token Dynamics:
   - Correlation changes with other tokens
   - Relative strength shifts
   - Market maker activity signals

Focus on actionable micro-trends and real-time market behavior. Identify minimal but significant price movements.
Keep the analysis technical but concise, emphasizing key shifts in market dynamics."""
        
        # Prediction Configuration
        self.PREDICTION_CONFIG: PredictionConfig = {
            'enabled_timeframes': ['1h', '24h', '7d'],  # Supported prediction timeframes
            'confidence_threshold': 70.0,  # Minimum confidence percentage to display
            'model_weights': {  # Default weights for prediction models
                'technical_analysis': 0.25,
                'statistical_models': 0.25,
                'machine_learning': 0.25,
                'client_enhanced': 0.25  # Changed from claude_enhanced to client_enhanced
            },
            'prediction_frequency': 60,  # Minutes between predictions for same token
            'fomo_factor': 1.2,  # Multiplier for FOMO-inducing predictions (1.0 = neutral)
            'range_factor': 0.015,  # Default price range factor (percentage)
            'accuracy_threshold': 60.0  # Minimum accuracy to highlight in posts
        }

        # Client Prediction Prompt Template
        self.client_PREDICTION_PROMPT: str = """Generate a precise price prediction for {token} in the next {timeframe}.

Technical Analysis Summary:
- Overall Trend: {trend}
- Trend Strength: {trend_strength}/100
- RSI: {rsi:.2f}
- MACD: {macd_signal}
- Bollinger Bands: {bb_signal}
- Volatility: {volatility:.2f}%

Statistical Models:
- ARIMA Forecast: ${arima_price:.4f}
- Confidence Interval: [${arima_lower:.4f}, ${arima_upper:.4f}]

Machine Learning:
- ML Model Forecast: ${ml_price:.4f}
- Confidence Interval: [${ml_lower:.4f}, ${ml_upper:.4f}]

Current Market Data:
- Current Price: ${current_price:.4f}
- 24h Change: {price_change:.2f}%
- Market Sentiment: {market_sentiment}

Provide a precise prediction following this format:
1. Exact price target with confidence level
2. Price range (lower and upper bounds)
3. Expected percentage change
4. Brief rationale (2-3 sentences)
5. Key factors influencing the prediction

Be precise but conversational. Aim for 70-80% confidence level. Create excitement without being unrealistic."""

        # Technology Content Configuration
        self.TECH_CONTENT_CONFIG: TechContentConfig = {
            'enabled': True,
            'categories': {
                'ai': {
                    'enabled': True,
                    'priority': 10,  # 1-10 scale, higher = more priority
                    'min_interval_minutes': 240,  # Min time between posts on this topic
                    'keywords': [
                        'ai', 'artificial intelligence', 'machine learning', 'deep learning',
                        'llm', 'neural network', 'transformer', 'gpt', 'chatgpt', 'claude',
                        'gemini', 'large language model', 'generative ai', 'agi'
                    ]
                },
                'quantum': {
                    'enabled': True,
                    'priority': 8,
                    'min_interval_minutes': 480,  # Less frequent than AI
                    'keywords': [
                        'quantum computing', 'quantum computer', 'qubit', 'quantum supremacy',
                        'quantum algorithm', 'quantum advantage', 'quantum encryption',
                        'post-quantum', 'quantum security', 'quantum annealing'
                    ]
                },
                'blockchain_tech': {
                    'enabled': True,
                    'priority': 9,
                    'min_interval_minutes': 360,
                    'keywords': [
                        'blockchain technology', 'blockchain scaling', 'zero-knowledge',
                        'zk-rollup', 'layer 2', 'sharding', 'smart contract', 'dao',
                        'decentralized identity', 'consensus mechanism', 'cryptography'
                    ]
                },
                'advanced_computing': {
                    'enabled': True,
                    'priority': 7,
                    'min_interval_minutes': 720,
                    'keywords': [
                        'edge computing', 'high performance computing', 'neuromorphic',
                        'in-memory computing', 'fpga', 'asic', 'tpu', 'accelerator',
                        'optical computing', 'biological computing', 'exascale'
                    ]
                }
            },
            'integration_weight': 1.5,  # Boost for posts that integrate crypto and tech
            'educational_boost': 1.3,   # Boost for educational content
            'post_frequency': 4,        # Hours between tech-focused posts
            'reply_frequency': 3,       # Hours between tech-focused replies
            'max_daily_tech_posts': 6   # Maximum tech posts per day
        }

        # Tech content prompt templates
        self.TECH_PROMPT_CONFIG: TechPromptConfig = {
            'educational_template': """Create an educational post about {tech_topic} that's both informative and engaging.

Key points to include:
1. {key_point_1}
2. {key_point_2}
3. {key_point_3}

Make it accessible to a {audience_level} audience. Include a subtle connection to {token} if relevant.
The post should be conversational but informative, between {min_length}-{max_length} characters.

Educational goal: Help readers understand {learning_objective}""",

            'integration_template': """Create a post that explores the relationship between {token} and {tech_topic}.

Focus on:
- How {tech_topic} technology impacts or could impact {token}
- Potential synergies between {token} and {tech_topic}
- What {token} enthusiasts should know about {tech_topic}

Make the content {mood} in tone, educational yet engaging, and {min_length}-{max_length} characters.""",

            'reply_template': """Craft a reply to this tech-related post about {tech_topic}:
"{original_post}"

Your reply should:
- Be educational and add new insight
- Connect to {token} where relevant
- Maintain a {tone} tone
- End with a subtle question to encourage further discussion
- Stay under 240 characters

Provide accurate information that would be valuable to someone interested in both {tech_topic} and {token}.""",

            'complexity_levels': {
                'beginner': 0.3,
                'intermediate': 0.6,
                'advanced': 0.9
            }
        }

        # Tech analysis prompt template for more advanced tech content generation
        self.client_TECH_ANALYSIS_PROMPT: str = """Analyze the intersection of {tech_topic} and {token}:

Current Context:
- {token} Price: ${price:,.2f} ({change:+.2f}%)
- {tech_topic} Status: {tech_status_summary}
- Integration Level: {integration_level}/10

Please provide a concise but insightful analysis:
1. Technology Impact:
   - How {tech_topic} is affecting {token} ecosystem
   - Key technological developments to watch
   - Potential disruption vectors

2. Integration Opportunities:
   - Current integration points between {tech_topic} and {token}
   - Competitive advantages or challenges
   - Future potential synergies

3. Educational Insights:
   - Core concepts {token} enthusiasts should understand about {tech_topic}
   - Common misconceptions or knowledge gaps
   - Key resources for further learning

Focus on genuinely educational content that helps readers understand both technologies better.
Aim for technical accuracy while remaining accessible to a {audience_level} audience."""

        # NEW: Trading bot configuration (from integrated_trading_bot.py)
        self.TRADING_CONFIG: Dict[str, Any] = {
            'max_daily_loss': 25.0,
            'max_daily_trades': 120,
            'max_concurrent_positions': 5,
            'min_confidence_threshold': 70.0,
            'position_size_percent': 20.0,
            'stop_loss_percent': 8.0,
            'take_profit_percent': 15.0,
            'trailing_stop_percent': 5.0,
            'rebalance_threshold': 0.15,
            'emergency_exit_enabled': True,
            'auto_compound': True,
            'risk_reward_ratio': 2.0,
            'save_frequency': 10,
            'cycle_interval': 300  # 5 minutes default
        }
        
        # NEW: Token-specific risk profiles
        self.TOKEN_RISK_PROFILES: Dict[str, Dict[str, float]] = {
            'BTC': {'max_position': 0.30, 'stop_loss': 6.0, 'take_profit': 12.0},
            'ETH': {'max_position': 0.25, 'stop_loss': 7.0, 'take_profit': 14.0},
            'SOL': {'max_position': 0.20, 'stop_loss': 10.0, 'take_profit': 20.0},
            'XRP': {'max_position': 0.15, 'stop_loss': 12.0, 'take_profit': 25.0},
            'BNB': {'max_position': 0.20, 'stop_loss': 8.0, 'take_profit': 16.0},
            'AVAX': {'max_position': 0.15, 'stop_loss': 12.0, 'take_profit': 24.0}
        }
        
        # NEW: Confidence thresholds for different market conditions
        self.CONFIDENCE_THRESHOLDS: Dict[str, Dict[str, int]] = {
            'BTC': {'base': 70, 'volatile': 80, 'stable': 65},
            'ETH': {'base': 72, 'volatile': 82, 'stable': 67},
            'SOL': {'base': 75, 'volatile': 85, 'stable': 70},
            'XRP': {'base': 78, 'volatile': 88, 'stable': 73},
            'BNB': {'base': 74, 'volatile': 84, 'stable': 69},
            'AVAX': {'base': 76, 'volatile': 86, 'stable': 71}
        }
        
        # NEW: Risk management configuration
        self.RISK_CONFIG: Dict[str, float] = {
            'max_portfolio_risk': 15.0,
            'correlation_limit': 0.7,
            'volatility_threshold': 25.0,
            'drawdown_limit': 20.0,
            'var_confidence': 0.95,
            'risk_free_rate': 0.02,
            'beta_limit': 1.5,
            'sharpe_minimum': 1.0
        }
        
        # NEW: Network configuration
        self.NETWORK_CONFIG: Dict[str, Any] = {
            'preferred_networks': ["polygon", "optimism", "base", "arbitrum"],
            'gas_limits': {
                'ethereum': 150000,
                'polygon': 100000,
                'optimism': 120000,
                'arbitrum': 130000,
                'base': 110000
            },
            'slippage_tolerance': 2.0,
            'max_gas_price_gwei': 50
        }
        
        # NEW: Network-specific reliability scores
        self.NETWORK_RELIABILITY: Dict[str, Dict[str, float]] = {
            "ethereum": {"gas_cost": 0.01, "reliability": 0.99},
            "polygon": {"gas_cost": 0.0001, "reliability": 0.95},
            "optimism": {"gas_cost": 0.005, "reliability": 0.96},
            "arbitrum": {"gas_cost": 0.003, "reliability": 0.97},
            "base": {"gas_cost": 0.002, "reliability": 0.90}
        }
        
        # NEW: Execution configuration
        self.EXECUTION_CONFIG: Dict[str, Any] = {
            'max_concurrent_executions': 3,
            'max_retry_attempts': 2,
            'execution_cooldown': 5,
            'monitoring_interval': 30,
            'save_frequency': 10
        }
        
        # NEW: Security configuration
        self.SECURITY_CONFIG: Dict[str, Any] = {
            'keyring_service': "crypto_trading_bot",
            'keyring_username': "wallet_private_key",
            'encryption_enabled': True,
            'secure_storage_enabled': True
        }

        # Additional validation layer settings
        self.ACCURACY_VALIDATION_SETTINGS = {
            'data_age_warning_thresholds': {
                '1h': 360,    # 6 hours
                '24h': 1440,  # 24 hours
                '7d': 4320    # 72 hours
            },
            'confidence_magnitude_rules': {
                'low_confidence_threshold': 50.0,
                'high_magnitude_threshold': 20.0,
                'extreme_magnitude_threshold': 30.0
            },
            'market_context_validation': True,
            'real_time_cross_reference': True
        }

        # ========================================================================
        # 🎯 MULTI-API DATA FORMAT STANDARDIZATION CONFIGURATION 🎯
        # ========================================================================

        # Initialize the centralized token mapping system (150+ tokens)
        self.token_mapper = TokenMappingManager()

        # Field mappings for data standardization (keep these parts)
        self.FIELD_MAPPINGS = {
            'database_to_standard': {
                'chain': 'symbol',  
                'price': 'current_price',  
                'price_change_24h': 'price_change_percentage_24h',  
                'volume': 'total_volume',  
                'market_cap': 'market_cap',  
                'timestamp': 'last_updated',  
                'id': 'database_id'  
            },
            'coingecko_to_standard': {
                'id': 'coingecko_id',
                'current_price': 'current_price',
                'price_change_percentage_24h': 'price_change_percentage_24h',
                'total_volume': 'total_volume',
                'market_cap': 'market_cap',
                'last_updated': 'last_updated'
            },
            'coinmarketcap_to_standard': {
                'slug': 'coinmarketcap_slug',
                'quote.USD.price': 'current_price',
                'quote.USD.percent_change_24h': 'price_change_percentage_24h',
                'quote.USD.volume_24h': 'total_volume',
                'quote.USD.market_cap': 'market_cap',
                'last_updated': 'last_updated'
            }
        }

        # Source detection priorities
        self.SOURCE_DETECTION_PRIORITY = ['database', 'coingecko', 'coinmarketcap']

    @staticmethod
    def get_database_path() -> str:
        """
        Centralized database path resolver - ensures all database connections use the same absolute path
        
        Returns:
            str: Absolute path to the crypto_history.db file
        """
        # Get the project root directory (two levels up from config.py)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Construct absolute path to database
        db_path = os.path.join(project_root, 'data', 'crypto_history.db')
        
        # Ensure the data directory exists
        data_dir = os.path.dirname(db_path)
        os.makedirs(data_dir, exist_ok=True)
        
        # Log the resolved path for debugging
        logger.logger.debug(f"Database path resolved to: {db_path}")
        
        return db_path    

    def _get_provider_api_key(self) -> str:
        """Get the API key for the currently selected provider"""
        if self.LLM_PROVIDER == 'anthropic':
            return os.getenv('CLAUDE_API_KEY', '')
        elif self.LLM_PROVIDER == 'openai':
            return os.getenv('OPENAI_API_KEY', '')
        elif self.LLM_PROVIDER == 'mistral':
            return os.getenv('MISTRAL_API_KEY', '')
        elif self.LLM_PROVIDER == 'groq':
            return os.getenv('GROQ_API_KEY', '')
        return ''

    def _get_provider_model(self) -> str:
        """Get the model for the currently selected provider"""
        if self.LLM_PROVIDER == 'anthropic':
            return os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet-20250219')
        elif self.LLM_PROVIDER == 'openai':
            return os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        elif self.LLM_PROVIDER == 'mistral':
            return os.getenv('MISTRAL_MODEL', 'mistral-medium')
        elif self.LLM_PROVIDER == 'groq':
            return os.getenv('GROQ_MODEL', 'llama2-70b-4096')
        return ''

    def _validate_config(self) -> None:
        """Validate required configuration settings"""
        required_settings: List[tuple[str, str]] = [
            ('TWITTER_USERNAME', self.TWITTER_USERNAME),
            ('TWITTER_PASSWORD', self.TWITTER_PASSWORD),
            ('CHROME_DRIVER_PATH', self.CHROME_DRIVER_PATH),
            ('Client API Key', self.client_API_KEY),  # Updated from CLAUDE_API_KEY
            ('GOOGLE_SHEETS_PROJECT_ID', self.GOOGLE_SHEETS_PROJECT_ID),
            ('GOOGLE_SHEETS_PRIVATE_KEY', self.GOOGLE_SHEETS_PRIVATE_KEY),
            ('GOOGLE_SHEETS_CLIENT_EMAIL', self.GOOGLE_SHEETS_CLIENT_EMAIL),
            ('GOOGLE_SHEET_ID', self.GOOGLE_SHEET_ID),
            ('CoinMarketCap API Key', self.COINMARKETCAP_API_KEY)
        ]
        
        missing_settings: List[str] = []
        for setting_name, setting_value in required_settings:
            if not setting_value or setting_value.strip() == '':
                missing_settings.append(setting_name)
        
        if missing_settings:
            error_msg = f"Missing required configuration: {', '.join(missing_settings)}"
            logger.log_error("Config", error_msg)
            raise ValueError(error_msg)

    def get_coingecko_markets_url(self) -> str:
        """Get CoinGecko markets API endpoint"""
        return f"{self.COINGECKO_BASE_URL}/coins/markets"

    def get_coingecko_params(self, timeframe: str = "24h") -> CoinGeckoParams:
        """
        Get CoinGecko API parameters with timeframe-appropriate price change settings
        
        Args:
            timeframe: Analysis timeframe ("1h", "24h", "7d") - determines price change parameters
        
        Returns:
            CoinGeckoParams dictionary with timeframe-appropriate settings
        """
        # Determine price change parameters based on timeframe
        if timeframe == "1h":
            price_change_param = "1h,24h"  # Focus on short-term changes for hourly analysis
        elif timeframe == "24h":
            price_change_param = "1h,24h,7d"  # Full range for daily analysis
        elif timeframe == "7d":
            price_change_param = "24h,7d,30d"  # Longer-term perspective for weekly analysis
        else:
            price_change_param = "1h,24h,7d"  # Default to full range for unknown timeframes
        
        return {
            "vs_currency": "usd",
            "ids": self.COINGECKO_PARAMS["ids"],  # 🎯 USE DYNAMIC TOKENS FROM INIT
            "order": "market_cap_desc",
            "per_page": 100,  # Ensure we get all tokens
            "page": 1,
            "sparkline": True,
            "price_change_percentage": price_change_param  # 🎯 TIMEFRAME-SPECIFIC
        }

    def get_provider_config(self) -> Dict[str, str]:
        """Get the configuration for the currently selected LLM provider"""
        return self.PROVIDER_CONFIG.get(self.LLM_PROVIDER, {})

    def get_tech_topics(self) -> List[Dict[str, Any]]:
        """
        Get available tech topics for content generation based on configuration

        Returns:
            List of tech topic configurations with metadata
        """
        topics = []
        
        for category, config in self.TECH_CONTENT_CONFIG['categories'].items():
            if config['enabled']:
                topics.append({
                    'category': category,
                    'priority': config['priority'],
                    'min_interval_minutes': config['min_interval_minutes'],
                    'keywords': config['keywords']
                })
        
        # Sort by priority (highest first)
        topics.sort(key=lambda x: x['priority'], reverse=True)
        
        return topics

    def get_tech_prompt_template(self, prompt_type: str, audience_level: str = 'intermediate') -> str:
        """
        Get a prompt template for tech content generation

        Args:
            prompt_type: Type of prompt ('educational', 'integration', or 'reply')
            audience_level: Target audience expertise level

        Returns:
            Formatted prompt template
        """
        if prompt_type in self.TECH_PROMPT_CONFIG:
            return self.TECH_PROMPT_CONFIG[prompt_type]
        
        # Default to educational template
        return self.TECH_PROMPT_CONFIG['educational_template']

    def is_tech_post_allowed(self, category: str, last_post_timestamp) -> bool:
        """
        Check if posting about a tech category is allowed based on configured intervals

        Args:
            category: Tech category to check
            last_post_timestamp: Timestamp of last post about this category

        Returns:
            Whether posting is allowed
        """
        if not self.TECH_CONTENT_CONFIG['enabled']:
            return False
            
        if category not in self.TECH_CONTENT_CONFIG['categories']:
            return False
            
        category_config = self.TECH_CONTENT_CONFIG['categories'][category]
        
        if not category_config['enabled']:
            return False
        
        # Import datetime here to avoid circular imports
        from datetime import datetime, timedelta
        
        # Get min interval in minutes
        min_interval = category_config['min_interval_minutes']
        
        # Calculate time since last post
        now = datetime.now()
        time_since_last = (now - last_post_timestamp).total_seconds() / 60
        
        # Allow post if enough time has passed
        return time_since_last >= min_interval

    # NEW METHODS FOR TRADING BOT INTEGRATION
    def get_trading_config(self, key: Optional[str] = None) -> Any:
        """Get trading configuration value(s)"""
        if key:
            return self.TRADING_CONFIG.get(key)
        return self.TRADING_CONFIG.copy()
    
    def get_risk_config(self, key: Optional[str] = None) -> Any:
        """Get risk management configuration value(s)"""
        if key:
            return self.RISK_CONFIG.get(key)
        return self.RISK_CONFIG.copy()
    
    def get_network_config(self, key: Optional[str] = None) -> Any:
        """Get network configuration value(s)"""
        if key:
            return self.NETWORK_CONFIG.get(key)
        return self.NETWORK_CONFIG.copy()
    
    def get_token_risk_profile(self, token: str) -> Dict[str, float]:
        """Get risk profile for specific token"""
        return self.TOKEN_RISK_PROFILES.get(token, self.TOKEN_RISK_PROFILES['BTC'])
    
    def get_confidence_threshold(self, token: str, condition: str = 'base') -> float:
        """Get confidence threshold for token and market condition"""
        thresholds = self.CONFIDENCE_THRESHOLDS.get(token, self.CONFIDENCE_THRESHOLDS['BTC'])
        return thresholds.get(condition, thresholds['base'])
    
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """Update configuration value"""
        try:
            if section == 'trading':
                config_dict = self.TRADING_CONFIG
            elif section == 'risk':
                config_dict = self.RISK_CONFIG
            elif section == 'network':
                config_dict = self.NETWORK_CONFIG
            elif section == 'execution':
                config_dict = self.EXECUTION_CONFIG
            elif section == 'security':
                config_dict = self.SECURITY_CONFIG
            else:
                logger.logger.error(f"Invalid configuration section: {section}")
                return False
            
            config_dict[key] = value
            logger.logger.info(f"Updated {section}.{key} = {value}")
            return True
        except Exception as e:
            logger.logger.error(f"Config update failed: {e}")
            return False

    # ========================================================================
    # 🎯 REBUILT: MULTI-API DATA FORMAT STANDARDIZATION METHODS 🎯
    # ========================================================================

    # Add these constants at the class level (inside Config class)
    SOURCE_DETECTION_PRIORITY = ['database', 'coinmarketcap', 'coingecko']

    FIELD_MAPPINGS = {
        'coingecko_to_standard': {
            'id': 'coin_id',
            'symbol': 'symbol',
            'name': 'name',
            'current_price': 'current_price',
            'market_cap': 'market_cap',
            'market_cap_rank': 'market_cap_rank',
            'total_volume': 'total_volume',
            'price_change_percentage_24h': 'price_change_percentage_24h',
            'price_change_percentage_7d_in_currency': 'price_change_percentage_7d',
            'price_change_percentage_1h_in_currency': 'price_change_percentage_1h',
            'last_updated': 'last_updated'
        },
        'coinmarketcap_to_standard': {
            'id': 'cmc_id',
            'symbol': 'symbol',
            'name': 'name',
            'slug': 'slug',
            'quote_price': 'current_price',
            'quote_market_cap': 'market_cap',
            'cmc_rank': 'market_cap_rank',
            'quote_volume_24h': 'total_volume',
            'quote_percent_change_24h': 'price_change_percentage_24h',
            'quote_percent_change_7d': 'price_change_percentage_7d',
            'quote_percent_change_1h': 'price_change_percentage_1h',
            'last_updated': 'last_updated'
        },
        'database_to_standard': {
            'chain': 'symbol',
            'price': 'current_price',
            'volume': 'total_volume',
            'price_change_24h': 'price_change_percentage_24h',
            'market_cap': 'market_cap',
            'timestamp': 'last_updated'
        }
    }

    def detect_data_source(self, item: Dict[str, Any]) -> str:
        """
        REBUILT: Detect which data source format an item is using with table_source priority
        
        Args:
            item: Data item to analyze
            
        Returns:
            Data source name ('database', 'coingecko', 'coinmarketcap', 'unknown')
        """
        # PRIORITY 1: Use table_source field if present (database records)
        if 'table_source' in item:
            table_source = item['table_source']
            if table_source in ['coingecko', 'coinmarketcap']:
                logger.logger.debug(f"✅ Source detected via table_source: {table_source} (database record)")
                return 'database'  # It's a database record, regardless of original API
        
        # PRIORITY 2: API response format detection (live API calls)
        # CoinMarketCap format - has 'slug' and 'quote' structure  
        if 'slug' in item and ('quote' in item or any(key.startswith('quote_') for key in item.keys())):
            logger.logger.debug("✅ Source detected: coinmarketcap (API response)")
            return 'coinmarketcap'
        
        # CoinGecko format - has 'id' and 'current_price'
        if 'id' in item and 'current_price' in item:
            logger.logger.debug("✅ Source detected: coingecko (API response)")
            return 'coingecko'
        
        # PRIORITY 3: Database record fallback detection
        if 'chain' in item and 'price' in item:
            logger.logger.debug("✅ Source detected: database (legacy format)")
            return 'database'
        
        if 'coin_id' in item or 'cmc_id' in item:
            logger.logger.debug("✅ Source detected: database (table record)")
            return 'database'
        
        logger.logger.warning(f"💡 Unknown data source format: {list(item.keys())[:5]}...")
        return 'unknown'

    def get_field_mapping(self, source: str) -> Dict[str, str]:
        """
        Get field mapping configuration for a specific data source
        
        Args:
            source: Data source name ('database', 'coingecko', 'coinmarketcap')
            
        Returns:
            Field mapping dictionary
        """
        mapping_key = f"{source}_to_standard"
        return self.FIELD_MAPPINGS.get(mapping_key, {})

    def get_token_symbol_from_source(self, item: Dict[str, Any], source: str) -> str:
        """
        Extract standardized token symbol from data item based on source format
        
        Args:
            item: Data item containing token information
            source: Data source format ('database', 'coingecko', 'coinmarketcap')
            
        Returns:
            Standardized token symbol (e.g., 'AAVE', 'BTC')
        """
        try:
            if source == 'database':
                # Check for table_source to determine original API
                table_source = item.get('table_source')
                
                if table_source == 'coingecko':
                    # Database record from CoinGecko table
                    coin_id = item.get('coin_id') or item.get('id', '')
                    if coin_id:
                        mapped_symbol = self.token_mapper.coingecko_id_to_symbol(coin_id)
                        if mapped_symbol:
                            return mapped_symbol
                    # Fallback to symbol field
                    symbol = item.get('symbol', '').upper()
                    return symbol if symbol else 'UNKNOWN'
                    
                elif table_source == 'coinmarketcap':
                    # Database record from CoinMarketCap table
                    symbol = item.get('symbol', '').upper()
                    return symbol if symbol else 'UNKNOWN'
                    
                else:
                    # Legacy database format (original market_data table)
                    chain_value = item.get('chain') or ''
                    if chain_value:
                        # Use TokenMappingManager for conversion
                        mapped_symbol = self.token_mapper.database_name_to_symbol(chain_value)
                        return mapped_symbol
                        
            elif source == 'coingecko':
                # CoinGecko API response - uses 'id' field, map to symbol
                coingecko_id = item.get('id') or ''
                if coingecko_id:
                    # Use TokenMappingManager for conversion
                    mapped_symbol = self.token_mapper.coingecko_id_to_symbol(coingecko_id)
                    return mapped_symbol
                    
            elif source == 'coinmarketcap':
                # CoinMarketCap API response - check symbol field first
                symbol = item.get('symbol', '').upper()
                if symbol:
                    return symbol
                
                # Fallback: CoinMarketCap uses 'slug' field, map to symbol
                cmc_slug = item.get('slug') or ''
                if cmc_slug:
                    # Use TokenMappingManager for conversion
                    mapped_symbol = self.token_mapper.cmc_slug_to_symbol(cmc_slug)
                    return mapped_symbol
                    
        except Exception as e:
            logger.logger.warning(f"Error extracting token symbol from {source}: {e}")
        
        # Final fallback: try to use 'symbol' field directly
        symbol_field = item.get('symbol') or 'UNKNOWN'
        return str(symbol_field).upper()

    def standardize_field_value(self, field_name: str, value: Any) -> Any:
        """
        Validate and standardize a field value according to validation rules
        
        Args:
            field_name: Name of the field to validate
            value: Value to validate and standardize
            
        Returns:
            Validated and standardized value, or fallback value if invalid
        """
        # Define validation rules inline (since we removed DATA_FORMAT_CONFIG)
        validation_rules = {
            'current_price': {
                'type': 'float',
                'min_value': 0.0,
                'max_value': 1000000.0,
                'required': True
            },
            'price_change_percentage_24h': {
                'type': 'float',
                'min_value': -99.0,
                'max_value': 10000.0,
                'required': False
            },
            'price_change_percentage_1h': {
                'type': 'float',
                'min_value': -99.0,
                'max_value': 1000.0,
                'required': False
            },
            'price_change_percentage_7d': {
                'type': 'float',
                'min_value': -99.0,
                'max_value': 10000.0,
                'required': False
            },
            'total_volume': {
                'type': 'float',
                'min_value': 0.0,
                'max_value': 1000000000000.0,
                'required': False
            },
            'market_cap': {
                'type': 'float',
                'min_value': 0.0,
                'max_value': 10000000000000.0,
                'required': False
            },
            'market_cap_rank': {
                'type': 'int',
                'min_value': 1,
                'max_value': 10000,
                'required': False
            },
            'symbol': {
                'type': 'str',
                'min_length': 1,
                'max_length': 10,
                'required': True
            },
            'name': {
                'type': 'str',
                'min_length': 1,
                'max_length': 100,
                'required': False
            }
        }
        
        fallback_values = {
            'current_price': 0.0,
            'price_change_percentage_24h': 0.0,
            'price_change_percentage_1h': 0.0,
            'price_change_percentage_7d': 0.0,
            'total_volume': 0.0,
            'market_cap': 0.0,
            'market_cap_rank': None,
            'symbol': 'UNKNOWN',
            'name': 'Unknown Token',
            'last_updated': None
        }
        
        # Get validation rule for this field
        field_rule = validation_rules.get(field_name, {})
        fallback_value = fallback_values.get(field_name, None)
        
        try:
            # Handle None values
            if value is None:
                return fallback_value
            
            # Type validation and conversion
            expected_type = field_rule.get('type', 'str')
            
            if expected_type == 'float':
                converted_value = float(value)
                
                # Range validation
                min_val = field_rule.get('min_value', float('-inf'))
                max_val = field_rule.get('max_value', float('inf'))
                
                if not (min_val <= converted_value <= max_val):
                    logger.logger.warning(f"Field {field_name} value {converted_value} outside valid range [{min_val}, {max_val}]")
                    return fallback_value
                
                return converted_value
                
            elif expected_type == 'int':
                converted_value = int(float(value))  # Convert via float to handle string numbers
                
                # Range validation
                min_val = field_rule.get('min_value', float('-inf'))
                max_val = field_rule.get('max_value', float('inf'))
                
                if not (min_val <= converted_value <= max_val):
                    logger.logger.warning(f"Field {field_name} value {converted_value} outside valid range [{min_val}, {max_val}]")
                    return fallback_value
                
                return converted_value
                
            elif expected_type == 'str':
                str_value = str(value).strip()
                
                # Length validation
                min_len = field_rule.get('min_length', 0)
                max_len = field_rule.get('max_length', 1000)
                
                if not (min_len <= len(str_value) <= max_len):
                    logger.logger.warning(f"Field {field_name} length {len(str_value)} outside valid range [{min_len}, {max_len}]")
                    return fallback_value
                
                return str_value
                
        except (ValueError, TypeError) as e:
            logger.logger.warning(f"Field {field_name} validation failed: {e}")
            return fallback_value
        
        return value

    def get_standard_fields(self) -> List[str]:
        """
        Get list of standard fields that should be present in standardized data
        
        Returns:
            List of standard field names
        """
        return [
            'symbol',
            'name',
            'current_price',
            'price_change_percentage_24h',
            'price_change_percentage_1h',
            'price_change_percentage_7d',
            'total_volume',
            'market_cap',
            'market_cap_rank',
            'last_updated'
        ]

    def get_data_format_config(self) -> Dict[str, Any]:
        """
        Get the complete data format standardization configuration
        
        Returns:
            Complete configuration dictionary
        """
        return {
            'field_mappings': self.FIELD_MAPPINGS,
            'source_detection_priority': self.SOURCE_DETECTION_PRIORITY,
            'token_mapper': self.token_mapper,
            'standard_fields': self.get_standard_fields()
        }

    @property
    def twitter_selectors(self) -> Dict[str, str]:
        """CSS Selectors for Twitter elements"""
        return {
            'username_input': 'input[autocomplete="username"]',
            'password_input': 'input[type="password"]',
            'login_button': '[data-testid="LoginForm_Login_Button"]',
            'tweet_input': '[data-testid="tweetTextarea_0"]',
            'tweet_button': '[data-testid="tweetButton"]'
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Note: Database cleanup is handled by the main bot cleanup process
            # Don't delete database reference here as other components may still need it
            logger.logger.info("Config cleanup completed")
        except Exception as e:
            logger.logger.warning(f"Error during cleanup: {e}")

# Create singleton instance
config = Config()