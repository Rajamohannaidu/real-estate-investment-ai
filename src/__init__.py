# src/__init__.py
"""
Real Estate Investment Advisor AI - Core Modules
"""

from .data_preprocessing import RealEstateDataPreprocessor
from .predictive_models import RealEstatePredictiveModels
from .investment_analytics import InvestmentAnalytics
from .explainability import ModelExplainability
from .chatbot import RealEstateInvestmentChatbot

__version__ = "1.0.0"
__author__ = "Real Estate Investment Team"

__all__ = [
    'RealEstateDataPreprocessor',
    'RealEstatePredictiveModels',
    'InvestmentAnalytics',
    'ModelExplainability',
    'RealEstateInvestmentChatbot'
]