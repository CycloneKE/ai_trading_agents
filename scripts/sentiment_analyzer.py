"""
Sentiment Analysis module for financial text processing.
Uses domain-specific models for accurate financial sentiment analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass


class FinancialSentimentAnalyzer:
    """
    Financial sentiment analyzer using multiple models and approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configurations
        self.use_finbert = config.get('use_finbert', True)
        self.use_vader = config.get('use_vader', True)
        self.use_textblob = config.get('use_textblob', True)
        
        # Initialize models
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.vader_analyzer = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Financial keywords and phrases
        self.positive_keywords = {
            'growth', 'profit', 'revenue', 'earnings', 'beat', 'exceed', 'strong',
            'bullish', 'optimistic', 'upgrade', 'buy', 'outperform', 'positive',
            'gain', 'rise', 'increase', 'surge', 'rally', 'boom', 'success'
        }
        
        self.negative_keywords = {
            'loss', 'decline', 'fall', 'drop', 'crash', 'bearish', 'pessimistic',
            'downgrade', 'sell', 'underperform', 'negative', 'weak', 'poor',
            'disappointing', 'miss', 'below', 'concern', 'risk', 'uncertainty'
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Financial sentiment analyzer initialized")
    
    def _initialize_models(self):
        """
        Initialize sentiment analysis models.
        """
        try:
            # Initialize FinBERT for financial sentiment
            if self.use_finbert:
                try:
                    model_name = "ProsusAI/finbert"
                    self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    self.finbert_model.to(self.device)
                    self.finbert_model.eval()
                    logger.info("FinBERT model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load FinBERT: {str(e)}")
                    self.use_finbert = False
            
            # Initialize VADER sentiment analyzer
            if self.use_vader:
                try:
                    self.vader_analyzer = SentimentIntensityAnalyzer()
                    logger.info("VADER sentiment analyzer loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load VADER: {str(e)}")
                    self.use_vader = False
            
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove special characters but keep important punctuation
            text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text
    
    def analyze_with_finbert(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment scores
        """
        try:
            if not self.use_finbert or not self.finbert_model or not text:
                return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            
            # Tokenize text
            inputs = self.finbert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [negative, neutral, positive]
            scores = predictions.cpu().numpy()[0]
            
            return {
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing with FinBERT: {str(e)}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
    
    def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment scores
        """
        try:
            if not self.use_vader or not self.vader_analyzer or not text:
                return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 0.0}
            
            scores = self.vader_analyzer.polarity_scores(text)
            
            return {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': scores['compound']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing with VADER: {str(e)}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 0.0}
    
    def analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment scores
        """
        try:
            if not self.use_textblob or not text:
                return {'polarity': 0.0, 'subjectivity': 0.0}
            
            blob = TextBlob(text)
            
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
            
        except Exception as e:
            logger.error(f"Error analyzing with TextBlob: {str(e)}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def analyze_financial_keywords(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment based on financial keywords.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with keyword-based sentiment analysis
        """
        try:
            if not text:
                return {'positive_count': 0, 'negative_count': 0, 'sentiment_score': 0.0}
            
            # Tokenize and lemmatize
            tokens = word_tokenize(text.lower())
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            # Count positive and negative keywords
            positive_count = sum(1 for token in lemmatized_tokens if token in self.positive_keywords)
            negative_count = sum(1 for token in lemmatized_tokens if token in self.negative_keywords)
            
            # Calculate sentiment score
            total_keywords = positive_count + negative_count
            if total_keywords > 0:
                sentiment_score = (positive_count - negative_count) / total_keywords
            else:
                sentiment_score = 0.0
            
            return {
                'positive_count': positive_count,
                'negative_count': negative_count,
                'sentiment_score': sentiment_score,
                'total_keywords': total_keywords
            }
            
        except Exception as e:
            logger.error(f"Error analyzing financial keywords: {str(e)}")
            return {'positive_count': 0, 'negative_count': 0, 'sentiment_score': 0.0}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis using multiple approaches.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with comprehensive sentiment analysis
        """
        try:
            if not text or not isinstance(text, str):
                return self._empty_sentiment_result()
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return self._empty_sentiment_result()
            
            # Analyze with different models
            finbert_scores = self.analyze_with_finbert(processed_text)
            vader_scores = self.analyze_with_vader(processed_text)
            textblob_scores = self.analyze_with_textblob(processed_text)
            keyword_scores = self.analyze_financial_keywords(processed_text)
            
            # Combine scores
            combined_sentiment = self._combine_sentiment_scores(
                finbert_scores, vader_scores, textblob_scores, keyword_scores
            )
            
            result = {
                'text': text[:200] + "..." if len(text) > 200 else text,
                'processed_text': processed_text[:200] + "..." if len(processed_text) > 200 else processed_text,
                'finbert': finbert_scores,
                'vader': vader_scores,
                'textblob': textblob_scores,
                'keywords': keyword_scores,
                'combined': combined_sentiment,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return self._empty_sentiment_result()
    
    def _combine_sentiment_scores(self, finbert: Dict, vader: Dict, 
                                textblob: Dict, keywords: Dict) -> Dict[str, float]:
        """
        Combine sentiment scores from different models.
        
        Args:
            finbert: FinBERT scores
            vader: VADER scores
            textblob: TextBlob scores
            keywords: Keyword-based scores
            
        Returns:
            Combined sentiment scores
        """
        try:
            # Weights for different models
            finbert_weight = 0.4
            vader_weight = 0.3
            textblob_weight = 0.2
            keyword_weight = 0.1
            
            # Calculate weighted positive sentiment
            positive_score = (
                finbert_weight * finbert.get('positive', 0.0) +
                vader_weight * vader.get('positive', 0.0) +
                textblob_weight * max(0, textblob.get('polarity', 0.0)) +
                keyword_weight * max(0, keywords.get('sentiment_score', 0.0))
            )
            
            # Calculate weighted negative sentiment
            negative_score = (
                finbert_weight * finbert.get('negative', 0.0) +
                vader_weight * vader.get('negative', 0.0) +
                textblob_weight * max(0, -textblob.get('polarity', 0.0)) +
                keyword_weight * max(0, -keywords.get('sentiment_score', 0.0))
            )
            
            # Calculate neutral sentiment
            neutral_score = (
                finbert_weight * finbert.get('neutral', 0.0) +
                vader_weight * vader.get('neutral', 0.0) +
                textblob_weight * (1 - abs(textblob.get('polarity', 0.0))) +
                keyword_weight * (1 - abs(keywords.get('sentiment_score', 0.0)))
            )
            
            # Normalize scores
            total = positive_score + negative_score + neutral_score
            if total > 0:
                positive_score /= total
                negative_score /= total
                neutral_score /= total
            
            # Calculate overall sentiment
            overall_sentiment = positive_score - negative_score
            
            # Determine sentiment label
            if overall_sentiment > 0.1:
                sentiment_label = 'positive'
            elif overall_sentiment < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'positive': positive_score,
                'negative': negative_score,
                'neutral': neutral_score,
                'overall': overall_sentiment,
                'label': sentiment_label,
                'confidence': max(positive_score, negative_score, neutral_score)
            }
            
        except Exception as e:
            logger.error(f"Error combining sentiment scores: {str(e)}")
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'overall': 0.0,
                'label': 'neutral',
                'confidence': 0.0
            }
    
    def _empty_sentiment_result(self) -> Dict[str, Any]:
        """
        Return empty sentiment analysis result.
        
        Returns:
            Empty sentiment result
        """
        return {
            'text': '',
            'processed_text': '',
            'finbert': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
            'vader': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0},
            'textblob': {'polarity': 0.0, 'subjectivity': 0.0},
            'keywords': {'positive_count': 0, 'negative_count': 0, 'sentiment_score': 0.0},
            'combined': {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'overall': 0.0,
                'label': 'neutral',
                'confidence': 0.0
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        try:
            results = []
            
            for text in texts:
                result = self.analyze_sentiment(text)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}")
            return [self._empty_sentiment_result() for _ in texts]
    
    def get_sentiment_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from sentiment analysis results.
        
        Args:
            results: List of sentiment analysis results
            
        Returns:
            Summary statistics
        """
        try:
            if not results:
                return {
                    'total_texts': 0,
                    'avg_sentiment': 0.0,
                    'positive_ratio': 0.0,
                    'negative_ratio': 0.0,
                    'neutral_ratio': 0.0,
                    'avg_confidence': 0.0
                }
            
            # Extract combined sentiment scores
            sentiments = [r['combined']['overall'] for r in results if 'combined' in r]
            confidences = [r['combined']['confidence'] for r in results if 'combined' in r]
            labels = [r['combined']['label'] for r in results if 'combined' in r]
            
            # Calculate statistics
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Count sentiment labels
            positive_count = labels.count('positive')
            negative_count = labels.count('negative')
            neutral_count = labels.count('neutral')
            total_count = len(labels)
            
            return {
                'total_texts': total_count,
                'avg_sentiment': avg_sentiment,
                'positive_ratio': positive_count / total_count if total_count > 0 else 0.0,
                'negative_ratio': negative_count / total_count if total_count > 0 else 0.0,
                'neutral_ratio': neutral_count / total_count if total_count > 0 else 0.0,
                'avg_confidence': avg_confidence,
                'sentiment_distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment summary: {str(e)}")
            return {
                'total_texts': 0,
                'avg_sentiment': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'avg_confidence': 0.0
            }

