"""
Text Processing module for financial documents and news.
Handles text extraction, cleaning, and feature extraction from various sources.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from collections import Counter
import spacy
from transformers import pipeline

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    pass


class FinancialTextProcessor:
    """
    Text processor specialized for financial documents and news.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        
        # Load spaCy model for NER
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Some features will be limited.")
        
        # Initialize summarization pipeline
        self.summarizer = None
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            logger.warning(f"Failed to load summarization model: {str(e)}")
        
        # Financial entity patterns
        self.financial_patterns = {
            'stock_symbols': re.compile(r'\b[A-Z]{1,5}\b'),
            'currency': re.compile(r'\$[\d,]+\.?\d*|\d+\s*(?:dollars?|USD|cents?)', re.IGNORECASE),
            'percentages': re.compile(r'\d+\.?\d*\s*%'),
            'dates': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
            'quarters': re.compile(r'\bQ[1-4]\s+\d{4}\b', re.IGNORECASE),
            'fiscal_years': re.compile(r'\bFY\s*\d{4}\b', re.IGNORECASE)
        }
        
        # Financial keywords
        self.financial_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'income', 'ebitda', 'eps'],
            'performance': ['growth', 'decline', 'increase', 'decrease', 'beat', 'miss', 'exceed'],
            'market': ['market', 'trading', 'volume', 'price', 'shares', 'stock', 'equity'],
            'guidance': ['guidance', 'forecast', 'outlook', 'projection', 'estimate', 'target'],
            'corporate': ['merger', 'acquisition', 'dividend', 'buyback', 'split', 'ipo'],
            'risk': ['risk', 'volatility', 'uncertainty', 'concern', 'challenge', 'headwind']
        }
        
        # Stop words
        self.stop_words = set(stopwords.words('english'))
        
        logger.info("Financial text processor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove extra punctuation
            text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\$\%]', ' ', text)
            
            # Normalize quotes
            text = re.sub(r'[""''`]', '"', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dict with extracted entities
        """
        try:
            entities = {
                'companies': [],
                'people': [],
                'locations': [],
                'stock_symbols': [],
                'currencies': [],
                'percentages': [],
                'dates': [],
                'quarters': [],
                'fiscal_years': []
            }
            
            if not text:
                return entities
            
            # Extract using regex patterns
            for pattern_name, pattern in self.financial_patterns.items():
                matches = pattern.findall(text)
                entities[pattern_name] = list(set(matches))
            
            # Extract named entities using spaCy
            if self.nlp:
                doc = self.nlp(text)
                
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'CORP']:
                        entities['companies'].append(ent.text)
                    elif ent.label_ == 'PERSON':
                        entities['people'].append(ent.text)
                    elif ent.label_ in ['GPE', 'LOC']:
                        entities['locations'].append(ent.text)
            
            # Remove duplicates and clean
            for key in entities:
                entities[key] = list(set([item.strip() for item in entities[key] if item.strip()]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {key: [] for key in ['companies', 'people', 'locations', 'stock_symbols', 
                                     'currencies', 'percentages', 'dates', 'quarters', 'fiscal_years']}
    
    def extract_keywords(self, text: str, top_k: int = 20) -> List[Tuple[str, int]]:
        """
        Extract important keywords from text.
        
        Args:
            text: Text to extract keywords from
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        try:
            if not text:
                return []
            
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            
            # Remove stop words and non-alphabetic tokens
            filtered_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalpha() and token not in self.stop_words and len(token) > 2
            ]
            
            # Count frequencies
            word_freq = Counter(filtered_tokens)
            
            # Get top keywords
            top_keywords = word_freq.most_common(top_k)
            
            return top_keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def categorize_content(self, text: str) -> Dict[str, float]:
        """
        Categorize financial content by topic.
        
        Args:
            text: Text to categorize
            
        Returns:
            Dict with category scores
        """
        try:
            if not text:
                return {category: 0.0 for category in self.financial_keywords}
            
            text_lower = text.lower()
            category_scores = {}
            
            for category, keywords in self.financial_keywords.items():
                score = 0
                for keyword in keywords:
                    # Count occurrences of each keyword
                    count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                    score += count
                
                # Normalize by text length
                category_scores[category] = score / len(text_lower.split()) if text_lower.split() else 0.0
            
            return category_scores
            
        except Exception as e:
            logger.error(f"Error categorizing content: {str(e)}")
            return {category: 0.0 for category in self.financial_keywords}
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Generate a summary of the text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Text summary
        """
        try:
            if not text or len(text.split()) < min_length:
                return text
            
            if self.summarizer:
                # Use transformer-based summarization
                try:
                    # Truncate text if too long
                    max_input_length = 1024
                    if len(text.split()) > max_input_length:
                        text = ' '.join(text.split()[:max_input_length])
                    
                    summary = self.summarizer(
                        text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    
                    return summary[0]['summary_text']
                    
                except Exception as e:
                    logger.warning(f"Transformer summarization failed: {str(e)}")
            
            # Fallback: extractive summarization
            return self._extractive_summary(text, max_length)
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def _extractive_summary(self, text: str, max_length: int) -> str:
        """
        Create extractive summary by selecting important sentences.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Extractive summary
        """
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) <= 2:
                return text
            
            # Score sentences based on keyword frequency
            sentence_scores = {}
            
            # Get important keywords
            keywords = [word for word, freq in self.extract_keywords(text, top_k=10)]
            
            for sentence in sentences:
                score = 0
                words = word_tokenize(sentence.lower())
                
                for word in words:
                    if word in keywords:
                        score += 1
                
                # Normalize by sentence length
                sentence_scores[sentence] = score / len(words) if words else 0
            
            # Select top sentences
            sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            
            summary = ""
            for sentence, score in sorted_sentences:
                if len(summary) + len(sentence) <= max_length:
                    summary += sentence + " "
                else:
                    break
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error in extractive summary: {str(e)}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def analyze_readability(self, text: str) -> Dict[str, float]:
        """
        Analyze text readability metrics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with readability metrics
        """
        try:
            if not text:
                return {
                    'avg_sentence_length': 0.0,
                    'avg_word_length': 0.0,
                    'complexity_score': 0.0
                }
            
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            # Calculate metrics
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            # Simple complexity score based on sentence and word length
            complexity_score = (avg_sentence_length * 0.6 + avg_word_length * 0.4) / 10
            
            return {
                'avg_sentence_length': avg_sentence_length,
                'avg_word_length': avg_word_length,
                'complexity_score': min(complexity_score, 1.0)  # Cap at 1.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing readability: {str(e)}")
            return {
                'avg_sentence_length': 0.0,
                'avg_word_length': 0.0,
                'complexity_score': 0.0
            }
    
    def process_document(self, text: str, include_summary: bool = True) -> Dict[str, Any]:
        """
        Comprehensive processing of a financial document.
        
        Args:
            text: Document text to process
            include_summary: Whether to include text summary
            
        Returns:
            Dict with comprehensive text analysis
        """
        try:
            if not text:
                return self._empty_processing_result()
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            if not cleaned_text:
                return self._empty_processing_result()
            
            # Extract various features
            entities = self.extract_entities(cleaned_text)
            keywords = self.extract_keywords(cleaned_text)
            categories = self.categorize_content(cleaned_text)
            readability = self.analyze_readability(cleaned_text)
            
            # Generate summary if requested
            summary = ""
            if include_summary:
                summary = self.summarize_text(cleaned_text)
            
            # Calculate basic statistics
            word_count = len(word_tokenize(cleaned_text))
            sentence_count = len(sent_tokenize(cleaned_text))
            
            result = {
                'original_text': text[:500] + "..." if len(text) > 500 else text,
                'cleaned_text': cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
                'summary': summary,
                'entities': entities,
                'keywords': keywords[:10],  # Top 10 keywords
                'categories': categories,
                'readability': readability,
                'statistics': {
                    'word_count': word_count,
                    'sentence_count': sentence_count,
                    'avg_words_per_sentence': word_count / sentence_count if sentence_count > 0 else 0
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return self._empty_processing_result()
    
    def _empty_processing_result(self) -> Dict[str, Any]:
        """
        Return empty processing result.
        
        Returns:
            Empty processing result
        """
        return {
            'original_text': '',
            'cleaned_text': '',
            'summary': '',
            'entities': {key: [] for key in ['companies', 'people', 'locations', 'stock_symbols', 
                                           'currencies', 'percentages', 'dates', 'quarters', 'fiscal_years']},
            'keywords': [],
            'categories': {category: 0.0 for category in self.financial_keywords},
            'readability': {
                'avg_sentence_length': 0.0,
                'avg_word_length': 0.0,
                'complexity_score': 0.0
            },
            'statistics': {
                'word_count': 0,
                'sentence_count': 0,
                'avg_words_per_sentence': 0
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def process_batch(self, texts: List[str], include_summary: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of documents.
        
        Args:
            texts: List of texts to process
            include_summary: Whether to include summaries
            
        Returns:
            List of processing results
        """
        try:
            results = []
            
            for text in texts:
                result = self.process_document(text, include_summary)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return [self._empty_processing_result() for _ in texts]
    
    def get_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from processing results.
        
        Args:
            results: List of processing results
            
        Returns:
            Summary statistics
        """
        try:
            if not results:
                return {
                    'total_documents': 0,
                    'total_words': 0,
                    'avg_words_per_doc': 0.0,
                    'top_entities': {},
                    'top_keywords': [],
                    'category_averages': {}
                }
            
            # Aggregate statistics
            total_words = sum(r['statistics']['word_count'] for r in results)
            total_docs = len(results)
            
            # Aggregate entities
            all_entities = {}
            for result in results:
                for entity_type, entities in result['entities'].items():
                    if entity_type not in all_entities:
                        all_entities[entity_type] = []
                    all_entities[entity_type].extend(entities)
            
            # Count entity frequencies
            top_entities = {}
            for entity_type, entities in all_entities.items():
                entity_counts = Counter(entities)
                top_entities[entity_type] = entity_counts.most_common(5)
            
            # Aggregate keywords
            all_keywords = []
            for result in results:
                all_keywords.extend([kw[0] for kw in result['keywords']])
            
            keyword_counts = Counter(all_keywords)
            top_keywords = keyword_counts.most_common(10)
            
            # Average category scores
            category_averages = {}
            for category in self.financial_keywords:
                scores = [r['categories'].get(category, 0.0) for r in results]
                category_averages[category] = np.mean(scores) if scores else 0.0
            
            return {
                'total_documents': total_docs,
                'total_words': total_words,
                'avg_words_per_doc': total_words / total_docs if total_docs > 0 else 0.0,
                'top_entities': top_entities,
                'top_keywords': top_keywords,
                'category_averages': category_averages
            }
            
        except Exception as e:
            logger.error(f"Error calculating processing summary: {str(e)}")
            return {
                'total_documents': 0,
                'total_words': 0,
                'avg_words_per_doc': 0.0,
                'top_entities': {},
                'top_keywords': [],
                'category_averages': {}
            }

