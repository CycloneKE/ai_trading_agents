"""
NLP Manager for coordinating text processing and sentiment analysis.
Handles financial news, earnings calls, analyst reports, and social media sentiment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from sentiment_analyzer import FinancialSentimentAnalyzer
from text_processor import FinancialTextProcessor

logger = logging.getLogger(__name__)


class NLPManager:
    """
    Manager for coordinating NLP tasks in financial trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        sentiment_config = config.get('sentiment_analyzer', {})
        processor_config = config.get('text_processor', {})
        
        self.sentiment_analyzer = FinancialSentimentAnalyzer(sentiment_config)
        self.text_processor = FinancialTextProcessor(processor_config)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        self.lock = threading.Lock()
        
        # Data storage
        self.processed_articles = []
        self.sentiment_history = []
        self.entity_tracking = {}
        
        # Configuration
        self.max_history_size = config.get('max_history_size', 1000)
        self.sentiment_threshold = config.get('sentiment_threshold', 0.1)
        
        logger.info("NLP Manager initialized")
    
    def process_news_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of news articles for sentiment and content analysis.
        
        Args:
            articles: List of article dictionaries with 'title', 'content', etc.
            
        Returns:
            Dict with processed results and aggregated insights
        """
        try:
            if not articles:
                return self._empty_news_result()
            
            # Process articles in parallel
            futures = []
            for article in articles:
                future = self.executor.submit(self._process_single_article, article)
                futures.append(future)
            
            # Collect results
            processed_articles = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    if result:
                        processed_articles.append(result)
                except Exception as e:
                    logger.error(f"Error processing article: {str(e)}")
            
            # Aggregate insights
            aggregated_insights = self._aggregate_news_insights(processed_articles)
            
            # Store results
            with self.lock:
                self.processed_articles.extend(processed_articles)
                if len(self.processed_articles) > self.max_history_size:
                    self.processed_articles = self.processed_articles[-self.max_history_size:]
            
            return {
                'processed_articles': processed_articles,
                'aggregated_insights': aggregated_insights,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'total_articles': len(processed_articles)
            }
            
        except Exception as e:
            logger.error(f"Error processing news articles: {str(e)}")
            return self._empty_news_result()
    
    def _process_single_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single news article.
        
        Args:
            article: Article dictionary
            
        Returns:
            Processed article data
        """
        try:
            # Extract text content
            title = article.get('title', '')
            content = article.get('content', '') or article.get('description', '')
            
            if not title and not content:
                return None
            
            # Combine title and content
            full_text = f"{title}. {content}".strip()
            
            # Process text
            text_analysis = self.text_processor.process_document(full_text, include_summary=True)
            
            # Analyze sentiment
            sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(full_text)
            
            # Extract metadata
            metadata = {
                'url': article.get('url', ''),
                'source': article.get('source', ''),
                'author': article.get('author', ''),
                'published_at': article.get('published_at', ''),
                'original_timestamp': article.get('timestamp', '')
            }
            
            # Combine results
            result = {
                'metadata': metadata,
                'text_analysis': text_analysis,
                'sentiment_analysis': sentiment_analysis,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing single article: {str(e)}")
            return None
    
    def _aggregate_news_insights(self, processed_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate insights from processed articles.
        
        Args:
            processed_articles: List of processed article results
            
        Returns:
            Aggregated insights
        """
        try:
            if not processed_articles:
                return {}
            
            # Aggregate sentiment
            sentiments = []
            for article in processed_articles:
                sentiment = article.get('sentiment_analysis', {}).get('combined', {})
                if sentiment:
                    sentiments.append(sentiment)
            
            sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(
                [{'combined': s} for s in sentiments]
            )
            
            # Aggregate entities
            all_entities = {
                'companies': [],
                'people': [],
                'stock_symbols': [],
                'locations': []
            }
            
            for article in processed_articles:
                entities = article.get('text_analysis', {}).get('entities', {})
                for entity_type in all_entities:
                    if entity_type in entities:
                        all_entities[entity_type].extend(entities[entity_type])
            
            # Count entity frequencies
            entity_frequencies = {}
            for entity_type, entities in all_entities.items():
                from collections import Counter
                entity_frequencies[entity_type] = dict(Counter(entities).most_common(10))
            
            # Aggregate categories
            category_scores = {}
            for category in self.text_processor.financial_keywords:
                scores = []
                for article in processed_articles:
                    categories = article.get('text_analysis', {}).get('categories', {})
                    if category in categories:
                        scores.append(categories[category])
                
                category_scores[category] = {
                    'average': np.mean(scores) if scores else 0.0,
                    'total_mentions': len([s for s in scores if s > 0])
                }
            
            # Extract key topics
            all_keywords = []
            for article in processed_articles:
                keywords = article.get('text_analysis', {}).get('keywords', [])
                all_keywords.extend([kw[0] for kw in keywords])
            
            from collections import Counter
            top_keywords = dict(Counter(all_keywords).most_common(20))
            
            return {
                'sentiment_summary': sentiment_summary,
                'entity_frequencies': entity_frequencies,
                'category_scores': category_scores,
                'top_keywords': top_keywords,
                'article_count': len(processed_articles)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating news insights: {str(e)}")
            return {}
    
    def analyze_social_media_sentiment(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment from social media posts.
        
        Args:
            posts: List of social media post dictionaries
            
        Returns:
            Social media sentiment analysis results
        """
        try:
            if not posts:
                return {
                    'sentiment_summary': {},
                    'post_sentiments': [],
                    'trending_topics': {},
                    'processing_timestamp': datetime.utcnow().isoformat()
                }
            
            # Extract text from posts
            post_texts = []
            post_metadata = []
            
            for post in posts:
                text = post.get('title', '') + ' ' + post.get('selftext', '')
                text = text.strip()
                
                if text:
                    post_texts.append(text)
                    post_metadata.append({
                        'id': post.get('id', ''),
                        'author': post.get('author', ''),
                        'score': post.get('score', 0),
                        'num_comments': post.get('num_comments', 0),
                        'created_utc': post.get('created_utc', 0),
                        'subreddit': post.get('subreddit', '')
                    })
            
            # Analyze sentiment for all posts
            sentiment_results = self.sentiment_analyzer.analyze_batch(post_texts)
            
            # Combine with metadata
            post_sentiments = []
            for i, (sentiment, metadata) in enumerate(zip(sentiment_results, post_metadata)):
                post_sentiments.append({
                    'metadata': metadata,
                    'sentiment': sentiment,
                    'text_preview': post_texts[i][:100] + "..." if len(post_texts[i]) > 100 else post_texts[i]
                })
            
            # Calculate weighted sentiment (by score/engagement)
            weighted_sentiments = []
            for post_sentiment in post_sentiments:
                sentiment_score = post_sentiment['sentiment']['combined']['overall']
                weight = max(1, post_sentiment['metadata']['score'])  # Use post score as weight
                weighted_sentiments.append(sentiment_score * weight)
            
            # Get overall sentiment summary
            sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(sentiment_results)
            
            # Add weighted sentiment
            if weighted_sentiments:
                sentiment_summary['weighted_avg_sentiment'] = np.average(
                    [s['combined']['overall'] for s in sentiment_results],
                    weights=[max(1, p['metadata']['score']) for p in post_sentiments]
                )
            else:
                sentiment_summary['weighted_avg_sentiment'] = 0.0
            
            # Extract trending topics
            trending_topics = self._extract_trending_topics(post_texts)
            
            # Store sentiment history
            with self.lock:
                self.sentiment_history.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'sentiment_summary': sentiment_summary,
                    'post_count': len(post_sentiments)
                })
                
                if len(self.sentiment_history) > self.max_history_size:
                    self.sentiment_history = self.sentiment_history[-self.max_history_size:]
            
            return {
                'sentiment_summary': sentiment_summary,
                'post_sentiments': post_sentiments,
                'trending_topics': trending_topics,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social media sentiment: {str(e)}")
            return {
                'sentiment_summary': {},
                'post_sentiments': [],
                'trending_topics': {},
                'processing_timestamp': datetime.utcnow().isoformat()
            }
    
    def _extract_trending_topics(self, texts: List[str]) -> Dict[str, int]:
        """
        Extract trending topics from social media texts.
        
        Args:
            texts: List of text content
            
        Returns:
            Dict with trending topics and frequencies
        """
        try:
            # Combine all texts
            combined_text = ' '.join(texts)
            
            # Extract keywords
            keywords = self.text_processor.extract_keywords(combined_text, top_k=20)
            
            # Extract stock symbols and financial entities
            entities = self.text_processor.extract_entities(combined_text)
            
            # Combine trending topics
            trending_topics = {}
            
            # Add keywords
            for keyword, freq in keywords:
                trending_topics[keyword] = freq
            
            # Add stock symbols
            for symbol in entities.get('stock_symbols', []):
                if symbol in trending_topics:
                    trending_topics[symbol] += 1
                else:
                    trending_topics[symbol] = 1
            
            # Sort by frequency
            trending_topics = dict(sorted(trending_topics.items(), key=lambda x: x[1], reverse=True)[:15])
            
            return trending_topics
            
        except Exception as e:
            logger.error(f"Error extracting trending topics: {str(e)}")
            return {}
    
    def track_entity_sentiment(self, entity: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Track sentiment for a specific entity over time.
        
        Args:
            entity: Entity to track (company name, stock symbol, etc.)
            time_window_hours: Time window in hours
            
        Returns:
            Entity sentiment tracking results
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Find relevant articles and posts
            relevant_items = []
            
            with self.lock:
                # Search processed articles
                for article in self.processed_articles:
                    timestamp_str = article.get('processing_timestamp', '')
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if timestamp > cutoff_time:
                            # Check if entity is mentioned
                            entities = article.get('text_analysis', {}).get('entities', {})
                            text = article.get('text_analysis', {}).get('cleaned_text', '')
                            
                            entity_mentioned = False
                            for entity_list in entities.values():
                                if entity.upper() in [e.upper() for e in entity_list]:
                                    entity_mentioned = True
                                    break
                            
                            if entity_mentioned or entity.upper() in text.upper():
                                relevant_items.append({
                                    'type': 'article',
                                    'timestamp': timestamp,
                                    'sentiment': article.get('sentiment_analysis', {}).get('combined', {}),
                                    'source': article.get('metadata', {}).get('source', '')
                                })
            
            if not relevant_items:
                return {
                    'entity': entity,
                    'time_window_hours': time_window_hours,
                    'total_mentions': 0,
                    'sentiment_trend': [],
                    'average_sentiment': 0.0,
                    'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
                }
            
            # Calculate sentiment trend
            sentiment_trend = []
            for item in sorted(relevant_items, key=lambda x: x['timestamp']):
                sentiment_score = item['sentiment'].get('overall', 0.0)
                sentiment_trend.append({
                    'timestamp': item['timestamp'].isoformat(),
                    'sentiment': sentiment_score,
                    'type': item['type'],
                    'source': item.get('source', '')
                })
            
            # Calculate statistics
            sentiments = [item['sentiment'].get('overall', 0.0) for item in relevant_items]
            average_sentiment = np.mean(sentiments) if sentiments else 0.0
            
            # Count sentiment distribution
            sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
            for item in relevant_items:
                label = item['sentiment'].get('label', 'neutral')
                if label in sentiment_distribution:
                    sentiment_distribution[label] += 1
            
            # Update entity tracking
            with self.lock:
                self.entity_tracking[entity] = {
                    'last_update': datetime.utcnow().isoformat(),
                    'total_mentions': len(relevant_items),
                    'average_sentiment': average_sentiment,
                    'sentiment_distribution': sentiment_distribution
                }
            
            return {
                'entity': entity,
                'time_window_hours': time_window_hours,
                'total_mentions': len(relevant_items),
                'sentiment_trend': sentiment_trend,
                'average_sentiment': average_sentiment,
                'sentiment_distribution': sentiment_distribution
            }
            
        except Exception as e:
            logger.error(f"Error tracking entity sentiment: {str(e)}")
            return {
                'entity': entity,
                'time_window_hours': time_window_hours,
                'total_mentions': 0,
                'sentiment_trend': [],
                'average_sentiment': 0.0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
            }
    
    def get_market_sentiment_signal(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Generate market sentiment signals for trading symbols.
        
        Args:
            symbols: List of trading symbols to analyze
            
        Returns:
            Market sentiment signals
        """
        try:
            signals = {}
            
            for symbol in symbols:
                # Track sentiment for this symbol
                sentiment_data = self.track_entity_sentiment(symbol, time_window_hours=24)
                
                # Generate signal based on sentiment
                avg_sentiment = sentiment_data['average_sentiment']
                total_mentions = sentiment_data['total_mentions']
                
                # Calculate signal strength
                if total_mentions == 0:
                    signal_strength = 0.0
                    signal_direction = 'neutral'
                elif avg_sentiment > self.sentiment_threshold:
                    signal_strength = min(1.0, abs(avg_sentiment) * (total_mentions / 10))
                    signal_direction = 'bullish'
                elif avg_sentiment < -self.sentiment_threshold:
                    signal_strength = min(1.0, abs(avg_sentiment) * (total_mentions / 10))
                    signal_direction = 'bearish'
                else:
                    signal_strength = 0.0
                    signal_direction = 'neutral'
                
                signals[symbol] = {
                    'signal_direction': signal_direction,
                    'signal_strength': signal_strength,
                    'average_sentiment': avg_sentiment,
                    'total_mentions': total_mentions,
                    'sentiment_distribution': sentiment_data['sentiment_distribution'],
                    'confidence': min(1.0, total_mentions / 20)  # Higher confidence with more mentions
                }
            
            return {
                'signals': signals,
                'generation_timestamp': datetime.utcnow().isoformat(),
                'time_window_hours': 24
            }
            
        except Exception as e:
            logger.error(f"Error generating market sentiment signals: {str(e)}")
            return {
                'signals': {symbol: {
                    'signal_direction': 'neutral',
                    'signal_strength': 0.0,
                    'average_sentiment': 0.0,
                    'total_mentions': 0,
                    'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                    'confidence': 0.0
                } for symbol in symbols},
                'generation_timestamp': datetime.utcnow().isoformat(),
                'time_window_hours': 24
            }
    
    def _empty_news_result(self) -> Dict[str, Any]:
        """
        Return empty news processing result.
        
        Returns:
            Empty news result
        """
        return {
            'processed_articles': [],
            'aggregated_insights': {},
            'processing_timestamp': datetime.utcnow().isoformat(),
            'total_articles': 0
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get NLP manager status and statistics.
        
        Returns:
            Status information
        """
        with self.lock:
            return {
                'processed_articles_count': len(self.processed_articles),
                'sentiment_history_count': len(self.sentiment_history),
                'tracked_entities_count': len(self.entity_tracking),
                'tracked_entities': list(self.entity_tracking.keys()),
                'last_processing_time': self.processed_articles[-1]['processing_timestamp'] if self.processed_articles else None,
                'config': {k: v for k, v in self.config.items() if 'key' not in k.lower()}
            }

