"""
Feature Extraction Module for Prompt Injection Detection
Combines traditional NLP features with modern transformer embeddings
"""

import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from typing import Dict, List, Any

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, pipeline
import torch

from spellchecker import SpellChecker
from textstat import textstat


class PromptInjectionFeatureExtractor:
    """
    Comprehensive feature extraction combining:
    1. Traditional NLP features (from 2023 Kaggle notebook)
    2. Modern transformer embeddings (2024-2025 research)
    3. Prompt injection specific patterns
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize all feature extractors
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config or {}
        
        # Load NLP models
        print("Loading NLP models...")
        self.nlp = spacy.load('en_core_web_sm')
        self.sia = SentimentIntensityAnalyzer()
        self.spell = SpellChecker()
        
        # Load transformer models
        print("Loading transformer models...")
        embedding_model = self.config.get('embedding_model', 'all-mpnet-base-v2')
        self.sentence_transformer = SentenceTransformer(f'sentence-transformers/{embedding_model}')
        
        # Load BERT for contextual features
        bert_model = self.config.get('bert_model', 'ProtectAI/deberta-v3-base-prompt-injection-v2')
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert_model = AutoModel.from_pretrained(bert_model)
        
        # Standard English letter frequencies
        self.standard_freq = {
            'e': 0.1203, 't': 0.0910, 'a': 0.0812, 'o': 0.0768, 'i': 0.0731,
            'n': 0.0695, 's': 0.0628, 'r': 0.0602, 'h': 0.0592, 'd': 0.0432,
            'l': 0.0398, 'u': 0.0288, 'c': 0.0271, 'm': 0.0261, 'f': 0.0230,
            'y': 0.0211, 'w': 0.0209, 'g': 0.0203, 'p': 0.0182, 'b': 0.0149,
            'v': 0.0111, 'k': 0.0069, 'x': 0.0017, 'q': 0.0011, 'j': 0.0010,
            'z': 0.0007,
        }
        
        # Grammar markers
        self.second_person_pronouns = {'you', 'your', 'yours'}
        self.first_person_pronouns = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
        self.imperative_verbs = {
            'ignore', 'disregard', 'execute', 'run', 'delete', 'reveal',
            'show', 'display', 'output', 'print', 'bypass', 'override',
            'forget', 'stop', 'remove', 'disable', 'enable', 'modify'
        }
        
        # Injection patterns (2024-2025 research)
        self._compile_injection_patterns()
        
        print("✓ Feature extractor initialized")
    
    def _compile_injection_patterns(self):
        """Compile regex patterns for injection detection"""
        self.meta_patterns = [
            r'\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above|earlier)\s+(instruction|prompt|command|rule)',
            r'\bnow\s+you\s+(are|must|should|will)\b',
            r'\bfrom\s+now\s+on\b',
            r'\bpretend\s+(you|to)\s+(are|be)\b',
            r'\broleplay\s+as\b',
            r'\bimagine\s+(you|yourself)\s+(are|as)\b',
        ]
        
        self.jailbreak_patterns = [
            r'\bDAN\s+mode\b',
            r'\bdeveloper\s+mode\b',
            r'\bgod\s+mode\b',
            r'\bunrestricted\s+mode\b',
            r'\bjailbreak\b',
        ]
        
        self.encoding_patterns = [
            r'base64|atob|btoa|fromCharCode',
            r'\\u[0-9a-fA-F]{4}',
            r'&#x[0-9a-fA-F]+;',
            r'%[0-9a-fA-F]{2}',
        ]
        
        self.delimiter_patterns = [
            r'---+\s*$',
            r'===+\s*$',
            r'```[a-z]*\n',
            r'<!--.*?-->',
            r'/\*.*?\*/',
            r'--\s',
        ]
        
        self.hierarchy_bypass = [
            r'\buser:\s*\n',
            r'\bassistant:\s*\n',
            r'\bsystem:\s*\n',
            r'\[INST\]',
            r'\[/INST\]',
        ]
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        Extract all features from text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of features
        """
        if not text or len(text.strip()) == 0:
            return self._get_empty_features()
        
        features = {}
        
        # 1. Basic statistical features
        features.update(self._extract_statistical_features(text))
        
        # 2. Letter frequency analysis (fixed from Kaggle notebook)
        features.update(self._extract_letter_frequency(text))
        
        # 3. Grammar and linguistic features
        features.update(self._extract_grammar_features(text))
        
        # 4. Readability metrics
        features.update(self._extract_readability(text))
        
        # 5. Sentiment analysis
        features.update(self._extract_sentiment(text))
        
        # 6. Spelling and obfuscation
        features.update(self._extract_spelling_features(text))
        
        # 7. POS tagging
        features.update(self._extract_pos_features(text))
        
        # 8. Injection-specific patterns
        features.update(self._extract_injection_patterns(text))
        
        # 9. Sentence transformer embeddings
        features.update(self._extract_embeddings(text))
        
        # 10. BERT contextual features (optional - expensive)
        if self.config.get('use_bert_features', False):
            features.update(self._extract_bert_features(text))
        
        return features
    
    def _extract_statistical_features(self, text: str) -> Dict[str, float]:
        """Basic statistical features"""
        words = text.split()
        sentences = sent_tokenize(text)
        lines = text.split('\n')
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'line_count': len(lines),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0,
            'whitespace_ratio': sum(1 for c in text if c.isspace()) / len(text) if text else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,  # Lexical diversity
        }
    
    def _extract_letter_frequency(self, text: str) -> Dict[str, float]:
        """
        Letter frequency analysis (FIXED from Kaggle notebook)
        Uses log-ratio for symmetric scaling
        """
        # Calculate letter frequencies
        letters = ''.join(c.lower() for c in text if c.isalpha())
        if not letters:
            return {f'freq_logratio_{k}': 0 for k in string.ascii_lowercase}
        
        letter_counts = Counter(letters)
        total = len(letters)
        observed_freq = {letter: count / total for letter, count in letter_counts.items()}
        
        # Calculate log-ratios (symmetric scaling)
        features = {}
        for letter in string.ascii_lowercase:
            obs = observed_freq.get(letter, 1e-10)
            exp = self.standard_freq.get(letter, 1e-10)
            features[f'freq_logratio_{letter}'] = np.log(obs / exp)
        
        # Cosine similarity with standard distribution
        obs_vector = np.array([observed_freq.get(l, 0) for l in string.ascii_lowercase])
        std_vector = np.array([self.standard_freq.get(l, 0) for l in string.ascii_lowercase])
        
        if np.linalg.norm(obs_vector) > 0 and np.linalg.norm(std_vector) > 0:
            cosine_sim = np.dot(obs_vector, std_vector) / (np.linalg.norm(obs_vector) * np.linalg.norm(std_vector))
        else:
            cosine_sim = 0
        
        features['letter_freq_cosine_sim'] = cosine_sim
        
        return features
    
    def _extract_grammar_features(self, text: str) -> Dict[str, float]:
        """Grammar and pronoun usage"""
        words = [w.lower() for w in text.split()]
        total = len(words) if words else 1
        
        return {
            'second_person_ratio': sum(1 for w in words if w in self.second_person_pronouns) / total,
            'first_person_ratio': sum(1 for w in words if w in self.first_person_pronouns) / total,
            'imperative_verb_ratio': sum(1 for w in words if w in self.imperative_verbs) / total,
            'command_score': (
                sum(1 for w in words if w in self.second_person_pronouns) *
                sum(1 for w in words if w in self.imperative_verbs)
            ) / (total ** 2) if total > 0 else 0,
        }
    
    def _extract_readability(self, text: str) -> Dict[str, float]:
        """Readability metrics using textstat"""
        try:
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'smog_index': textstat.smog_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'automated_readability_index': textstat.automated_readability_index(text),
            }
        except:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'smog_index': 0,
                'coleman_liau_index': 0,
                'automated_readability_index': 0,
            }
    
    def _extract_sentiment(self, text: str) -> Dict[str, float]:
        """Sentiment analysis using VADER"""
        scores = self.sia.polarity_scores(text)
        return {
            'sentiment_compound': scores['compound'],
            'sentiment_positive': scores['pos'],
            'sentiment_negative': scores['neg'],
            'sentiment_neutral': scores['neu'],
            'sentiment_extremity': abs(scores['compound']),
        }
    
    def _extract_spelling_features(self, text: str) -> Dict[str, float]:
        """Spelling errors and obfuscation detection"""
        words = text.split()
        if not words:
            return {'misspelling_ratio': 0, 'substitution_count': 0}
        
        misspelled = self.spell.unknown(words)
        
        # Check for character substitutions (l33t speak, etc.)
        substitutions = sum(1 for w in words if any(c.isdigit() for c in w))
        
        # Check for non-ASCII characters (unicode tricks)
        non_ascii = sum(1 for c in text if ord(c) > 127)
        
        return {
            'misspelling_ratio': len(misspelled) / len(words),
            'substitution_count': substitutions,
            'non_ascii_ratio': non_ascii / len(text) if text else 0,
        }
    
    def _extract_pos_features(self, text: str) -> Dict[str, float]:
        """Part-of-speech tagging features"""
        doc = self.nlp(text[:1000000])  # Limit length for performance
        if len(doc) == 0:
            return {
                'verb_ratio': 0,
                'noun_ratio': 0,
                'adj_ratio': 0,
                'verb_noun_ratio': 0,
            }
        
        pos_counts = Counter([token.pos_ for token in doc])
        total = len(doc)
        
        verb_ratio = pos_counts.get('VERB', 0) / total
        noun_ratio = pos_counts.get('NOUN', 0) / total
        adj_ratio = pos_counts.get('ADJ', 0) / total
        
        return {
            'verb_ratio': verb_ratio,
            'noun_ratio': noun_ratio,
            'adj_ratio': adj_ratio,
            'verb_noun_ratio': verb_ratio / noun_ratio if noun_ratio > 0 else 0,
        }
    
    def _extract_injection_patterns(self, text: str) -> Dict[str, float]:
        """Injection-specific pattern matching"""
        features = {}
        
        for pattern_type, patterns in [
            ('meta', self.meta_patterns),
            ('jailbreak', self.jailbreak_patterns),
            ('encoding', self.encoding_patterns),
            ('delimiter', self.delimiter_patterns),
            ('hierarchy', self.hierarchy_bypass),
        ]:
            count = sum(
                len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
                for pattern in patterns
            )
            features[f'{pattern_type}_pattern_count'] = count
            features[f'has_{pattern_type}_pattern'] = int(count > 0)
        
        return features
    
    def _extract_embeddings(self, text: str) -> Dict[str, float]:
        """Sentence transformer embeddings"""
        # Truncate to avoid memory issues
        text_truncated = text[:512] if len(text) > 512 else text
        embedding = self.sentence_transformer.encode(text_truncated)
        
        return {f'embed_{i}': float(val) for i, val in enumerate(embedding)}
    
    def _extract_bert_features(self, text: str) -> Dict[str, float]:
        """BERT contextual features (expensive, optional)"""
        inputs = self.bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        # Only use first 128 dimensions to avoid bloat
        return {f'bert_cls_{i}': float(val) for i, val in enumerate(cls_embedding[:128])}
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return zero features for empty input"""
        features = {}
        
        # Statistical
        for key in ['char_count', 'word_count', 'sentence_count', 'line_count',
                    'avg_word_length', 'avg_sentence_length', 'caps_ratio',
                    'digit_ratio', 'special_char_ratio', 'whitespace_ratio', 'unique_word_ratio']:
            features[key] = 0
        
        # Letter frequency
        for letter in string.ascii_lowercase:
            features[f'freq_logratio_{letter}'] = 0
        features['letter_freq_cosine_sim'] = 0
        
        # Grammar
        for key in ['second_person_ratio', 'first_person_ratio', 'imperative_verb_ratio', 'command_score']:
            features[key] = 0
        
        # Readability
        for key in ['flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index',
                    'coleman_liau_index', 'automated_readability_index']:
            features[key] = 0
        
        # Sentiment
        for key in ['sentiment_compound', 'sentiment_positive', 'sentiment_negative',
                    'sentiment_neutral', 'sentiment_extremity']:
            features[key] = 0
        
        # Spelling
        for key in ['misspelling_ratio', 'substitution_count', 'non_ascii_ratio']:
            features[key] = 0
        
        # POS
        for key in ['verb_ratio', 'noun_ratio', 'adj_ratio', 'verb_noun_ratio']:
            features[key] = 0
        
        # Patterns
        for pattern_type in ['meta', 'jailbreak', 'encoding', 'delimiter', 'hierarchy']:
            features[f'{pattern_type}_pattern_count'] = 0
            features[f'has_{pattern_type}_pattern'] = 0
        
        # Embeddings (768 dims for mpnet)
        embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        for i in range(embedding_dim):
            features[f'embed_{i}'] = 0
        
        return features


def extract_features_batch(texts: List[str], config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Extract features from a batch of texts
    
    Args:
        texts: List of texts to process
        config: Configuration dictionary
        
    Returns:
        DataFrame with features
    """
    extractor = PromptInjectionFeatureExtractor(config)
    
    features_list = []
    for text in texts:
        features_list.append(extractor.extract_all_features(text))
    
    return pd.DataFrame(features_list)
