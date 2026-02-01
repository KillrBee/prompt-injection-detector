"""
Simplified Feature Extractor
Combines all feature extraction methods
"""

import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from typing import List, Dict
from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from sentence_transformers import SentenceTransformer
from textstat import textstat


class FeatureExtractor:
    """Extract features for prompt injection detection"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        print("Loading models...")
        
        # Load models
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
        self.sia = SentimentIntensityAnalyzer()
        
        model_name = self.config.get('embedding_model', 'all-mpnet-base-v2')
        self.embedder = SentenceTransformer(f'sentence-transformers/{model_name}')
        
        # Standard letter frequencies
        self.std_freq = {
            'e': 0.1203, 't': 0.0910, 'a': 0.0812, 'o': 0.0768, 'i': 0.0731,
            'n': 0.0695, 's': 0.0628, 'r': 0.0602, 'h': 0.0592, 'd': 0.0432,
            'l': 0.0398, 'u': 0.0288, 'c': 0.0271, 'm': 0.0261, 'f': 0.0230,
            'y': 0.0211, 'w': 0.0209, 'g': 0.0203, 'p': 0.0182, 'b': 0.0149,
            'v': 0.0111, 'k': 0.0069, 'x': 0.0017, 'q': 0.0011, 'j': 0.0010,
            'z': 0.0007,
        }
        
        # Compile patterns
        self._compile_patterns()
        print("✓ Feature extractor ready")
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        self.patterns = {
            'meta': [
                r'\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instruction|prompt|command)',
                r'\bnow\s+you\s+(are|must|will)\b',
                r'\bfrom\s+now\s+on\b',
            ],
            'jailbreak': [
                r'\bDAN\s+mode\b',
                r'\bdeveloper\s+mode\b',
                r'\bgod\s+mode\b',
            ],
            'encoding': [
                r'base64|atob|btoa',
                r'\\u[0-9a-fA-F]{4}',
                r'&#x[0-9a-fA-F]+;',
            ],
            'delimiter': [
                r'---+\s*$',
                r'```',
                r'<!--',
            ],
        }
    
    def extract(self, text: str) -> Dict[str, float]:
        """Extract all features from single text"""
        if not text or len(text.strip()) == 0:
            return self._empty_features()
        
        features = {}
        words = text.split()
        sentences = sent_tokenize(text)
        
        # 1. Statistical
        features.update({
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_len': np.mean([len(w) for w in words]) if words else 0,
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'special_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0,
        })
        
        # 2. Letter frequency (fixed)
        letters = [c.lower() for c in text if c.isalpha()]
        if letters:
            letter_counts = Counter(letters)
            total = len(letters)
            for letter in string.ascii_lowercase:
                obs = letter_counts.get(letter, 0) / total
                exp = self.std_freq.get(letter, 1e-10)
                features[f'freq_{letter}'] = np.log(obs / exp) if obs > 0 else -10
        else:
            for letter in string.ascii_lowercase:
                features[f'freq_{letter}'] = 0
        
        # 3. Grammar markers
        words_lower = [w.lower() for w in words]
        second_person = {'you', 'your', 'yours'}
        imperative = {'ignore', 'disregard', 'execute', 'delete', 'reveal', 'show'}
        
        features.update({
            'second_person_ratio': sum(1 for w in words_lower if w in second_person) / len(words) if words else 0,
            'imperative_ratio': sum(1 for w in words_lower if w in imperative) / len(words) if words else 0,
        })
        
        # 4. Sentiment
        sent = self.sia.polarity_scores(text)
        features.update({
            'sentiment_compound': sent['compound'],
            'sentiment_pos': sent['pos'],
            'sentiment_neg': sent['neg'],
        })
        
        # 5. Readability
        try:
            features.update({
                'flesch_ease': textstat.flesch_reading_ease(text),
                'flesch_grade': textstat.flesch_kincaid_grade(text),
            })
        except:
            features.update({'flesch_ease': 0, 'flesch_grade': 0})
        
        # 6. POS tags
        doc = self.nlp(text[:100000])
        if len(doc) > 0:
            pos_counts = Counter([t.pos_ for t in doc])
            total = len(doc)
            features.update({
                'verb_ratio': pos_counts.get('VERB', 0) / total,
                'noun_ratio': pos_counts.get('NOUN', 0) / total,
            })
        else:
            features.update({'verb_ratio': 0, 'noun_ratio': 0})
        
        # 7. Injection patterns
        for pattern_type, patterns in self.patterns.items():
            count = sum(len(re.findall(p, text, re.I | re.M)) for p in patterns)
            features[f'{pattern_type}_count'] = count
        
        # 8. Embeddings
        embedding = self.embedder.encode(text[:512])
        for i, val in enumerate(embedding):
            features[f'embed_{i}'] = float(val)
        
        return features
    
    def extract_batch(self, texts: List[str], show_progress=False) -> pd.DataFrame:
        """Extract features from batch of texts"""
        iterator = tqdm(texts) if show_progress else texts
        features_list = [self.extract(text) for text in iterator]
        return pd.DataFrame(features_list)
    
    def _empty_features(self):
        """Return zero features"""
        features = {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_len': 0, 'caps_ratio': 0, 'digit_ratio': 0, 'special_ratio': 0,
            'second_person_ratio': 0, 'imperative_ratio': 0,
            'sentiment_compound': 0, 'sentiment_pos': 0, 'sentiment_neg': 0,
            'flesch_ease': 0, 'flesch_grade': 0,
            'verb_ratio': 0, 'noun_ratio': 0,
        }
        
        for letter in string.ascii_lowercase:
            features[f'freq_{letter}'] = 0
        
        for pattern_type in ['meta', 'jailbreak', 'encoding', 'delimiter']:
            features[f'{pattern_type}_count'] = 0
        
        embedding_dim = self.embedder.get_sentence_embedding_dimension()
        for i in range(embedding_dim):
            features[f'embed_{i}'] = 0
        
        return features
