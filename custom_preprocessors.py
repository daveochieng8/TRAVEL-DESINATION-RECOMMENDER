# custom_preprocessor.py

import re
import string
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import spacy

# Load spaCy language model
nlp = spacy.load('en_core_web_sm')

class PreprocessText(BaseEstimator, TransformerMixin):
    """
    Custom transformer for text preprocessing using spaCy.
    Handles text cleaning, normalization, and lemmatization.
    """
    
    def __init__(self, text_column='Description'):
        """
        Initialize the preprocessor.
        
        Parameters:
        text_column (str): Name of the column containing text data when input is DataFrame
        """
        self.text_column = text_column
        
    def fit(self, X, y=None):
        """Nothing to fit, return self."""
        return self
        
    def transform(self, X):
        """
        Transform the input text data.
        
        Parameters:
        X : array-like or pd.Series
            The input text data to preprocess
            
        Returns:
        pd.Series : The preprocessed text data
        """
        # Convert input to pandas Series if it isn't already
        if isinstance(X, np.ndarray):
            X = pd.Series(X)
        elif isinstance(X, pd.DataFrame):
            X = X[self.text_column]
        elif not isinstance(X, pd.Series):
            raise ValueError("Input must be numpy array, pandas Series, or DataFrame")
            
        def preprocess_text(text):
            """Helper function to preprocess individual text"""
            try:
                # Convert to string and lowercase
                text = str(text).lower()
                
                # Remove punctuation and digits
                text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
                text = re.sub(r"\w*\d\w*", "", text)
                
                # Lemmatize using spaCy
                doc = nlp(text)
                lemmatized = ' '.join([token.lemma_ for token in doc if not token.is_stop])
                
                return lemmatized
            except Exception as e:
                print(f"Error preprocessing text: {str(e)}")
                return ""
                
        # Apply preprocessing to each text
        print(f"Starting preprocessing of {len(X)} texts...")
        preprocessed = X.apply(preprocess_text)
        print("Preprocessing completed.")
        
        # Verify output
        if len(preprocessed) != len(X):
            raise ValueError(f"Preprocessing changed data length: {len(X)} → {len(preprocessed)}")
            
        return preprocessed