import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Download necessary NLTK resources
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
except Exception as e:
    print(f"Failed to download NLTK resources: {e}")

# Load the data
try:
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')
except FileNotFoundError as e:
    print(f"Data file not found: {e}")
except Exception as e:
    print(f"Error loading data: {e}")


def lemmatization(text: str) -> str:
    """
    Lemmatizes the input text using NLTK's WordNetLemmatizer.

    Args:
        text (str): The text string to be lemmatized.

    Returns:
        str: A lemmatized version of the input text.
    """
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = text.split()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmatized_tokens)
    except Exception as e:
        print(f"Error during lemmatization: {e}")
        return text  # Return original text if error occurs


def remove_stop_words(text: str) -> str:
    """
    Removes English stopwords from the input text.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text without stopwords.
    """
    try:
        stop_words = set(stopwords.words("english"))
        words = str(text).lower().split()  # Lowercase before removing
        filtered_words = [word for word in words if word not in stop_words]
        return " ".join(filtered_words)
    except Exception as e:
        print(f"Error in removing stop words: {e}")
        return text


def removing_numbers(text: str) -> str:
    try:
        return ''.join([i for i in text if not i.isdigit()])
    except Exception as e:
        print(f"Error removing numbers from text: {text}\n{e}")
        return text


def lower_case(text: str) -> str:
    try:
        return " ".join([word.lower() for word in text.split()])    
    except Exception as e:
        print(f"Error converting text to lowercase: {text}\n{e}")
        return text


def removing_punctuations(text: str) -> str:
    try:
        text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', '')
        text = re.sub('\s+', ' ', text).strip()
        return text    
    except Exception as e:
        print(f"Error removing punctuations from text: {text}\n{e}")
        return text


def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)    
    except Exception as e:
        print(f"Error removing URLs from text: {text}\n{e}")
        return text


def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        words = str(text).lower().split()
        return " ".join([word for word in words if word not in stop_words])    
    except Exception as e:
        print(f"Error removing stopwords from text: {text}\n{e}")
        return text


def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])    
    except Exception as e:
        print(f"Error lemmatizing text: {text}\n{e}")
        return text


def remove_small_sentences(df: pd.DataFrame, column: str = 'text') -> pd.DataFrame:
    try:
        df[column] = df[column].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)
        return df    
    except Exception as e:
        print(f"Error removing small sentences from DataFrame column '{column}':\n{e}")
        return df


def normalize_text(df: pd.DataFrame, column: str = 'content') -> pd.DataFrame:
    try:
        df = df.copy()
        df[column] = df[column].apply(lower_case)
        df[column] = df[column].apply(remove_stop_words)
        df[column] = df[column].apply(removing_numbers)
        df[column] = df[column].apply(removing_punctuations)
        df[column] = df[column].apply(removing_urls)
        df[column] = df[column].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error normalizing DataFrame column '{column}':\n{e}")
        return df


def save_data(data_path: str, train_processed_data: pd.DataFrame, test_processed_data: pd.DataFrame) -> None:
    try:
        processed_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(processed_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(processed_path, 'test_processed.csv'), index=False)

        print("Data saved successfully in 'data/processed'")
    except Exception as e:
        print(f"Error saving processed data: {e}")


def main():
    try:
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        save_data('data', train_processed_data, test_processed_data)
    except Exception as e:
        print(f"Main function error: {e}")


if __name__ == '__main__':
    main()