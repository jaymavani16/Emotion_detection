import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml

# fetch the data from /data/processed
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

def load_params(params_path: str) -> float:
    try:
        with open(params_path,'r') as file:
                params = yaml.safe_load(file)
        max_features = params['build_features']['max_features']
        return max_features
    except FileNotFoundError:
        raise FileNotFoundError(f"🔴 Parameters file not found at: {params_path}")

    except KeyError as e:
        raise KeyError(f"🔴 Missing key in YAML config: {e}")

    except yaml.YAMLError as e:
        raise Exception(f"🔴 Error parsing YAML file: {e}")

    except Exception as e:
        raise Exception(f"🔴 Unexpected error while loading parameters: {e}")
    
# filling the missing values, It is generally a part of data preprocessing but we are practicing pipeline building so i am doing it here.
try:
    train_data.fillna('', inplace=True)
    test_data.fillna('', inplace=True)
    print("Missing values successfully replaced with empty strings.")
except Exception as e:
    print(f"Error occurred while filling missing values: {e}")


def split_features_and_labels(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    text_column: str = 'content',
    label_column: str = 'sentiment'
) -> tuple:
    """
    Splits train and test DataFrames into features (X) and labels (y).

    Args:
        train_data (pd.DataFrame): Processed training data.
        test_data (pd.DataFrame): Processed test data.
        text_column (str): Column name containing text data.
        label_column (str): Column name containing target labels.

    Returns:
        tuple: X_train, y_train, X_test, y_test (all numpy arrays)
    """
    try:
        X_train = train_data[text_column].values
        y_train = train_data[label_column].values
        X_test = test_data[text_column].values
        y_test = test_data[label_column].values
        return X_train, y_train, X_test, y_test

    except KeyError as e:
        raise KeyError(f"Missing column in data: {e}")
    except AttributeError:
        raise TypeError("Input data must be pandas DataFrames.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")



# Apply Bag of Words (CountVectorizer)
def apply_bow_vectorization(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    max_features: int = 1000
) -> tuple:
    """
    Applies Bag of Words vectorization using CountVectorizer.

    Args:
        X_train (array-like): Training text data.
        X_test (array-like): Test text data.
        y_train (array-like): Training labels.
        y_test (array-like): Test labels.
        max_features (int): Maximum number of features for vectorizer.

    Returns:
        tuple: train_df, test_df (vectorized DataFrames with labels), and the fitted CountVectorizer object.
    """
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        return train_df, test_df, vectorizer

    except ValueError as e:
        raise ValueError(f"Vectorization failed: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during vectorization: {e}")


def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Saves the train and test DataFrames to CSV files in a 'features' subdirectory.

    Args:
        data_path (str): Base directory to save the data.
        train_df (pd.DataFrame): DataFrame containing training features and labels.
        test_df (pd.DataFrame): DataFrame containing test features and labels.
    """
    try:
        data_path = os.path.join(data_path, 'features')
        os.makedirs(data_path, exist_ok=True)

        train_csv_path = os.path.join(data_path, 'train_bow.csv')
        test_csv_path = os.path.join(data_path, 'test_bow.csv')

        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)

        print(f"Data successfully saved to {data_path}")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Directory not found: {e}")
    except PermissionError as e:
        raise PermissionError(f"Permission denied when writing files: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while saving data: {e}")


def main():
    try:
        max_features = load_params(params_path='params.yaml')
        X_train, y_train, X_test, y_test = split_features_and_labels(
            train_data=train_data,
            test_data=test_data
        )
        train_df, test_df, vectorizer = apply_bow_vectorization(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            max_features=max_features
        )
        save_data(data_path='data', train_df=train_df, test_df=test_df)
        print("BoW feature extraction and saving completed successfully.")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except KeyError as e:
        print(f"Missing column or key: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
     main()