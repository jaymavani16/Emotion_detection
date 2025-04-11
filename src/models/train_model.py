import numpy as np
import pandas as pd

import pickle
import os

from sklearn.ensemble import GradientBoostingClassifier

import yaml
import sys


def load_params(params_path: str) -> float:
    """
    Loads n_estimators and learning_rate from a YAML config file.

    Args:
        params_path (str): Path to the params.yaml file.

    Returns:
        Tuple[float, int]: test_size and random_state values.
    """
    try:
        with open(params_path,'r') as file:
                params = yaml.safe_load(file)
        n_estimators = params['train_model']['n_estimators']
        learning_rate = params['train_model']['learning_rate']
        return n_estimators,learning_rate    
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameters file not found at: {params_path}")
    except KeyError as e:
        raise KeyError(f"Missing key in YAML config: {e}")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error while loading parameters: {e}")

# fetching the data
try:
    train_data = pd.read_csv('./data/features/train_bow.csv')
except FileNotFoundError as e:
    print(f"Data file not found: {e}")
except Exception as e:
    print(f"Error loading data: {e}")


def data_splitiing(train_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits the DataFrame into features and labels.

    Args:
        train_data (pd.DataFrame): Training data with features and labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) and labels (y).

    Raises:
        ValueError: If the DataFrame is empty or has fewer than 2 columns.
    """
    try:
        if train_data.empty:
            raise ValueError("Input DataFrame is empty.")

        if train_data.shape[1] < 2:
            raise ValueError("DataFrame must have at least two columns: features and label.")

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        return X_train, y_train
    except Exception as e:
        print(f"Error in data_splitiing: {e}")
        raise


def train_gradient_model(X_train: np.ndarray, y_train: np.ndarray, 
                         n_estimators: int = 100, learning_rate: float = 0.1) -> GradientBoostingClassifier:
    """
    Trains a Gradient Boosting Classifier on the provided data.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        n_estimators (int): Number of boosting stages.
        learning_rate (float): Learning rate shrinks the contribution of each tree.

    Returns:
        GradientBoostingClassifier: Trained model.

    Raises:
        ValueError: If input arrays are empty or mismatched.
    """
    try:
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("Training data cannot be empty.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Mismatched number of samples between features and labels.")
        
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        print(f"Error during model training: {e}")
        raise


# save the model in model.pkl file.
def save_model(model: GradientBoostingClassifier, output_path: str = 'model.pkl') -> None:
    """
    Saves the trained model to the specified path using pickle.

    Args:
        model (GradientBoostingClassifier): Trained model to be saved.
        output_path (str): File path to save the model.

    Raises:
        Exception: If saving the model fails.
    """
    try:
        dir_path = os.path.dirname(output_path)
        if dir_path:  # Only try to create if it's not an empty string
            os.makedirs(dir_path, exist_ok=True)

        with open(output_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {output_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")
        raise



def main():
    try:
        n_estimators, learning_rate = load_params(params_path='params.yaml')
        X_train, y_train = data_splitiing(train_data=train_data)
        clf = train_gradient_model(
            X_train=X_train,
            y_train=y_train,
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
        save_model(model=clf)
        print("Model training and saving completed successfully.")
    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
        sys.exit(1)
    except ValueError as val_error:
        print(f"Value error during training: {val_error}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred in main(): {e}")
        sys.exit(1)


if __name__ == '__main__':
     main()
