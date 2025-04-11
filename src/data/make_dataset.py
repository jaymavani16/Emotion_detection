import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml


def load_parameters(params_path: str) -> tuple[float, int]:
    """
    Loads test_size and random_state from a YAML config file.

    Args:
        params_path (str): Path to the params.yaml file.

    Returns:
        Tuple[float, int]: test_size and random_state values.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        test_size = params['make_dataset']['test_size']
        random_state = params['make_dataset']['random_state']
        return test_size, random_state
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameters file not found at: {params_path}")
    except KeyError as e:
        raise KeyError(f"Missing key in YAML config: {e}")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error while loading parameters: {e}")


def load_data(data_url: str) -> pd.DataFrame:
    """
    Loads a CSV file from a given URL or file path into a DataFrame.

    Args:
        data_url (str): URL or local path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        df = pd.read_csv(data_url)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file was not found at: {data_url}")
    except pd.errors.ParserError as e:
        raise Exception(f"Error parsing CSV file: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error while loading data: {e}")


def preprocessed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame by removing unnecessary columns
    and filtering for binary classification (happiness vs sadness).

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    try:
        # Drop unwanted column
        df.drop(columns=['tweet_id'], inplace=True)

        # Filter for binary sentiment classification
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]

        # Replace sentiment labels with numeric values
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        return final_df
    except KeyError as e:
        raise KeyError(f"Missing expected column: {e}")
    except Exception as e:
        raise Exception(f"Error during preprocessing: {e}")


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """
    Saves train and test DataFrames to CSV files in the specified directory.

    Args:
        data_path (str): Base directory where data will be saved.
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.

    Raises:
        Exception: If saving files fails.
    """
    try:
        full_path = os.path.join(data_path, 'raw')
        os.makedirs(full_path, exist_ok=True)

        train_file = os.path.join(full_path, 'train.csv')
        test_file = os.path.join(full_path, 'test.csv')

        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)

        print(f"Data saved successfully to: {train_file} and {test_file}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")


def main():
    try:
        test_size, random_state = load_parameters(params_path='params.yaml')
    except Exception as e:
        print(f"Failed to load parameters: {e}")
        return

    try:
        df = load_data(data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    try:
        final_df = preprocessed_data(df)
    except Exception as e:
        print(f"Failed during preprocessing: {e}")
        return

    try:
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=random_state)
    except Exception as e:
        print(f"Failed during train-test split: {e}")
        return

    try:
        save_data(data_path='data', train_data=train_data, test_data=test_data)
    except Exception as e:
        print(f"Failed to save data: {e}")
        return

    print("Data processing pipeline completed successfully.")


if __name__ == '__main__':
    main()
