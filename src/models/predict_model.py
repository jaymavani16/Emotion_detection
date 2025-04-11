import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import json

# loading the data 
try:
    test_data = pd.read_csv('./data/features/test_bow.csv')
except FileNotFoundError as e:
    print(f"Data file not found: {e}")
except Exception as e:
    print(f"Error loading data: {e}")


def test_data_splitting(test_data: pd.DataFrame) -> tuple:
    try:
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        return X_test, y_test
    except IndexError as idx_error:
        print(f"Indexing error: {idx_error}")
        return None, None
    except AttributeError as attr_error:
        print(f"Attribute error: {attr_error}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred in test_data_splitting: {e}")
        return None, None


# loading our model.
def load_model(output_path: str = 'models/model.pkl') -> GradientBoostingClassifier:
    try:
        with open(output_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        print(f"Model file not found at path: {output_path}")
    except pickle.UnpicklingError:
        print(f"Error unpickling the model file: {output_path}")
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")


# making prediction.
def model_predict(X_test: np.ndarray, model: GradientBoostingClassifier) -> tuple[np.ndarray, np.ndarray]:
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        return y_pred, y_pred_proba
    except AttributeError as e:
        print(f"Model object is invalid or not fitted: {e}")
    except ValueError as e:
        print(f"Invalid input data for prediction: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")



# Calculate evaluation metrics
def matrics_calculation(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> tuple[float, float, float, float]:
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        return accuracy, precision, recall, auc
    except ValueError as e:
        print(f"Value error in metrics calculation: {e}")
        return 0.0, 0.0, 0.0, 0.0
    except Exception as e:
        print(f"Unexpected error during metrics calculation: {e}")
        return 0.0, 0.0, 0.0, 0.0


# dumping file in json
def dump_json(
    accuracy: float,
    precision: float,
    recall: float,
    auc: float,
    output_path: str = 'reports/metrics.json'
) -> dict[str, float]:
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

    try:
        with open(output_path, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
    except (IOError, TypeError) as e:
        print(f"Error while saving metrics to JSON: {e}")
        return {}
    
    return metrics_dict


def main():
    try:
        X_test, y_test = test_data_splitting(test_data=test_data)
    except Exception as e:
        print(f"Error during test data splitting: {e}")
        return

    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    try:
        y_pred, y_pred_proba = model_predict(X_test=X_test, model=model)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    try:
        accuracy, precision, recall, auc = matrics_calculation(
            y_test=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba
        )
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return

    try:
        dump_json(accuracy=accuracy, precision=precision, recall=recall, auc=auc)
    except Exception as e:
        print(f"Error saving metrics to JSON: {e}")
        return


if __name__ == '__main__':
    main()