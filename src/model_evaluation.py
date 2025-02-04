import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Loads parameters from a YAML file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_model(file_path: str):
    """Loads the trained model from a file"""
    try:
        with open(file_path,'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('An unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded successfully from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('An unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(classifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluates the model and returns the evaluation metrics"""
    try:
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)[:,1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        logger.debug('Model evaluation metrics calculated successfully')
        return metrics_dict
    except Exception as e:
        logger.error('An unexpected error occurred during model evaluation: %s', e)
        raise
        
def save_metrics(metrics: dict, file_path: str) -> None:
    """Saves the evaluation metrics to a JSON file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved successfully to %s', file_path)
    except Exception as e:
        logger.error('An unexpected error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        classifier = load_model('./models/model.pkl')
        test_data = load_data('./data/transformed/test_tfidf.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(classifier, X_test, y_test) 

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params)

        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Model Evaluation process failed: %s', e)
        print(f'An unexpected error occurred: {e}')


if __name__ == '__main__':
    main()