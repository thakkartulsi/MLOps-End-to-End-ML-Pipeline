import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_building.log')
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

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('An unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """Trains the RandomForest model
        :param X_train: Training features
        :param y_train: Training labels
        :param params: Dictionary of hyperparameters
        :return: Trained RandomForestClassifier
    """

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('The number of samples in X_train and y_train must be same')
        
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        classifier = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        logger.debug('Model training started with %d samples', X_train.shape[0])
        classifier.fit(X_train, y_train)
        logger.debug('Model has been trained successfully')
        return classifier
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('An unexpected error occurred during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Saves the trained model into a file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved successfully to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('An unexpected error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # params = {'n_estimators': 30, 'random_state':1}
        params = load_params('params.yaml')['model_building']
        train_data = load_data('./data/transformed/train_tfidf.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:, -1].values

        classifier = train_model(X_train,y_train, params)

        saved_model_path = 'models/model.pkl'
        save_model(classifier, saved_model_path)

    except Exception as e:
        logger.error('Model Building process failed: %s', e)
        print(f'An unexpected error occurred: {e}')


if __name__ == '__main__':
    main()