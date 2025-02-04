import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming"""
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

def preprocess_df(df, text_col='text', target_col='target'):
    """Preprocesses the dataFrame by encoding the target column, removing duplicates, and transforming the text column"""
    try:
        logger.debug('Dataframe Preprocessing has been started')
        encoder = LabelEncoder()
        df[target_col] = encoder.fit_transform(df[target_col])
        logger.debug('Target column encoded')

        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates has been removed successfully')

        df.loc[:, text_col] = df[text_col].apply(transform_text)
        logger.debug('Text transformation completed successfully')
        return df
    except KeyError as e:
        logger.error('Column missing in the dataframe: %s',e)
        raise
    except Exception as e:
        logger.error('An unexpected error occurred during text normalization: %s', e)
        raise

def main(text_col='text', target_col='target'):
    """Loads raw data, preprocesses it, and saves the processed data"""
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')

        train_processed_data = preprocess_df(train_data, text_col, target_col)
        test_processed_data = preprocess_df(test_data, text_col, target_col)

        data_path = os.path.join('./data', 'processed')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

        logger.debug('Processed Train and Test data saved successfully to %s', data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('No data found: %s', e)
    except Exception as e:
        logger.error('Data Transformation process failed: %s', e)
        print(f'An unexpected error occurred: {e}')

if __name__ == '__main__':
    main()
