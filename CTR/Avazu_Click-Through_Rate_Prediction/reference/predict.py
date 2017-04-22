from pandas import DataFrame
from sklearn.externals import joblib
import numpy as np

from utils import load_df


def create_submission(ids, predictions, filename='submission.csv'):
    submissions = np.concatenate((ids.reshape(len(ids), 1), predictions.reshape(len(predictions), 1)), axis=1)
    df = DataFrame(submissions)
    df.to_csv(filename, header=['id', 'click'], index=False)

classifier = joblib.load('classifier.pkl')
test_data_df = load_df('csv/test', training=False)
ids = test_data_df.values[0:, 0]
predictions = classifier.predict(test_data_df.values[0:, 1:])
create_submission(ids, predictions)




