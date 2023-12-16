import pandas as pd
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split


def load(train_size=5000, test_size=200):
    heart = pd.read_csv('./data/heart_disease_health_indicators_BRFSS2015.csv')
    heart_X = heart.drop(columns=['HeartDiseaseorAttack'])
    heart_y = heart['HeartDiseaseorAttack']
    return train_test_split(heart_X, heart_y, train_size=train_size, test_size=test_size, random_state=0, stratify=heart_y)

def binarizer():
    heart_transformer = make_column_transformer(
        (preprocessing.OneHotEncoder(drop='first', sparse_output=False), ['Diabetes']),
        (preprocessing.KBinsDiscretizer(random_state=0, subsample=None, n_bins=5, strategy='uniform', encode='onehot-dense'), ['Age', 'Education', 'Income', 'BMI', 'MentHlth', 'PhysHlth', 'GenHlth']),
        remainder='passthrough'
    )
    return [heart_transformer, ]