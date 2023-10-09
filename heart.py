import pandas as pd
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split


def load_heart():
    heart = pd.read_csv('./data/heart_disease_health_indicators_BRFSS2015.csv')
    heart_X = heart.drop(columns=['HeartDiseaseorAttack'])
    heart_y = heart['HeartDiseaseorAttack']
    return train_test_split(heart_X, heart_y, test_size=0.2, random_state=0)


def transformers_heart():
    heart_transformer = make_column_transformer(
        (preprocessing.OneHotEncoder(drop='first', sparse_output=False), ['Diabetes']),
        (preprocessing.KBinsDiscretizer(random_state=0, subsample=None, n_bins=5, strategy='uniform', encode='onehot-dense'), ['Age', 'Education', 'Income', 'BMI', 'MentHlth', 'PhysHlth', 'GenHlth']),
        remainder='passthrough'
    )
    return [heart_transformer, ]