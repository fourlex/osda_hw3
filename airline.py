import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def load_airline(train_size=5000, test_size=200):
    train = pd.read_csv('./data/airline_satisfaction_train.csv', index_col=0).set_index('id').sample(train_size)
    X_train = train.drop(columns=['satisfaction'])
    y_train = pd.get_dummies(train['satisfaction'], drop_first=True, dtype=float).iloc[:, 0]
    test = pd.read_csv('./data/airline_satisfaction_test.csv', index_col=0).set_index('id').sample(test_size)
    X_test = test.drop(columns=['satisfaction'])
    y_test = pd.get_dummies(test['satisfaction'], drop_first=True, dtype=float).iloc[:, 0]
    return X_train, X_test, y_train, y_test


def transformers_airline():
    transformers = [
        make_column_transformer(
            (SimpleImputer(strategy='constant', fill_value=0), [
                'Arrival Delay in Minutes'
            ]),
            verbose_feature_names_out=False,
            remainder='passthrough'
        ),
        make_column_transformer(
            (OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), [
                'Gender',
                'Customer Type',
                'Type of Travel',
                'Class',
                'Inflight wifi service',
                'Departure/Arrival time convenient',
                'Ease of Online booking',
                'Gate location',
                'Food and drink',
                'Online boarding',
                'Seat comfort',
                'Inflight entertainment',
                'On-board service',
                'Leg room service',
                'Baggage handling',
                'Checkin service',
                'Inflight service',
                'Cleanliness'
            ]),
            (KBinsDiscretizer(random_state=0, n_bins=5, encode='onehot-dense', subsample=None, strategy='uniform'), [
                'Age',
                'Flight Distance',
                'Departure Delay in Minutes',
                'Arrival Delay in Minutes'
            ]),
        ),
    ]

    for transformer in transformers:
        transformer.set_output(transform='pandas')

    return transformers