import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer, KBinsDiscretizer, OneHotEncoder


def load_hotel():
    hotel_booking = pd.read_csv('./data/hotel_booking.csv')
    hotel_booking_X = hotel_booking.drop(columns=['is_canceled', 'reservation_status'])
    hotel_booking_y = hotel_booking['is_canceled']
    return train_test_split(hotel_booking_X, hotel_booking_y, test_size=0.2, random_state=0)


def transformers_hotel():
    transformers = [
        make_column_transformer(
            (SimpleImputer(strategy='constant', fill_value=0), ['children']),
            verbose_feature_names_out=False,
            remainder='passthrough'
        ),
        make_column_transformer(
            (OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), [
                'deposit_type',
                'customer_type',
                'arrival_date_year',
                'market_segment',
                'arrival_date_month',
                'arrival_date_day_of_month',
                'meal'
            ]),
            (Binarizer(threshold=3), ['stays_in_week_nights']),
            (KBinsDiscretizer(n_bins=5, encode='onehot-dense', subsample=None, strategy='uniform', random_state=0), ['lead_time']),
            (Binarizer(threshold=1), [
                'total_of_special_requests',
                'days_in_waiting_list',
                'adults',
                'children',
                'babies',
                'booking_changes'
            ]),
            ('drop', ['reservation_status_date', 'name', 'email', 'phone-number', 'credit_card', 'adr', 'required_car_parking_spaces', 'arrival_date_week_number', 'stays_in_weekend_nights', 'hotel', 'country', 'distribution_channel', 'is_repeated_guest', 'previous_bookings_not_canceled', 'previous_cancellations']),
            remainder='drop'
        )
    ]

    for transformer in transformers:
        transformer.set_output(transform='pandas')

    return transformers