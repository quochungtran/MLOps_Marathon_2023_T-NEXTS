import argparse
import logging
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
# Data Processing
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

from problem_config import ProblemConfig, ProblemConst, get_prob_config


class RawDataProcessor:
    @staticmethod
    def build_category_features(data, categorical_cols=None):
        if categorical_cols is None:
            categorical_cols = []
        category_index = {}
        if len(categorical_cols) == 0:
            return data, category_index

        df = data.copy()
        # process category features
        for col in categorical_cols:
            df[col] = df[col].astype("category")
            category_index[col] = df[col].cat.categories
            df[col] = df[col].cat.codes
        return df, category_index

    @staticmethod
    def apply_category_features(
        raw_df, categorical_cols=None, category_index: dict = None
    ):
        if categorical_cols is None:
            categorical_cols = []
        if len(categorical_cols) == 0:
            return raw_df

        apply_df = raw_df.copy()
        for col in categorical_cols:
            apply_df[col] = apply_df[col].astype("object")
            apply_df[col] = apply_df[col].astype("category")
            apply_df[col] = apply_df[col].fillna("Unknown")     # fill nan with Unknown
            apply_df[col] = pd.Categorical(
                apply_df[col],
                categories=category_index[col],
            ).codes
        return apply_df


    @staticmethod
    def process_raw_data(prob_config: ProblemConfig):
        logging.info("start process_raw_data")
        training_data = pd.read_parquet(prob_config.raw_data_path)
        training_data, category_index = RawDataProcessor.build_category_features(
            training_data, prob_config.categorical_cols
        )
        train, dev = train_test_split(
            training_data,
            test_size=prob_config.test_size,
            random_state=prob_config.random_state,
        )

        with open(prob_config.category_index_path, "wb") as f:
            pickle.dump(category_index, f)

        target_col = prob_config.target_col        
        train_x = train.drop([target_col], axis=1)        
        y_train = train[[target_col]]
        test_x = dev.drop([target_col], axis=1)
        y_test = dev[[target_col]]

        # Encoder transform 
        encoder = ce.TargetEncoder(cols=target_col)
        X_train = encoder.fit_transform(train_x, y_train)
        X_test = encoder.fit_Transform(test_x, y_test)

        # Scaling 
        for col in list(target_col): 
            sc = StandardScaler()
            scale_train = sc.fit_transform(X_train[col].values.reshape(-1, 1))
            scale_test = sc.transform(X_test[col].values.reshape(-1, 1))
            # scale_val = sc.transform(X_val[col].values.reshape(-1, 1))
    
            # Assign the scaled data back
            X_train.loc[:, col] = scale_train.flatten()
            X_test.loc[:, col] = scale_test.flatten()
            # X_val.loc[:, col] = scale_val.flatten()

        # To parquet
        X_train.to_parquet(prob_config.train_x_path, index=False)
        y_train.to_parquet(prob_config.train_y_path, index=False)
        X_test.to_parquet(prob_config.test_x_path, index=False)
        y_test.to_parquet(prob_config.test_y_path, index=False)
        logging.info("finish process_raw_data")

    @staticmethod
    def load_train_data(prob_config: ProblemConfig):
        train_x_path = prob_config.train_x_path
        train_y_path = prob_config.train_y_path
        X_train = pd.read_parquet(train_x_path)
        y_train = pd.read_parquet(train_y_path)
        return X_train, y_train[prob_config.target_col]

    @staticmethod
    def load_test_data(prob_config: ProblemConfig):
        dev_x_path = prob_config.test_x_path
        dev_y_path = prob_config.test_y_path
        dev_x = pd.read_parquet(dev_x_path)
        dev_y = pd.read_parquet(dev_y_path)
        return dev_x, dev_y[prob_config.target_col]

    @staticmethod
    def load_category_index(prob_config: ProblemConfig):
        with open(prob_config.category_index_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_capture_data(prob_config: ProblemConfig):
        captured_x_path = prob_config.captured_x_path
        captured_y_path = prob_config.uncertain_y_path
        captured_x = pd.read_parquet(captured_x_path)
        captured_y = pd.read_parquet(captured_y_path)
        return captured_x, captured_y[prob_config.target_col]

