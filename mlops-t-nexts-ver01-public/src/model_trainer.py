import argparse
import logging

import mlflow
import numpy as np
import lightgbm as lgbm
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score

from problem_config import (
    ProblemConfig,
    ProblemConst,
    get_prob_config,
)
from raw_data_processor import RawDataProcessor
from utils import AppConfig
# show logged mlflow data 
from mlflow_utils import fetch_logged_data



class ModelTrainer:
    EXPERIMENT_NAME = "lightgbm-1"

    @staticmethod
    def train_model(prob_config: ProblemConfig, model_params, add_captured_data=False):
        logging.info("start train_model")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(
            f"{prob_config.phase_id}_{prob_config.prob_id}_{ModelTrainer.EXPERIMENT_NAME}"
        )

        # load train data
        X_train, y_train = RawDataProcessor.load_train_data(prob_config)
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        logging.info(f"loaded {len(X_train)} samples")

        if add_captured_data:
            captured_x, captured_y = RawDataProcessor.load_capture_data(prob_config)
            captured_x = captured_x.to_numpy()
            captured_y = captured_y.to_numpy()
            X_train = np.concatenate((X_train, captured_x))
            y_train = np.concatenate((y_train, captured_y))
            logging.info(f"added {len(captured_x)} captured samples")

        # train model
        if len(np.unique(y_train)) == 2:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"
        # model = lgbm.LGBMClassifier(objective=objective, **model_params)
        model = lgbm.LGBMClassifier(objective=objective, model__num_leaves=190, model__min_data_in_leaf=67,
                                                         model__max_depth=11, model__learning_rate= 0.09998507726440822,
                                                        model__num_iterations= 27)
        model.fit(X_train, y_train)

        # evaluate
        X_test, y_test = RawDataProcessor.load_test_data(prob_config)
        predictions = model.predict(X_test)
        auc_score = roc_auc_score(y_test, predictions)
        accuracy_score= accuracy_score(y_test, predictions)
        metrics = {"test_auc": auc_score, "accuracy_score":accuracy_score}
        logging.info(f"metrics: {metrics}")

        # mlflow log
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_test, predictions)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
            signature=signature,
        )

        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

        run_id = mlflow.last_active_run().info.run_id
        print("Logged data and model in run {}".format(run_id))

        # show logged data
        for key, data in fetch_logged_data(run_id).items():
            print("\n---------- logged {} ----------".format(key))
            print(data)

        # end mflow session
        mlflow.end_run()
        logging.info("finish train_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument(
        "--add-captured-data", type=lambda x: (str(x).lower() == "true"), default=False
    )
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    model_config = {"random_state": prob_config.random_state}
    ModelTrainer.train_model(
        prob_config, model_config, add_captured_data=args.add_captured_data
    )
