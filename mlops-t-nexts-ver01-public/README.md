# MLOps Marathon 2023 - Sample solution

This repository is the sample solution for MLOps Marathon 2023.

## Quickstart

1.  Prepare environment

    ```bash
    # Install python 3.9
    # Install docker version 20.10.17
    # Install docker-compose version v2.6.1
    pip install -r requirements.txt
    make mlflow_up
    ```

2.  Train model

    -   Download data, `./data/raw_data` dir should look like

        ```bash
        data/raw_data
        ├── .gitkeep
        └── phase-1
            └── prob-1
                ├── features_config.json
                └── raw_train.parquet
                └── description.csv
        └── phase-2
            └── prob-2
                ├── features_config.json
                └── raw_train.parquet
                └── description.csv
        ```

    -   Process data

        ```bash
        python src/raw_data_process.py --phase-id phase-1 --prob-id prob-1
        ```

        ```bash
        python src/raw_data_process.py --phase-id phase-2 --prob-id prob-2
        ```

    -   After processing data, `./data/train_data` dir should look like

        ```bash
        data/train_data
        ├── .gitkeep
        └── phase-1
            └── prob-1
                ├── category_index.pickle
                ├── X_test.parquet
                ├── y_test.parquet
                ├── X_train.parquet
                └── y_train.parquet
        ```

    -   Train model

        ```bash
        export MLFLOW_TRACKING_URI=http://localhost:5040
        python src/model_trainer.py --phase-id phase-1 --prob-id prob-1
        ```

        ```bash
        export MLFLOW_TRACKING_URI=http://localhost:5040
        python src/model_trainer.py --phase-id phase-2 --prob-id prob-2
        ```

    -   Register model: Go to mlflow UI at <http://localhost:5040> and register a new model named **phase-1_prob-1_model-1** and **phase-1_prob-2_model-2**

3.  Deploy model predictor

    -   Create model config at `data/model_config/phase-1/prob-1/model-1.yaml` with content:

        ```yaml
        phase_id: "phase-1"
        prob_id: "prob-1"
        model_name: "phase-1_prob-1_model-1"
        model_version: "1"
        ```

    -   Create model config at `data/model_config/phase-1/prob-2/model-1.yaml` with content:

        ```yaml
        phase_id: "phase-1"
        prob_id: "prob-2"
        model_name: "phase-1_prob-2_model-1"
        model_version: "1"
        ```

    -   Test model predictor

        ```bash
        # run model predictor
        export MLFLOW_TRACKING_URI=http://localhost:5040
        python src/model_predictor.py --config-path data/model_config/phase-1/prob-1/model-1.yaml --port 8000

        # run model predictor
        export MLFLOW_TRACKING_URI=http://localhost:5040
        python src/model_predictor.py --config-path data/model_config/phase-1/prob-2/model-1.yaml --port 8000

        # curl in another terminal
        curl -X POST http://localhost:8000/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json

        curl -X POST http://localhost:8000/phase-1/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-2/payload-1.json

        # stop the predictor above
        ```

    -   Deploy model predictor

        ```bash
        make predictor_up
        make predictor_curl
        ```

    -   After running `make predictor_curl` to send requests to the server, `./data/captured_data` dir should look like:

        ```bash
         data/captured_data
         ├── .gitkeep
         └── phase-1
             └── prob-1
                 ├── 123.parquet
                 └── 456.parquet
        ```

4.  Improve model

    -   The technique to improve model by using the prediction data is described in `improve_model.md`.
    -   Label the captured data, taking around 3 minutes

        ```bash
        python src/label_captured_data.py --phase-id phase-1 --prob-id prob-1
        ```

    -   After label the captured data, `./data/captured_data` dir should look like:

        ```bash
        data/captured_data
        ├── .gitkeep
        └── phase-1
            └── prob-1
                ├── 123.parquet
                ├── 456.parquet
                └── processed
                    ├── captured_x.parquet
                    └── uncertain_y.parquet
        ```

    -   Improve model with updated data

        ```bash
        export MLFLOW_TRACKING_URI=http://localhost:5040
        python src/model_trainer.py --phase-id phase-1 --prob-id prob-1 --add-captured-data true
        ```

    -   Register model: Go to mlflow UI at <http://localhost:5040> and register model using the existing name **phase-1_prob-1_model-1**. The latest model version now should be `2`.

    -   Update model config at `data/model_config/phase-1/prob-1/model-1.yaml` to:

        ```yaml
        phase_id: "phase-1"
        prob_id: "prob-1"
        model_name: "phase-1_prob-1_model-1"
        model_version: "2"
        ```

    -   Deploy new model version

        ```bash
        make predictor_restart
        make predictor_curl
        ```

5.  Teardown

    ```bash
    make teardown
    ```
