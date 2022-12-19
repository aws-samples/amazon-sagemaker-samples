
"""Evaluation script for measuring model accuracy."""
import json
import os
import tarfile
import logging
import pickle

import pandas as pd
import xgboost

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    print("Loading test input data")
    test_path = "/opt/ml/processing/test/test.csv"
    testdf = pd.read_csv(test_path)

    logger.debug("Reading test data.")
    y_test = testdf.iloc[:, 0].to_numpy()
    testdf.drop(testdf.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(testdf)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    print("Creating classification evaluation report")
    acc = accuracy_score(y_test, predictions.round())
    auc = roc_auc_score(y_test, predictions)

    # The metrics reported can change based on the model used, but it must be a specific name per (https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation": "NaN",
            },
            "auc": {"value": auc, "standard_deviation": "NaN"},
        },
    }

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join(
        "/opt/ml/processing/evaluation", "evaluation.json"
    )
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
