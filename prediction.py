import os
import mlflow.sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as skpreprocessing
from sklearn.linear_model import SGDRegressor   
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature
import mlflow
import pandas as pd
import utils
import numpy as np

data_url = "https://dkube-examples-data.s3.us-west-2.amazonaws.com/monitoring-insurance/training-data/insurance.csv"


class InsuranceModel:

    def __init__(self):
        self._regressor = SGDRegressor()

    @property
    def model(self):
        """
        Getter for the property the model
        :return: return the trained decision tree model
        """

        return self._regressor

    def mlflow_run(self, run_name="Insurance cost prediction example"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param run_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (experiment_id, run_id)
        """
        with mlflow.start_run(run_name=run_name) as run:

            # get current run and experiment id
            run_id = run.info.run_uuid
            experiment_id = run.info.experiment_id

            # loading and preprocessing training data
            data = pd.read_csv(data_url)
            insurance_input = data.drop(['charges','timestamp','unique_id'],axis=1)
            insurance_target = data['charges']
            
            for col in ['sex', 'smoker', 'region']:
                if (insurance_input[col].dtype == 'object'):
                    le = skpreprocessing.LabelEncoder()
                    le = le.fit(insurance_input[col])
                    insurance_input[col] = le.transform(insurance_input[col])
                    print('Completed Label encoding on',col)
            
            #standardize data
            x_scaled = StandardScaler().fit_transform(insurance_input)
            x_train, x_test, y_train, y_test = train_test_split(x_scaled,
                                                            insurance_target,
                                                            test_size = 0.25,
                                                            random_state=1211)

            # train and predict
            self._regressor.fit(x_train, y_train)


            y_pred_train = self._regressor.predict(x_train)    # Predict on train data.
            y_pred_train[y_pred_train < 0] = y_pred_train.mean()
            y_pred = self._regressor.predict(x_test)   # Predict on test data.
            y_pred[y_pred < 0] = y_pred.mean()
            
            #######--- Calculating metrics ---############
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

            signature = infer_signature(x_train, self._regressor.predict(x_train))

            # Log model and params using the MLflow sklearn APIs
            mlflow.sklearn.log_model(self.model, "sgd-regressor", signature=signature)

            # log metrics in mlflow
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)

            # Logging train data
            temp_name = "insurance.csv"
            data.to_csv(temp_name, index=False)
            mlflow.log_artifact(temp_name, "training-data")
            try:
                os.remove(temp_name)
            except FileNotFoundError as e:
                print(f"{temp_name} file is not found")

            # Logging serving transformer
            transformerfile_name = "transformer.py"
            mlflow.log_artifact(transformerfile_name, "transformer")

            # Logging monitoring train data transformer
            traindata_transformerfile_name = "transform-data.py"
            mlflow.log_artifact(traindata_transformerfile_name, "traindata-transformer")

            print("<->" * 40)
            print("Inside MLflow Run with run_id {run_id} and experiment_id {experiment_id}")
            print("MAE", mae)
            print("MSE", mse)
            print("RMSE", rmse)

            return experiment_id, run_id
