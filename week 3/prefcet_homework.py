import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import mlflow
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner, ConcurrentTaskRunner
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule, CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta, date
from datetime import date as dt
from dateutil.relativedelta import relativedelta as duc

@task
def get_path(date):
    if date == None:
        date = dt.today()
        train = date - duc(months=4)
        val = date - duc(months=3)
        # print(train, val)
        train_path =f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{train.strftime('%Y-%m')}.parquet"
        val_path =f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{val.strftime('%Y-%m')}.parquet"
        # print(train_path, val_path)
        return train_path, val_path
    else:
        year, month, day = date.split('-', 2)
        date = dt(int(year), int(month), int(day))
        train = date - duc(months=2)
        val = date - duc(months=1)
        # print(train, val)
        train_path =f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{train.strftime('%Y-%m')}.parquet"
        val_path =f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{val.strftime('%Y-%m')}.parquet"
        # print(train_path, val_path)
        return train_path, val_path

@task
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.dropOff_datetime = pd.to_datetime(df.dropOff_datetime)
    df.pickup_datetime = pd.to_datetime(df.pickup_datetime)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

@task
def add_features(df_train, df_val):
    # df_train = read_dataframe(train_path)
    # df_val = read_dataframe(val_path)

    print(len(df_train), len(df_val))

    df_train['PU_DO'] = df_train['PUlocationID'] + '_' + df_train['DOlocationID']
    df_val['PU_DO'] = df_val['PUlocationID'] + '_' + df_val['DOlocationID']


    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']

    dv = DictVectorizer()

    train_dicts = df_train[categorical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv

#######################################################
# # Modelling
@task
def train_best_model(train, valid, y_val, dv, date):
    mlflow.xgboost.autolog()

    with mlflow.start_run():

        mlflow.set_tag("developer 1.0", "abideen")
        mlflow.log_param("train-data-path", "https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-01.parquet")
        mlflow.log_param("valid-data-path", "https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-02.parquet")
        
        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        with open(f"models/model-{date}.bin", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(f"models/model-{date}.bin", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")



@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    train_path, val_path = get_path(date).result()
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val).result()
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    train_best_model(train, valid, y_val, dv, date)



# main(date='2021-08-15')
# main()

DeploymentSpec(
    flow=main,
    name='homework',
    schedule=CronSchedule(cron='0 9 15 * *'),
    flow_runner=SubprocessFlowRunner(),
    tags=['ml']
)