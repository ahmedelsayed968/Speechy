from typing import Tuple
import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm
import wandb
from wandb.sklearn import plot_classifier
from omegaconf import OmegaConf
from config.paths import CONFIG_DIR
from dotenv import load_dotenv
from datasets import load_dataset
from sklearn.utils import all_estimators
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)
from loguru import logger
import os

from modelling.enums import HypterParameterSettings, SupportedModels
load_dotenv()

# wandb.setup(wandb.Settings(reinit="create_new"))

def prepare_dataset(config:dict):
    # load dataset from huggingface 
    ds = load_dataset(path=os.environ.get("DATASET_REPO_ID"),
                      revision=config['training']['dataset_revision'])
    # select inputs and label columns
    ds = ds.select_columns(['inputs',"label"])
    # create df from the dataset
    df = {
        split: split_ds.to_pandas()
        for split,split_ds in ds.items()
    }
    # convert the inputs to 2d np array
    X_train = np.array(df['train']['inputs'].tolist())
    X_val = np.array(df['val']['inputs'].tolist())
    X_test = np.array(df['test']['inputs'].tolist())
    # select the target column
    y_train = df['train']['label']
    y_val = df['val']['label']
    y_test = df['test']['label']
    # labels
    labels = ds['train'].features['label'].names
    return {
        "train":(X_train,y_train),
        "val":(X_val,y_val),
        "test":(X_test,y_test),
        "labels": labels
    }


def model_factory(model_type:SupportedModels,**kwargs)->BaseEstimator:
    if model_type == SupportedModels.LOGISTIC_REGRESSION:
        return LogisticRegression(**kwargs)
    elif model_type == SupportedModels.ADA_BOOST:
        return AdaBoostClassifier(**kwargs)
    elif model_type == SupportedModels.DECISION_TREE:
        return DecisionTreeClassifier(**kwargs)
    elif model_type == SupportedModels.GRADIENT_BOOSTING:
        return GradientBoostingClassifier(**kwargs)
    elif model_type == SupportedModels.SGD_CLASSIFIER:
        return SGDClassifier(**kwargs)
    else:
        raise TypeError(f"{model_type} is not supported yet!")

def safe_metric(metric_func, *args, **kwargs):
    try:
        result = metric_func(*args, **kwargs)
        if np.isnan(result):
            return None
        return float(result)
    except Exception as e:
        logger.warning(f"Failed to compute {metric_func.__name__}: {e}")
        return None
    
def parse_included_models(config:dict)->Tuple[SupportedModels,BaseEstimator]:
    models = []
    for model_name,model_settings in config['training']['models'].items():
        if model_settings['include']:
            hyperparameter_setting = HypterParameterSettings(model_settings['hyperparameter_setting'])
            model_name = SupportedModels(model_name)
            if hyperparameter_setting == HypterParameterSettings.DEFAULT:
                model = model_factory(model_name)
                models.append((model_name,model))
            else:
                custom_config = model_settings['config']['custom']
                model = model_factory(model_type=model_name,**custom_config)
            models.append((model_name,model))
    return models
def train(config):
    data = prepare_dataset(config)
    models= all_estimators(type_filter="classifier")

    X_train,y_train = data['train']
    X_val , y_val = data['val']
    X_test, y_test = data['test']
    labels = data['labels']
    # scaling all features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    # X_test  = scaler.transform(X_test)

    for model_name,model_cls in tqdm(models,desc="Training"):
        logger.info(f"Start Training {model_name}")
        try:
            model = model_cls()
            model.fit(X_train, y_train)
        except Exception as e:
            logger.warning(f"Skipping {model_name} due to error during fit: {e}")
            continue

        try:
            y_pred = model.predict(X_val)
        except Exception as e:
            logger.warning(f"Skipping {model_name} due to error during prediction: {e}")
            continue


        y_probas_full = None
        y_probas = None
        if hasattr(model,"predict_proba"):
            try:
                y_probas_full = model.predict_proba(X_val)   
                y_probas = y_probas_full[:, 1]  
            except Exception as e:
                logger.warning(f"Skipping {model_name} due to error during prediction probe: {e}")

        model_params = model.get_params()
        run = wandb.init(name=model_name,reinit=True,config=model_params,**config['wandb-settings']) 
        # Compute metrics
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
            "roc_auc": safe_metric(roc_auc_score, y_val, y_probas) if y_probas is not None else None,
            "log_loss": safe_metric(log_loss, y_val, y_probas) if y_probas is not None else None
        }

        if y_probas_full is not None and metrics['roc_auc'] is not None:
            plot_classifier(
                model,
                X_train,
                X_val,
                y_train,
                y_val,
                y_pred,
                y_probas_full,
                labels,
                model_name=model_name,
                feature_names=None,
                is_binary=True
            )
        else:
            logger.warning(f"Skipping plot_classifier for {model_name}")
        

        # Log metrics to wandb
        run.log(metrics)
        run.finish()

if __name__ == "__main__":
    training_config = OmegaConf.load(CONFIG_DIR/"training.yaml")
    train(training_config)