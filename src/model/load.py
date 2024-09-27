from enum import Enum
import json

import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class ModelType(Enum):
    SVM = 1
    RF = 2
    LGBM = 3
    XGB = 4


def convert_modeltype(modelname: str) -> ModelType:
    validate_modeltype(modelname)
    return ModelType[modelname.upper()]


def validate_modeltype(modelname: str):
    if modelname.upper() not in ModelType.__members__:
        raise ValueError("Invalid model type")


def load_model(param_dict_path: str, target_modelname: str = "svm") -> object:

    model = convert_modeltype(target_modelname)

    if model == ModelType.SVM:
        classifer = SVC()
    elif model == ModelType.RF:
        classifer = RandomForestClassifier()
    elif model == ModelType.LGBM:
        classifer = LGBMClassifier()
    elif model == ModelType.XGB:
        classifer = XGBClassifier()

    if param_dict_path:
        try:
            with open(param_dict_path, "r") as f:
                loaded_params = json.load(f)
                classifer.set_params(**loaded_params)
        except FileNotFoundError:
            raise FileNotFoundError("The parameter dictionary file does not exist")

    return classifer
