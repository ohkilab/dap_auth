from enum import Enum
from typing import Union

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


def load_model(
    model_path: str, target_modelname: Union[str, ModelType] = "svm"
) -> object:

    if model_path:
        try:
            with open(model_path, "rb") as f:
                classifer = pickle.load(f)
                return classifer
        except FileNotFoundError:
            raise FileNotFoundError("The parameter dictionary file does not exist")

    if type(target_modelname) != ModelType:
        model_type = convert_modeltype(target_modelname)
    else:
        model_type = target_modelname

    if model_type == ModelType.SVM:
        classifer = SVC(probability=True)
    elif model_type == ModelType.RF:
        classifer = RandomForestClassifier()
    elif model_type == ModelType.LGBM:
        classifer = LGBMClassifier()
    elif model_type == ModelType.XGB:
        classifer = XGBClassifier()

    return classifer
