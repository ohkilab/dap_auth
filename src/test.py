import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from scipy import optimize
from scipy import interpolate
from sklearn.metrics import roc_curve, auc


from encapsulate_preprocess import extract_feature_from_old_data
from model.load import load_model, ModelType


def calculate_classifier_score(label_list, pred_proba, target_label):
    eer_df = pd.DataFrame(columns=["EER"], index=[target_label])
    fpr, tpr, thresholds = roc_curve(label_list, pred_proba)
    fnr = 1 - tpr
    eer = optimize.brentq(
        lambda x: 1.0 - x - interpolate.interp1d(fpr, tpr)(x), 0.0, 1.0
    )
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    eer_threshold = thresholds[eer_idx]
    eer_df["EER"] = eer

    return eer_df, fpr, tpr, fnr, eer_threshold, thresholds


def classifier_scorer(
    classifier,
    target_label,
    feat,
    label_list,
    all_fpr,
    all_tpr,
    all_fnr,
    all_thr,
    all_eer,
):
    pred_proba = classifier.predict_proba(feat)[:, 1]
    eer_df, fpr, tpr, fnr, _, thresholds = calculate_classifier_score(
        label_list, pred_proba, target_label
    )
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_fnr.append(fnr)
    all_thr.append(thresholds)
    all_eer = pd.concat([all_eer, eer_df], axis=0)
    return all_fpr, all_tpr, all_fnr, all_thr, all_eer


@hydra.main(version_base=None, config_path="../conf", config_name="test")
def test(cfg: DictConfig) -> None:
    assert cfg.correct_user1 is None, "Do not specify the argument correct_user1."
    assert cfg.correct_user2 is None, "Do not specify the argument correct_user2."

    feat_df, label_list, pair_list = extract_feature_from_old_data(cfg)

    pair_id_list = list(set(label_list))

    eer_df = pd.DataFrame(columns=["EER"])
    all_tpr = []
    all_fpr = []
    all_fnr = []
    all_thr = []

    for correct_pair_id in pair_id_list:
        label_list = [1 if i == correct_pair_id else 0 for i in pair_list]
        # svm_classifier = load_model(None, ModelType.SVM)
        # svm_classifier.fit(feat_df, label_list)
        rf_classifier = load_model(None, ModelType.RF)
        rf_classifier.fit(feat_df, label_list)
        # lgbm_classifier = load_model(None, ModelType.LGBM)
        # lgbm_classifier.fit(feat_df, label_list)
        # xgb_classifier = load_model(None, ModelType.XGB)
        # xgb_classifier.fit(feat_df, label_list)

        all_fpr, all_tpr, all_fnr, all_thr, eer_df = classifier_scorer(
            rf_classifier,
            correct_pair_id,
            feat_df,
            label_list,
            all_fpr,
            all_tpr,
            all_fnr,
            all_thr,
            eer_df,
        )
        print("hoge")


if __name__ == "__main__":
    test()
