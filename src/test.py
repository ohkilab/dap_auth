import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from scipy import optimize
from scipy import interpolate
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


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


def plot_roc(all_fpr, all_tpr, all_fnr, all_thr):
    mean_fpr = np.unique(np.concatenate([all_fpr[i] for i in range(0, 8)]))
    mean_tpr = np.zeros_like(mean_fpr)

    for i in range(0, 8):
        mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i], left=0, right=1)
    mean_tpr /= 8
    mean_thr = np.zeros_like(mean_fpr)

    for i in range(0, 8):
        mean_thr += np.interp(mean_fpr, all_fpr[i], all_thr[i], left=0, right=1)
    mean_thr /= 8
    mean_fnr = np.zeros_like(mean_fpr)

    for i in range(0, 8):
        mean_fnr += np.interp(mean_fpr, all_fpr[i], all_fnr[i], left=0, right=1)
    mean_fnr /= 8
    mean_tpr = np.insert(mean_tpr, 0, 0)
    mean_fpr = np.insert(mean_fpr, 0, 0)
    # mean_thr = np.insert(mean_thr, 9 , 0)

    auc_macro = auc(mean_fpr, mean_tpr)
    print(auc_macro)

    return mean_fnr, mean_fpr, mean_tpr, mean_thr


def plt_bar(df, img_label):
    ax = df.plot.bar(legend=False)
    ax.tick_params(axis="x", rotation=0)
    plt.xlabel("認証ペア")
    plt.ylabel("EER")
    plt.ylim(0, 1)
    plt.savefig("img/" + "EERs.png")
    plt.show()


@hydra.main(version_base=None, config_path="../conf", config_name="test")
def test(cfg: DictConfig) -> None:
    assert cfg.correct_user1 is None, "Do not specify the argument correct_user1."
    assert cfg.correct_user2 is None, "Do not specify the argument correct_user2."

    train_feat_df, train_label_list, train_pair_list = extract_feature_from_old_data(
        cfg
    )
    test_feat_df, test_label_list, test_pair_list = extract_feature_from_old_data(
        cfg, is_train=False
    )

    pair_id_list = list(set(train_label_list))

    eer_df = pd.DataFrame(columns=["EER"])
    all_tpr = []
    all_fpr = []
    all_fnr = []
    all_thr = []

    for correct_pair_id in pair_id_list:
        train_label_list = [1 if i == correct_pair_id else 0 for i in train_pair_list]
        # svm_classifier = load_model(None, ModelType.SVM)
        # svm_classifier.fit(feat_df, label_list)
        rf_classifier = load_model(None, ModelType.RF)
        rf_classifier.fit(train_feat_df, train_label_list)
        # lgbm_classifier = load_model(None, ModelType.LGBM)
        # lgbm_classifier.fit(feat_df, label_list)
        # xgb_classifier = load_model(None, ModelType.XGB)
        # xgb_classifier.fit(feat_df, label_list)

        all_fpr, all_tpr, all_fnr, all_thr, eer_df = classifier_scorer(
            rf_classifier,
            correct_pair_id,
            test_feat_df,
            test_label_list,
            all_fpr,
            all_tpr,
            all_fnr,
            all_thr,
            eer_df,
        )
    mean_fnr1, mean_fpr1, mean_tpr1, mean_thr1 = plot_roc(
        all_fpr, all_tpr, all_fnr, all_thr
    )
    plt.plot(mean_fpr1, mean_tpr1, linestyle="-", linewidth=2, label="session1")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.savefig("img/" + "_ROC.png")
    plt.show()

    plt_bar(eer_df, "normal")


if __name__ == "__main__":
    test()
