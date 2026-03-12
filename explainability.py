import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import shap
from lime.lime_tabular import LimeTabularExplainer


class KerasSklearnWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]

        out_dim = self.model.output_shape[-1]
        self.classes_ = np.arange(out_dim)
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "is_fitted_") and self.is_fitted_

    def predict_proba(self, X):
        X = np.asarray(X)
        X3d = X.reshape(X.shape[0], X.shape[1], 1)
        return self.model.predict(X3d, verbose=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


def _predict_proba_from_flat(model, X_flat):
    X_flat = np.asarray(X_flat)
    X3d = X_flat.reshape(X_flat.shape[0], X_flat.shape[1], 1)
    return model.predict(X3d, verbose=0)



# --------------------------------------------------
# 1. Permutation Feature Importance (PFI)
# --------------------------------------------------
def run_pfi(model, X, y, feature_names):
    print("Running Permutation Feature Importance...")

    X_flat = X.reshape(X.shape[0], X.shape[1])
    estimator = KerasSklearnWrapper(model).fit(X_flat, y)

    r = permutation_importance(
        estimator,
        X_flat,
        y,
        n_repeats=10,
        random_state=42,
        scoring="accuracy"
    )

    importances = r.importances_mean
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.title("Permutation Feature Importance")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 2. SHAP (Global + Local)
# --------------------------------------------------
def run_shap(model, X, feature_names):
    print("Running SHAP explainability...")

    X_flat = X.reshape(X.shape[0], X.shape[1])

    bg_n = min(50, X_flat.shape[0])
    eval_n = min(5, X_flat.shape[0])

    predictor = lambda data: _predict_proba_from_flat(model, data)
    explainer = shap.KernelExplainer(predictor, X_flat[:bg_n])
    shap_values = explainer.shap_values(X_flat[:eval_n])

    shap.summary_plot(shap_values, X_flat[:eval_n], feature_names=feature_names)


# --------------------------------------------------
# 3. LIME (Local Explanation)
# --------------------------------------------------
def run_lime(model, X, feature_names):
    print("Running LIME explainability...")

    X_flat = X.reshape(X.shape[0], X.shape[1])

    explainer = LimeTabularExplainer(
        training_data=X_flat,
        feature_names=feature_names,
        class_names=["Healthy", "Myopathy", "Neuropathy"],
        mode="classification"
    )

    exp = explainer.explain_instance(
        X_flat[0],
        lambda data: _predict_proba_from_flat(model, data),
        num_features=8
    )

    exp.show_in_notebook(show_table=True)
    exp.as_pyplot_figure()
    plt.show()


# --------------------------------------------------
# 4. Partial Dependence Plot (PDP)
# --------------------------------------------------
def run_pdp(model, X, feature_names, target_class=0):
    print("Running Partial Dependence Plots...")

    X_flat = X.reshape(X.shape[0], X.shape[1])
    estimator = KerasSklearnWrapper(model).fit(X_flat)

    fig, ax = plt.subplots(figsize=(8, 5))
    PartialDependenceDisplay.from_estimator(
        estimator,
        X_flat,
        features=[0, 1, 2],
        feature_names=feature_names,
        target=target_class,
        ax=ax
    )
    plt.show()




