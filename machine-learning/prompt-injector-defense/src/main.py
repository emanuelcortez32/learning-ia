import re
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from pathlib import Path

current_dir = Path(__file__).resolve()
current_dir_parent = current_dir.parent
top_folder = current_dir_parent.parent
assets_folder = top_folder / "assets"
out_folder = top_folder / "out"

train_file_name = "train_es.parquet"
test_file_name = "test_es.parquet"

train_file_path = assets_folder / train_file_name
test_file_path = assets_folder / test_file_name

df_train = pd.read_parquet(train_file_path)
df_test = pd.read_parquet(test_file_path)


print("Train Data")
print("\n-------------------------")
print(df_train.head())
print("\n-------------------------")
print("Test Data")
print("\n-------------------------")
print(df_test.head())
print("\n-------------------------")


if df_train["label"].dtype == bool:
    df_train["label"] = df_train["label"].astype(int)
else:
    df_train["label"] = (
        df_train["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
        .fillna(df_train["label"])
    )
    df_train["label"] = df_train["label"].astype(int)


if df_test["label"].dtype == bool:
    df_test["label"] = df_test["label"].astype(int)
else:
    df_test["label"] = (
        df_test["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
        .fillna(df_test["label"])
    )
    df_test["label"] = df_test["label"].astype(int)


X_Train = df_train["text"].astype(str).values
y_train = df_train["label"].values


X_Test = df_test["text"].astype(str).values
y_test = df_test["label"].values

# -------------------------
# Text normalization
# -------------------------
_whitespace_re = re.compile(r"\s+")
def normalize(s: str) -> str:
    s = s.lower()
    s = _whitespace_re.sub(" ", s).strip()
    return s

# -------------------------
# Vectorization: word + char
# -------------------------
word_tfidf = TfidfVectorizer(
    preprocessor=normalize,
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

char_tfidf = TfidfVectorizer(
    preprocessor=normalize,
    analyzer="char_wb",   # robust to token boundaries + obfuscation
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

feats = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf),
])

svc_base = LinearSVC(class_weight="balanced")
svc = CalibratedClassifierCV(
    estimator=svc_base,
    method="sigmoid",
    cv=3
)

model = Pipeline([
    ("feats", feats),
    ("clf", svc)
])

model.fit(X_Train, y_train)

proba = model.predict_proba(X_Test)[:, 1]
pred_05 = (proba >= 0.5).astype(int)

print("PR-AUC:", average_precision_score(y_test, proba))
print("\nConfusion matrix @0.5:\n", confusion_matrix(y_test, pred_05))
print("\nReport @0.5:\n", classification_report(y_test, pred_05, digits=4))

# -------------------------
# Threshold selection utilities
# -------------------------
def threshold_for_precision(y_true, y_score, target_precision=0.98):
    p, r, t = precision_recall_curve(y_true, y_score)
    # p,r length = len(t)+1
    best = None
    for precision, recall, thr in zip(p[:-1], r[:-1], t):
        if precision >= target_precision:
            if best is None or recall > best["recall"]:
                best = {"threshold": float(thr), "precision": float(precision), "recall": float(recall)}
    return best

def threshold_for_recall(y_true, y_score, target_recall=0.98):
    p, r, t = precision_recall_curve(y_true, y_score)
    best = None
    for precision, recall, thr in zip(p[:-1], r[:-1], t):
        if recall >= target_recall:
            if best is None or precision > best["precision"]:
                best = {"threshold": float(thr), "precision": float(precision), "recall": float(recall)}
    return best

best_p = threshold_for_precision(y_test, proba, target_precision=0.98)
best_r = threshold_for_recall(y_test, proba, target_recall=0.98)

print("\nBest threshold for precision>=0.98:", best_p)
print("Best threshold for recall>=0.98:", best_r)

# Example: pick one threshold policy for prod
chosen = best_p or {"threshold": 0.9, "precision": None, "recall": None}
THRESHOLD = chosen["threshold"]
pred_thr = (proba >= THRESHOLD).astype(int)

print(f"\nConfusion matrix @{THRESHOLD:.4f}:\n", confusion_matrix(y_test, pred_thr))
print(f"\nReport @{THRESHOLD:.4f}:\n", classification_report(y_test, pred_thr, digits=4))


joblib.dump(model, out_folder / "models" / "joblib" /  "prompt_injection_defense.joblib")

#onnx_model = convert_sklearn(model, initial_types=[("input", StringTensorType([None,1]))])

#with open(out_folder / "models" / "onnx" / "prompt_injection_defense.onnx", "wb") as f:
#    f.write(onnx_model.SerializeToString())