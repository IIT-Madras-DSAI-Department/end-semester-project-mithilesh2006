import numpy as np
import pandas as pd
import time
from collections import Counter
import matplotlib.pyplot
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def compute_metrics(y_true, y_pred, num_classes=10):
    eps = 1e-12
    precisions = []
    recalls = []
    f1s = []

    tp_global = fp_global = fn_global = 0

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        tp_global += tp
        fp_global += fp
        fn_global += fn

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    micro_p = tp_global / (tp_global + fp_global + eps)
    micro_r = tp_global / (tp_global + fn_global + eps)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps)

    return macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1



class PCA:
    def __init__(self, n_components=100, max_samples_for_fit=20000):
        self.n_components = n_components
        self.max_samples_for_fit = max_samples_for_fit
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        m = X.shape[0]

        if m > self.max_samples_for_fit:
            idx = np.random.choice(m, self.max_samples_for_fit, replace=False)
            X = X[idx]

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[:self.n_components].T  

    def transform(self, X):
        return (X - self.mean_) @ self.components_



class KNNClassifier:
    def __init__(self, k=5, max_train_samples=60000):
        self.k = k
        self.max_train_samples = max_train_samples

    def fit(self, X, y):
        m = X.shape[0]

        if m > self.max_train_samples:
            idx = np.random.choice(m, self.max_train_samples, replace=False)
            self.X = X[idx].astype(np.float32)
            self.y = y[idx].astype(int)
            print(f"[KNN] Using {self.X.shape[0]} subsampled training samples.")
        else:
            self.X = X.astype(np.float32)
            self.y = y.astype(int)

    def predict(self, X):
        X = X.astype(np.float32)
        m_test = X.shape[0]
        y_pred = np.zeros(m_test, int)

        batch = 200
        for i in range(0, m_test, batch):
            Xb = X[i:i + batch]

            dists = (
                np.sum(Xb**2, axis=1, keepdims=True)
                + np.sum(self.X**2, axis=1, keepdims=True).T
                - 2 * (Xb @ self.X.T)
        )


            knn_idx = np.argpartition(dists, self.k, axis=1)[:, :self.k]
            labels = self.y[knn_idx]

            for j in range(labels.shape[0]):
                labs, counts = np.unique(labels[j], return_counts=True)
                y_pred[i+j] = labs[np.argmax(counts)]

        return y_pred





def load_mnist(train_path="MNIST_train.csv", val_path="MNIST_validation.csv"):
    print("Loading MNIST...")
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    train.drop(columns=['even'], inplace=True)
    val.drop(columns=['even'], inplace=True)

    X_train = train.iloc[:, 1:].values.astype(np.float32) / 255.0
    y_train = train.iloc[:, 0].values.astype(int)

    X_val = val.iloc[:, 1:].values.astype(np.float32) / 255.0
    y_val = val.iloc[:, 0].values.astype(int)

    print("Train:", X_train.shape, "Val:", X_val.shape)
    return X_train, y_train, X_val, y_val


X_train, y_train, X_val, y_val = load_mnist()



def evaluate_knn_train(knn, X_train, y_train):
    print("\n========== TRAIN SET EVALUATION ==========")
    
    t0 = time.time()
    y_pred_train = knn.predict(X_train)
    pred_time_train = time.time() - t0

    acc = np.mean(y_pred_train == y_train)
    mp, mr, mf1, microp, micror, microf1 = compute_metrics(y_train, y_pred_train)

    print(f"Train Accuracy        : {acc:.4f}")
    print(f"Train Macro Precision : {mp:.4f}")
    print(f"Train Macro Recall    : {mr:.4f}")
    print(f"Train Macro F1        : {mf1:.4f}")
    print(f"Train Micro Precision : {microp:.4f}")
    print(f"Train Micro Recall    : {micror:.4f}")
    print(f"Train Micro F1        : {microf1:.4f}")
    print(f"Train Prediction Time : {pred_time_train:.2f}s")

    return y_pred_train


def evaluate_knn_val(knn, X_val, y_val):
    print("\n========== VALIDATION SET EVALUATION ==========")

    t0 = time.time()
    y_pred_val = knn.predict(X_val)
    pred_time_val = time.time() - t0

    acc = np.mean(y_pred_val == y_val)
    mp, mr, mf1, microp, micror, microf1 = compute_metrics(y_val, y_pred_val)

    print(f"Validation Accuracy        : {acc:.4f}")
    print(f"Validation Macro Precision : {mp:.4f}")
    print(f"Validation Macro Recall    : {mr:.4f}")
    print(f"Validation Macro F1        : {mf1:.4f}")
    print(f"Validation Micro Precision : {microp:.4f}")
    print(f"Validation Micro Recall    : {micror:.4f}")
    print(f"Validation Micro F1        : {microf1:.4f}")
    print(f"Validation Prediction Time : {pred_time_val:.2f}s")

    return y_pred_val



knn = KNNClassifier(k=5, max_train_samples=20000)


print("\nTraining KNN model...")
t0 = time.time()
knn.fit(X_train, y_train)
train_time = time.time() - t0
print(f"KNN Training Time: {train_time:.2f}s")

train_preds = evaluate_knn_train(knn, X_train, y_train)


val_preds = evaluate_knn_val(knn, X_val, y_val)





val_preds = knn.predict(X_val)
print("Unique predictions:", np.unique(val_preds))





labels = np.arange(10)

cm = confusion_matrix(y_val, val_preds, labels=labels)

fig, ax = plt.subplots(figsize=(9, 9))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d', ax=ax)

plt.title("KNN Confusion Matrix (10×10)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
