import numpy as np
import time
import pandas as pd


class DecisionTree:
    def __init__(self, max_depth=7, min_samples_split=10,
                 feature_subsample=0.5, n_thresholds=10, num_classes=10):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsample = feature_subsample
        self.n_thresholds = n_thresholds
        self.num_classes = num_classes
        self.root = None

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0.0
        g = 1.0
        for c in range(self.num_classes):
            p = np.sum(y == c) / m
            g -= p * p
        return g

    def _majority(self, y):
        labs, cts = np.unique(y, return_counts=True)
        return labs[np.argmax(cts)]

    def _best_split(self, X, y):
        m, n = X.shape
        n_feats = max(1, int(n * self.feature_subsample))
        feat_idx = np.random.choice(n, n_feats, replace=False)

        best_gini = 999
        best_feat = None
        best_thr = None

        for f in feat_idx:
            col = X[:, f]
            thresholds = np.quantile(col, np.linspace(0.05, 0.95, self.n_thresholds))

            for t in np.unique(thresholds):
                left = col <= t
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue

                g_left = self._gini(y[left])
                g_right = self._gini(y[right])
                g_tot = (left.sum() / m) * g_left + (right.sum() / m) * g_right

                if g_tot < best_gini:
                    best_gini = g_tot
                    best_feat = f
                    best_thr = t

        return best_feat, best_thr

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return {"leaf": True, "value": self._majority(y)}

        f, thr = self._best_split(X, y)
        if f is None:
            return {"leaf": True, "value": self._majority(y)}

        left = X[:, f] <= thr
        right = ~left

        return {
            "leaf": False,
            "feat": f,
            "thr": thr,
            "left": self._build(X[left], y[left], depth + 1),
            "right": self._build(X[right], y[right], depth + 1)
        }

    def fit(self, X, y):
        self.root = self._build(X, y, 0)

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feat"]] <= node["thr"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


def train_model(model, X_train, y_train, name="Model"):
    print("\n" + "="*60)
    print(f"TRAINING: {name}")
    print("="*60)

    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()

    print(f"Training Time: {t1 - t0:.2f} seconds")
    print("="*60)

    return model


def validate_model(model, X_val, y_val, name="Model"):
    print("\n" + "="*60)
    print(f"VALIDATION: {name}")
    print("="*60)

    t0 = time.time()
    y_pred = model.predict(X_val)
    t1 = time.time()

    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Prediction Time: {t1 - t0:.2f} seconds")

    return y_pred


def print_precision_recall_f1(y_true, y_pred, name="Model"):
    (macro_p, macro_r, macro_f1,
     micro_p, micro_r, micro_f1) = compute_metrics(y_true, y_pred)

    print("\n" + "="*60)
    print(f"PRECISION / RECALL / F1 : {name}")
    print("="*60)
    print(f"Macro Precision : {macro_p:.4f}")
    print(f"Macro Recall    : {macro_r:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Micro Precision : {micro_p:.4f}")
    print(f"Micro Recall    : {micro_r:.4f}")
    print(f"Micro F1        : {micro_f1:.4f}")
    print("="*60)



class RandomForestClassifier:
    def __init__(self, n_estimators=20, subsample=0.7, feature_subsample=0.3,
                 n_thresholds=10, max_depth=8, min_samples_split=10, num_classes=10):
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.feature_subsample = feature_subsample
        self.n_thresholds = n_thresholds
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_classes = num_classes
        self.trees = []
        self.training_time_ = 0.0

    def fit(self, X, y):
        start = time.time()
        m = len(y)
        self.trees = []

        for i in range(self.n_estimators):
            n_samples = int(m * self.subsample)
            idx = np.random.choice(m, n_samples, replace=True)
            X_bag = X[idx]
            y_bag = y[idx]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_subsample=self.feature_subsample,
                n_thresholds=self.n_thresholds,
                num_classes=self.num_classes
            )
            tree.fit(X_bag, y_bag)
            self.trees.append(tree)

            if (i+1) % 5 == 0 or i == 0:
                print(f"[RandomForest] Trained {i+1}/{self.n_estimators} trees")

        end = time.time()
        self.training_time_ = end - start
        

    def predict(self, X):
        if len(self.trees) == 0:
            return np.zeros(len(X), dtype=int)

        
        all_preds = np.zeros((len(X), self.n_estimators), dtype=int)
        for i, tree in enumerate(self.trees):
            all_preds[:, i] = tree.predict(X)

        
        y_pred = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            labels, counts = np.unique(all_preds[i], return_counts=True)
            y_pred[i] = labels[np.argmax(counts)]
        return y_pred



def train_and_validate_random_forest(X_train, y_train, X_val, y_val):
    
    print("   TRAINING RANDOM FOREST")
    

    rf = RandomForestClassifier(
        n_estimators=20,
        subsample=0.7,
        feature_subsample=0.3,
        n_thresholds=10,
        max_depth=8,
        min_samples_split=10,
        num_classes=10
    )

    # ----- TRAIN -----
    t0 = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - t0

    # ----- VALIDATE -----
    t0 = time.time()
    y_pred = rf.predict(X_val)
    pred_time = time.time() - t0

    # ----- METRICS -----
    acc = accuracy_score(y_val, y_pred)
    mp, mr, mf1, microp, micror, microf1 = compute_metrics(y_val, y_pred)

    
    print("     RANDOM FOREST RESULTS")
    
    print(f"Accuracy         : {acc:.4f}")
    print(f"Macro Precision  : {mp:.4f}")
    print(f"Macro Recall     : {mr:.4f}")
    print(f"Macro F1 Score   : {mf1:.4f}")
    print(f"Micro Precision  : {microp:.4f}")
    print(f"Micro Recall     : {micror:.4f}")
    print(f"Micro F1 Score   : {microf1:.4f}")
    print(f"Training Time    : {train_time:.2f} sec")
    print(f"Prediction Time  : {pred_time:.2f} sec")
    print("==============================\n")

    return {
        "accuracy": acc,
        "macro_p": mp,
        "macro_r": mr,
        "macro_f1": mf1,
        "micro_p": microp,
        "micro_r": micror,
        "micro_f1": microf1,
        "train_time": train_time,
        "pred_time": pred_time
    }


X_train_raw, y_train, X_val_raw, y_val = load_mnist()
rf_results = train_and_validate_random_forest(X_train_raw, y_train, X_val_raw, y_val)
print("Random Forest Results:", rf_results)




def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def add_bias(X):
    m = X.shape[0]
    return np.hstack([np.ones((m,1)), X])

def accuracy_score(y_true,y_pred):
    return np.mean(y_true==y_pred)

def compute_metrics(y_true,y_pred,num_classes=10):
    eps = 1e-12
    precisions=[]; recalls=[]; f1s=[]
    tp_global=fp_global=fn_global=0

    for c in range(num_classes):
        tp=np.sum((y_pred==c)&(y_true==c))
        fp=np.sum((y_pred==c)&(y_true!=c))
        fn=np.sum((y_pred!=c)&(y_true==c))

        tp_global+=tp; fp_global+=fp; fn_global+=fn

        p=tp/(tp+fp+eps)
        r=tp/(tp+fn+eps)
        f1=2*p*r/(p+r+eps)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_p=np.mean(precisions)
    macro_r=np.mean(recalls)
    macro_f1=np.mean(f1s)

    micro_p=tp_global/(tp_global+fp_global+eps)
    micro_r=tp_global/(tp_global+fn_global+eps)
    micro_f1=2*micro_p*micro_r/(micro_p+micro_r+eps)

    return macro_p,macro_r,macro_f1,micro_p,micro_r,micro_f1


class PCA:
    def __init__(self,n_components=60,max_samples_for_fit=20000):
        self.n_components=n_components
        self.max_samples_for_fit=max_samples_for_fit

    def fit(self,X):
        m=X.shape[0]
        if m>self.max_samples_for_fit:
            idx=np.random.choice(m,self.max_samples_for_fit,replace=False)
            X=X[idx]

        self.mean_=np.mean(X,axis=0)
        Xc=X-self.mean_
        U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
        self.components_=Vt[:self.n_components].T

    def transform(self,X):
        return (X-self.mean_) @ self.components_
    

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






# LOGISTIC REGRESSION OVR

def _fit_binary_logistic(X,y,lr,epochs,bs):
    Xb=add_bias(X)
    m,n=Xb.shape
    y=y.reshape(-1,1)
    w=np.zeros((n,1))

    for e in range(epochs):
        idx=np.random.permutation(m)
        Xb_sh=Xb[idx]; y_sh=y[idx]
        for i in range(0,m,bs):
            Xb_i=Xb_sh[i:i+bs]
            y_i=y_sh[i:i+bs]
            pred=sigmoid(Xb_i@w)
            grad=Xb_i.T@(pred-y_i)/Xb_i.shape[0]
            w-=lr*grad

    return w

class OvRLogisticRegression:
    def __init__(self,lr=0.05,epochs=8,bs=128,num_classes=10):
        self.lr=lr; self.epochs=epochs; self.bs=bs; self.C=num_classes

    def fit(self,X,y):
        Xb=add_bias(X)
        m,n=Xb.shape
        self.W=np.zeros((self.C,n))
        for c in range(self.C):
            print(f"[OvR] Training class {c}")
            w=_fit_binary_logistic(X,(y==c).astype(int),
                                   self.lr,self.epochs,self.bs)
            self.W[c]=w.ravel()

    def predict(self,X):
        Xb=add_bias(X)
        logits=Xb@self.W.T
        return np.argmax(sigmoid(logits),axis=1)
    


logi = OvRLogisticRegression()
train_model(logi, X_train, y_train, "Logistic OvR")

logi_pred = validate_model(logi, X_val, y_val, "Logistic OvR")
print_precision_recall_f1(y_val, logi_pred, "Logistic OvR")



# MODEL EVALUATION

def evaluate_model(name,model,X_train,y_train,X_val,y_val,train=True):
    print("\n"+"="*80)
    print("MODEL:",name)
    print("="*80)

    if train:
        t0=time.time()
        model.fit(X_train,y_train)
        train_time=time.time()-t0
    else:
        train_time=0.0

    t0=time.time()
    y_pred=model.predict(X_val)
    pred_time=time.time()-t0

    acc=accuracy_score(y_val,y_pred)
    (mp, mr, mf1, micp, micr, micf1)=compute_metrics(y_val,y_pred)

    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro Precision : {mp:.4f}")
    print(f"Macro Recall    : {mr:.4f}")
    print(f"Macro F1        : {mf1:.4f}")
    print(f"Micro Precision : {micp:.4f}")
    print(f"Micro Recall    : {micr:.4f}")
    print(f"Micro F1        : {micf1:.4f}")
    print(f"Training Time   : {train_time:.2f}")
    print(f"Prediction Time : {pred_time:.2f}")

    return {
        "name":name,
        "accuracy":acc,
        "macro_p":mp,
        "macro_r":mr,
        "macro_f1":mf1,
        "micro_p":micp,
        "micro_r":micr,
        "micro_f1":micf1,
        "train_time":train_time,
        "pred_time":pred_time
    }



# SOFTMAX REGRESSION

class SoftmaxRegression:
    def __init__(self,lr=0.1,epochs=25,num_classes=10):
        self.lr=lr
        self.epochs=epochs
        self.C=num_classes

    def fit(self,X,y):
        m,n=X.shape
        self.W=np.zeros((n,self.C))
        self.b=np.zeros((1,self.C))

        Y=np.zeros((m,self.C))
        Y[np.arange(m),y]=1

        for e in range(1,self.epochs+1):
            logits=X@self.W+self.b
            probs=softmax(logits)

            grad=(probs-Y)/m
            self.W-=self.lr*(X.T@grad)
            self.b-=self.lr*np.sum(grad,axis=0,keepdims=True)

            if e%5==0:
                loss=-np.mean(np.sum(Y*np.log(probs+1e-12),axis=1))
                print(f"[Softmax] Epoch {e}/{self.epochs}, Loss={loss:.4f}")

    def predict(self,X):
        return np.argmax(softmax(X@self.W+self.b),axis=1)





class XGBStump:
    def __init__(self, lambda_reg=1.0):
        self.f_idx = None
        self.thresh = None
        self.left_val = None
        self.right_val = None
        self.lambda_reg = lambda_reg

    def fit(self, X, g, h, n_thresholds=20, feature_subsample=0.4):
        m, n = X.shape
        n_feats = max(1, int(n * feature_subsample))
        feat_idx_list = np.random.choice(n, n_feats, replace=False)

        best_gain = -1e18
        G_total = np.sum(g)
        H_total = np.sum(h)

        parent_gain = (G_total**2) / (H_total + self.lambda_reg)

        for f in feat_idx_list:
            Xf = X[:, f]
            thresholds = np.quantile(Xf, np.linspace(0.05, 0.95, n_thresholds))

            for t in np.unique(thresholds):
                left = (Xf <= t)
                if left.sum() == 0 or left.sum() == m:
                    continue

                G_left = g[left].sum()
                H_left = h[left].sum()
                G_right = G_total - G_left
                H_right = H_total - H_left

                gain = (G_left**2)/(H_left+self.lambda_reg) + \
                       (G_right**2)/(H_right+self.lambda_reg) - parent_gain

                if gain > best_gain:
                    best_gain = gain
                    self.f_idx = f
                    self.thresh = t
                    self.left_val  = - G_left  / (H_left  + self.lambda_reg)
                    self.right_val = - G_right / (H_right + self.lambda_reg)

    def predict(self, X):
        col = X[:, self.f_idx]
        return np.where(col <= self.thresh, self.left_val, self.right_val)

class XGBoostBinary:
    def __init__(self, n_estimators=120, lr=0.15, lambda_reg=1.0):
        self.n_estimators = n_estimators
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.stumps = []
        self.base_score = 0.0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    def fit(self, X, y):
        eps = 1e-9
        
        p0 = np.clip(np.mean(y), eps, 1 - eps)
        self.base_score = np.log(p0/(1-p0))

        y_pred = np.full(len(y), self.base_score)

        for i in range(self.n_estimators):
            p = self.sigmoid(y_pred)
            g = p - y
            h = p * (1 - p)

            stump = XGBStump(lambda_reg=self.lambda_reg)
            stump.fit(X, g, h)

            update = stump.predict(X)
            y_pred += self.lr * update

            self.stumps.append(stump)

            if (i+1) % 20 == 0:
                print(f"[XGB-Binary] Tree {i+1}/{self.n_estimators}")
                

    def predict_proba(self, X):
        pred = np.full(X.shape[0], self.base_score)
        for stump in self.stumps:
            pred += self.lr * stump.predict(X)
        return self.sigmoid(pred)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class XGBoostMulti:
    def __init__(self, num_classes=10, n_estimators=120, lr=0.15):
        self.C = num_classes
        self.models = [
            XGBoostBinary(n_estimators=n_estimators, lr=lr)
            for _ in range(self.C)
        ]

    def fit(self, X, y):
        print("\n TRAINING XGBOOST (MULTI-CLASS) ")
        for c in range(self.C):
            
            y_binary = (y == c).astype(int)
            self.models[c].fit(X, y_binary)

    def predict(self, X):
        scores = np.zeros((X.shape[0], self.C))
        for c in range(self.C):
            scores[:, c] = self.models[c].predict_proba(X)
        return np.argmax(scores, axis=1)
    

    xgb = XGBoostMulti(num_classes=10, n_estimators=120, lr=0.15)
    train_model(xgb, X_train, y_train, "XGBoost Multi")
    xgb_pred = validate_model(xgb, X_val, y_val, "XGBoost Multi")
    print_precision_recall_f1(y_val, xgb_pred, "XGBoost Multi")






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

X_train, y_train, X_val, y_val = load_mnist()



knn = KNNClassifier(k=5, max_train_samples=20000)


print("\nTraining KNN model...")
t0 = time.time()
knn.fit(X_train, y_train)
train_time = time.time() - t0
print(f"KNN Training Time: {train_time:.2f}s")

train_preds = evaluate_knn_train(knn, X_train, y_train)


val_preds = evaluate_knn_val(knn, X_val, y_val)





# MAIN
if __name__=="__main__":
    np.random.seed(42)

    X_train_raw,y_train,X_val_raw,y_val=load_mnist()

    print("\nFitting PCA...")
    pca=PCA(n_components=60)
    pca.fit(X_train_raw)
    X_train=pca.transform(X_train_raw)
    X_val=pca.transform(X_val_raw)
    print("PCA completed. Features:",X_train.shape[1])

    results=[]

    
    soft=SoftmaxRegression()
    results.append(evaluate_model("Softmax",soft,X_train,y_train,X_val,y_val))

    
    logi=OvRLogisticRegression()
    results.append(evaluate_model("Logistic OvR",logi,X_train,y_train,X_val,y_val))
   
   
    results.append(evaluate_model("Random Forest", rf_clf, X_train, y_train, X_val, y_val))
    # kNN
    
    
    xgb = XGBoostMulti(num_classes=10, n_estimators=120, lr=0.15)
    results.append(evaluate_model("XGBoost (NumPy)", xgb, X_train, y_train, X_val, y_val))

    print("\n\nSUMMARY\n"+"-"*60)
    for r in results:
        print(r)





import numpy as np

class Stacked_KNN_Softmax:
    

    def __init__(self, knn_model, softmax_model, meta_model):
        self.knn = knn_model
        self.soft = softmax_model
        self.meta = meta_model

    def fit(self, X, y):
        print("\n[Stack] Training KNN...")
        self.knn.fit(X, y)

        print("[Stack] Training Softmax...")
        self.soft.fit(X, y)

        print("[Stack] Training Meta-Model (OvR Logistic)...")
        knn_pred = self.knn.predict(X)
        soft_pred = self.soft.predict(X)

        
        stacked_X = np.column_stack((knn_pred, soft_pred))

        self.meta.fit(stacked_X, y)

    def predict(self, X):
        knn_pred = self.knn.predict(X)
        soft_pred = self.soft.predict(X)

        stacked_X = np.column_stack((knn_pred, soft_pred))

        return self.meta.predict(stacked_X)




import numpy as np

class Stacked_KNN_RF_Logistic:
    
    def __init__(self, knn_model, rf_model, logistic_model, meta_softmax):
        self.knn = knn_model
        self.rf = rf_model
        self.logi = logistic_model
        self.meta = meta_softmax

    def fit(self, X, y):
        print("\n[Stack] Training KNN...")
        self.knn.fit(X, y)

        print("[Stack] Training Random Forest...")
        self.rf.fit(X, y)

        print("[Stack] Training Logistic OvR...")
        self.logi.fit(X, y)

        print("[Stack] Training Meta Softmax...")
        p1 = self.knn.predict(X)
        p2 = self.rf.predict(X)
        p3 = self.logi.predict(X)

        stacked_X = np.column_stack((p1, p2, p3))
        self.meta.fit(stacked_X, y)

    def predict(self, X):
        p1 = self.knn.predict(X)
        p2 = self.rf.predict(X)
        p3 = self.logi.predict(X)

        stacked_X = np.column_stack((p1, p2, p3))
        return self.meta.predict(stacked_X)



import numpy as np

class Stacked_Logistic_XGB_RF:
    

    def __init__(self, logistic_model, xgb_model, rf_model, meta_knn):
        self.logi = logistic_model
        self.xgb = xgb_model
        self.rf = rf_model
        self.meta = meta_knn

    def fit(self, X, y):
        print("\n[Stack] Training Logistic OvR...")
        self.logi.fit(X, y)

        print("[Stack] Training XGBoost...")
        self.xgb.fit(X, y)

        print("[Stack] Training Random Forest...")
        self.rf.fit(X, y)

        print("[Stack] Training Meta KNN...")
        p1 = self.logi.predict(X)
        p2 = self.xgb.predict(X)
        p3 = self.rf.predict(X)

        stacked_X = np.column_stack((p1, p2, p3))

        self.meta.fit(stacked_X, y)

    def predict(self, X):
        p1 = self.logi.predict(X)
        p2 = self.xgb.predict(X)
        p3 = self.rf.predict(X)

        stacked_X = np.column_stack((p1, p2, p3))
        return self.meta.predict(stacked_X)



class SimpleStacker:
    
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        print("\n[STACK] Training base models...")
        base_preds = []

        for model in self.base_models:
            name = model.__class__.__name__
            print(f"[STACK] Training base model: {name}")
            model.fit(X, y)
            base_preds.append(model.predict(X))

        # Combine predictions as features
        stacked_X = np.column_stack(base_preds)

        print("[STACK] Training meta model...")
        self.meta_model.fit(stacked_X, y)

    def predict(self, X):
        base_preds = []

        for model in self.base_models:
            base_preds.append(model.predict(X))

        stacked_X = np.column_stack(base_preds)

        return self.meta_model.predict(stacked_X)



stack1 = SimpleStacker(
    base_models = [
        KNNClassifier(k=5),
        SoftmaxRegression()
    ],
    meta_model = OvRLogisticRegression()
)
results.append(
    evaluate_model("Stack: KNN + Softmax → Logistic", stack1,
                   X_train, y_train, X_val, y_val)
)




stack2 = SimpleStacker(
    base_models=[
        KNNClassifier(k=5),
        RandomForestClassifier(...),
        OvRLogisticRegression()
    ],
    meta_model=SoftmaxRegression()
)



stack2 = SimpleStacker(
    base_models = [
        KNNClassifier(k=5),
        RandomForestClassifier(
            n_estimators=15,
            subsample=0.7,
            max_depth=7,
            min_samples_split=10,
            feature_subsample=0.5,
            n_thresholds=10,
            num_classes=10
        ),
        OvRLogisticRegression()
    ],
    meta_model = SoftmaxRegression()
)
results.append(
    evaluate_model("Stack: KNN + RF + Logistic → Softmax", stack2,
                   X_train, y_train, X_val, y_val)
)




stack3 = SimpleStacker(
    base_models=[
        OvRLogisticRegression(
            lr=0.05,
            epochs=8,
            bs=128,
            num_classes=10
        ),

        XGBoostMulti(
            num_classes=10,
            n_estimators=120,
            lr=0.15
        ),

        RandomForestClassifier(
            n_estimators=15,
            subsample=0.7,
            max_depth=7,
            min_samples_split=10,
            feature_subsample=0.5,
            n_thresholds=10,
            num_classes=10
        )
    ],
    meta_model=KNNClassifier(k=5)
)



results.append(
    evaluate_model("Stack 3: Logistic + XGB + RF → KNN",
                   stack3,
                   X_train, y_train,
                   X_val, y_val)
)
