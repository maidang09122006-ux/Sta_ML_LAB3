import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

class SVM:
    def __init__(self, C: float = 1, lr: float = 0.01, n_epochs: int = 1000, random_state: int = 42):
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.losses = []

        self.W = None
        self.b = None

    # Tách riêng hàm tính Hinge Loss cho từng sample
    def hinge_loss(self, y_i: float, y_pred_val: float):
        return max(0, 1 - y_i * y_pred_val)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        N, dim = X.shape
        self.W = np.zeros((dim, ))
        self.b = 0

        pbar = tqdm(range(self.n_epochs), desc="Training SVM (Epochs)") 

        for epoch in pbar:
            indices = np.random.permutation(N)
            epoch_hinge_loss = 0.0 # Chỉ cộng dồn Hinge Loss
            
            for idx in indices:
                x_i = X[idx]
                y_i = y[idx]
                
                y_pred_array = self.predict(np.expand_dims(x_i, axis=0))
                y_pred = y_pred_array.item()
                
                # Cộng dồn Hinge Loss của điểm dữ liệu này
                epoch_hinge_loss += self.hinge_loss(y_i, y_pred)

                dW = None
                db = None
                if y_i * y_pred >= 1:
                    dW = self.W
                    db = 0
                else:
                    dW = self.W + self.C * (-y_i * x_i)
                    db = self.C * (-y_i)
                
                self.W = self.W - self.lr * dW
                self.b = self.b - self.lr * db
            
            avg_hinge_loss = epoch_hinge_loss / N
            total_epoch_loss = 0.5 * np.dot(self.W.T, self.W) + self.C * avg_hinge_loss
            
            self.losses.append(total_epoch_loss)
            pbar.set_postfix({"Total Loss": f"{total_epoch_loss:.4f}"})

    def predict(self, X: np.ndarray):
        return X @ self.W + self.b

    def get_metrics(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        y_pred = np.where(y_pred >= 0, 1, -1)

        P = precision_score(y, y_pred, pos_label=-1)
        R = recall_score(y, y_pred, pos_label=-1)
        f1= f1_score(y, y_pred, pos_label=-1)

        return {
            "Precision": P,
            "Recall": R,
            "F1": f1
        }