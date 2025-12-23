import numpy as np


class SimpleMLP:
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        lr=1e-2,
        epochs=100,
        seed=42,
        weight_decay=0.0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.weight_decay = weight_decay
        self._init_weights()

        # para logging
        self.history_ = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def _init_weights(self):
        rng = np.random.RandomState(self.seed)

        # Inicialização tipo Xavier para ReLU
        limit1 = np.sqrt(6.0 / (self.input_dim + self.hidden_dim))
        self.W1 = rng.uniform(-limit1, limit1, (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)

        limit2 = np.sqrt(6.0 / (self.hidden_dim + self.output_dim))
        self.W2 = rng.uniform(-limit2, limit2, (self.hidden_dim, self.output_dim))
        self.b2 = np.zeros(self.output_dim, dtype=np.float32)

    @staticmethod
    def _relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_derivative(z):
        return (z > 0).astype(np.float32)

    @staticmethod
    def _softmax(logits):
        # estabilidade numérica
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    @staticmethod
    def _cross_entropy(probs, y):
        n = y.shape[0]
        # evita log(0)
        log_likelihood = -np.log(probs[np.arange(n), y] + 1e-12)
        return np.mean(log_likelihood)

    @staticmethod
    def _accuracy(probs, y):
        preds = np.argmax(probs, axis=1)
        return np.mean(preds == y)

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1  # (n, hidden_dim)
        a1 = self._relu(z1)         # (n, hidden_dim)
        logits = a1 @ self.W2 + self.b2  # (n, output_dim)
        probs = self._softmax(logits)

        cache = {
            "X": X,
            "z1": z1,
            "a1": a1,
            "logits": logits,
            "probs": probs,
        }
        return probs, cache

    def _backward(self, cache, y):
        X = cache["X"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        probs = cache["probs"]

        n = X.shape[0]

        # gradiente da entropia cruzada + softmax
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n  # média

        # gradientes da camada de saída
        dW2 = a1.T @ dlogits              # (hidden_dim, output_dim)
        db2 = np.sum(dlogits, axis=0)     # (output_dim,)

        # retropropagação para a camada escondida
        da1 = dlogits @ self.W2.T         # (n, hidden_dim)
        dz1 = da1 * self._relu_derivative(z1)

        dW1 = X.T @ dz1                   # (input_dim, hidden_dim)
        db1 = np.sum(dz1, axis=0)         # (hidden_dim,)

        grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }
        return grads

    def _update_params(self, grads):
        if self.weight_decay > 0.0:
            grads["dW1"] += self.weight_decay * self.W1
            grads["dW2"] += self.weight_decay * self.W2

        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]


    def fit(self, X_train, y_train, X_val=None, y_val=None):
        for epoch in range(1, self.epochs + 1):
            # forward
            probs_train, cache = self._forward(X_train)
            train_loss = self._cross_entropy(probs_train, y_train)
            train_acc = self._accuracy(probs_train, y_train)

            # backward
            grads = self._backward(cache, y_train)
            self._update_params(grads)

            # validação (se fornecida)
            if X_val is not None and y_val is not None:
                probs_val, _ = self._forward(X_val)
                val_loss = self._cross_entropy(probs_val, y_val)
                val_acc = self._accuracy(probs_val, y_val)
            else:
                val_loss = None
                val_acc = None

            # salva histórico
            self.history_["epoch"].append(epoch)
            self.history_["train_loss"].append(train_loss)
            self.history_["train_acc"].append(train_acc)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_acc"].append(val_acc)

            # print simples (pode comentar se encher o saco)
            if epoch % 10 == 0 or epoch == 1:
                msg = f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                print(msg)

        return self

    def predict_proba(self, X):
        probs, _ = self._forward(X)
        return probs

    def predict(self, X):
        probs, _ = self._forward(X)
        return np.argmax(probs, axis=1)
