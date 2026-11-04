import numpy as np


class Perceptron:
    """Single layer perceptron (one neuron, step activation)."""

    def __init__(self, n_features, lr=0.1, epochs=100, shuffle=True,
                 lr_decay=1.0, seed=42):
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.normal(0, 0.01, n_features)  # small random weights
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.shuffle = shuffle
        self.lr_decay = lr_decay          # <1.0 shrinks lr every epoch
        self.errors_per_epoch = []

    @staticmethod
    def _step(z):
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._step(X @ self.w + self.b)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        lr = self.lr
        best_err, best_w, best_b = np.inf, self.w.copy(), self.b

        for _ in range(self.epochs):
            idx = self.rng.permutation(len(X)) if self.shuffle else np.arange(len(X))
            errors = 0
            for i in idx:
                pred = self._step(X[i] @ self.w + self.b)
                update = lr * (y[i] - pred)       # perceptron learning rule
                self.w += update * X[i]
                self.b += update
                errors += int(update != 0)
            self.errors_per_epoch.append(errors)

            # keep the best weights seen so far ("pocket" improvement)
            if errors < best_err:
                best_err, best_w, best_b = errors, self.w.copy(), self.b
            if errors == 0:                        # converged
                break
            lr *= self.lr_decay

        self.w, self.b = best_w, best_b
        return self

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))


# --------------------------------------------------------------------------
# Task 1: AND / OR gates
# --------------------------------------------------------------------------
def run_logic_gates():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    gates = {"AND": np.array([0, 0, 0, 1]),
             "OR":  np.array([0, 1, 1, 1])}

    for name, y in gates.items():
        # Baseline: few epochs, tiny learning rate -> may not learn fully
        base = Perceptron(2, lr=0.001, epochs=2, shuffle=False).fit(X, y)
        # Improved: more epochs, better lr, shuffling, lr decay, best-weights
        best = Perceptron(2, lr=0.1, epochs=100, shuffle=True,
                          lr_decay=0.99).fit(X, y)

        print(f"\n{name} gate")
        print(f"  baseline accuracy : {base.score(X, y) * 100:.0f}%")
        print(f"  improved accuracy : {best.score(X, y) * 100:.0f}%")
        print(f"  weights={best.w.round(3)}, bias={best.b:.3f}")
        for xi, p in zip(X, best.predict(X)):
            print(f"    {xi} -> {p}")


# --------------------------------------------------------------------------
# Task 2: Random dataset
# --------------------------------------------------------------------------
def make_random_dataset(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 2))
    # label from a hidden line plus a little noise
    y = (2 * X[:, 0] + 1.5 * X[:, 1] + rng.normal(0, 0.3, n) > 0).astype(int)
    return X, y


def run_random_dataset():
    X, y = make_random_dataset()

    # train/test split (80/20)
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]
    X_train, y_train, X_test, y_test = X[tr], y[tr], X[te], y[te]

    # feature standardisation (using train statistics only)
    mu, sd = X_train.mean(0), X_train.std(0)
    X_train, X_test = (X_train - mu) / sd, (X_test - mu) / sd

    model = Perceptron(2, lr=0.05, epochs=200, lr_decay=0.98).fit(X_train, y_train)

    print("\nRandom dataset")
    print(f"  train accuracy: {model.score(X_train, y_train) * 100:.2f}%")
    print(f"  test accuracy : {model.score(X_test, y_test) * 100:.2f}%")
    print("  first 10 predictions:", model.predict(X_test[:10]))
    print("  first 10 actual     :", y_test[:10])


if __name__ == "__main__":
    run_logic_gates()
    run_random_dataset()
