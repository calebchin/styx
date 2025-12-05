from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
#  TORCH BASELINE MODEL (gradient-based SGD)
# ============================================================

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # keep it linear for apples-to-apples vs your numpy model
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def torch_loss(output, target):
    # MSE between predicted scalar and digit label
    loss = (output.flatten() - target.float()) ** 2
    return loss.mean()


def torch_baseline():
    digits = load_digits()
    X = digits.data.astype("float32")
    y = digits.target.astype("int64")

    xTr, xTe, yTr, yTe = train_test_split(X, y, test_size=0.2, random_state=1)
    xTr = xTr[:100]
    yTr = yTr[:100]
    xTe = xTe[:50]
    yTe = yTe[:50]

    epochs = 20
    batch_size = 10
    learning_rate = 1e-3

    model = BaselineModel()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    xTr_t = torch.from_numpy(xTr)
    xTe_t = torch.from_numpy(xTe)
    yTr_t = torch.from_numpy(yTr)
    yTe_t = torch.from_numpy(yTe)

    train_loader = DataLoader(TensorDataset(xTr_t, yTr_t), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(xTe_t, yTe_t), batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = torch_loss(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * data.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = torch_loss(output, target)
                total_test_loss += loss.item() * data.size(0)
        avg_test_loss = total_test_loss / len(test_loader.dataset)

        print(f"[Torch] Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # Show a few preds
    model.eval()
    print("\n[Torch] Sample predictions (train):")
    with torch.no_grad():
        for i in range(5):
            x = xTr_t[i]
            label = yTr_t[i].item()
            pred = model(x).item()
            print(f"  pred={pred:.3f}, label={label}, error={pred-label:.3f}")

    print("\n[Torch] Sample predictions (test):")
    with torch.no_grad():
        for i in range(5):
            x = xTe_t[i]
            label = yTe_t[i].item()
            pred = model(x).item()
            print(f"  pred={pred:.3f}, label={label}, error={pred-label:.3f}")


# ============================================================
#  NUMPY MODEL (same architecture) + UTILITIES
# ============================================================

INPUT_DIM = 64
H1 = 128
H2 = 128
OUT = 1

NUM_PARAMS = H1*INPUT_DIM + H2*H1 + OUT*H2  # 64*128 + 128*128 + 1*128


def init_weights(scale=0.1):
    # Gaussian init is usually better behaved than uniform [-0.5,0.5]
    return np.random.randn(NUM_PARAMS) * scale


def unpack_weights(w):
    """Turn flat vector into (W1, W2, W3)."""
    idx1 = H1 * INPUT_DIM
    idx2 = idx1 + H2 * H1

    W1 = w[:idx1].reshape(H1, INPUT_DIM)
    W2 = w[idx1:idx2].reshape(H2, H1)
    W3 = w[idx2:].reshape(OUT, H2)
    return W1, W2, W3


def forward_batch(X, w, activation=None):
    """
    X: (N, 64)
    w: (NUM_PARAMS,)
    returns: preds shape (N,)
    """
    W1, W2, W3 = unpack_weights(w)
    # shape: (H1, N)
    h1 = W1 @ X.T
    if activation is not None:
        h1 = activation(h1)
    # shape: (H2, N)
    h2 = W2 @ h1
    if activation is not None:
        h2 = activation(h2)
    # shape: (1, N)
    out = W3 @ h2
    return out.flatten()


def mse_loss(X, y, w, activation=None):
    preds = forward_batch(X, w, activation=activation)
    return np.mean((preds - y) ** 2)


# ============================================================
#  ZERO-ORDER OPTIMIZER 1: Random Search / Hill Climbing
# ============================================================

def random_search_optimize(X, y, w0, step_scale=0.1, iters=50, candidates=32, activation=None):
    """
    Simple hill climbing:
      - sample 'candidates' perturbations around current best
      - keep the best if it improves loss
    """
    best_w = w0.copy()
    best_loss = mse_loss(X, y, best_w, activation=activation)
    print(f"[RandomSearch] Initial loss: {best_loss:.4f}")

    dim = w0.size
    for t in range(iters):
        noises = np.random.randn(candidates, dim) * step_scale
        ws = best_w[None, :] + noises   # (candidates, dim)

        losses = np.array([mse_loss(X, y, w, activation=activation) for w in ws])
        idx_best = np.argmin(losses)

        if losses[idx_best] < best_loss:
            best_loss = losses[idx_best]
            best_w = ws[idx_best].copy()

        if (t+1) % 5 == 0 or t == 0:
            print(f"[RandomSearch] Iter {t+1:03d} | best_loss = {best_loss:.4f}")

    return best_w, best_loss


# ============================================================
#  ZERO-ORDER OPTIMIZER 2: Evolution Strategies–style ES
# ============================================================

def es_optimize(X, y, w0, sigma=0.01, lr=1e-3, iters=50, popsize=64, activation=None):
    """
    ES with:
      - antithetic sampling
      - fitness normalization
      - smaller sigma & lr to avoid blow-up
    """
    w = w0.copy()
    dim = w.size

    for t in range(iters):
        eps = np.random.randn(popsize, dim)

        losses_plus = np.empty(popsize)
        losses_minus = np.empty(popsize)

        for i in range(popsize):
            w_plus  = w + sigma * eps[i]
            w_minus = w - sigma * eps[i]
            losses_plus[i]  = mse_loss(X, y, w_plus,  activation=activation)
            losses_minus[i] = mse_loss(X, y, w_minus, activation=activation)

        # Turn losses into fitness (we want to maximize fitness = -loss)
        fitness_plus  = -losses_plus
        fitness_minus = -losses_minus

        fitness = np.concatenate([fitness_plus, fitness_minus], axis=0)
        eps_all = np.concatenate([eps, -eps], axis=0)

        # Normalize fitness
        fitness = (fitness - fitness.mean()) / (fitness.std() + 1e-8)

        # Gradient estimate
        grad_est = (fitness.reshape(-1, 1) * eps_all).mean(axis=0) / sigma

        # Optional gradient clipping
        max_norm = 1.0
        gnorm = np.linalg.norm(grad_est)
        if gnorm > max_norm:
            grad_est *= max_norm / (gnorm + 1e-8)

        # Gradient ascent on fitness (descent on loss)
        w = w + lr * grad_est

        current_loss = mse_loss(X, y, w, activation=activation)
        if (t + 1) % 5 == 0 or t == 0:
            print(f"[ES] Iter {t+1:03d} | loss = {current_loss:.4f}")

    final_loss = mse_loss(X, y, w, activation=activation)
    return w, final_loss


# ============================================================
#  RUNNING THE ZERO-ORDER EXPERIMENT(S)
# ============================================================

def zero_order_experiments():
    # Load and subset data (same as torch baseline)
    digits = load_digits()
    X = digits.data.astype("float32")
    y = digits.target.astype("float32")  # keep float for MSE

    xTr, xTe, yTr, yTe = train_test_split(X, y, test_size=0.2, random_state=1)
    xTr = xTr[:100]
    yTr = yTr[:100]
    xTe = xTe[:50]
    yTe = yTe[:50]

    # No nonlinearity for now (like your original PoC)
    activation = None  # or e.g. activation = lambda z: np.maximum(z, 0)

    # ----- Random Search -----
    print("\n=== Random Search / Hill Climbing ===")
    w0 = init_weights(scale=0.1)
    rs_w, rs_loss_tr = random_search_optimize(
        xTr, yTr, w0,
        step_scale=0.05,
        iters=50,
        candidates=32,
        activation=activation
    )
    rs_loss_te = mse_loss(xTe, yTe, rs_w, activation=activation)
    print(f"[RandomSearch] Final train loss: {rs_loss_tr:.4f}, test loss: {rs_loss_te:.4f}")

    # Show a few predictions
    preds = forward_batch(xTe, rs_w, activation=activation)
    print("\n[RandomSearch] Sample predictions (test):")
    for i in range(5):
        print(f"  pred={preds[i]:.3f}, label={yTe[i]}, error={preds[i]-yTe[i]:.3f}")

    # ----- ES Optimization -----
    print("\n=== Evolution Strategies (ES) ===")
    w0 = init_weights(scale=0.1)
    es_w, es_loss_tr = es_optimize(
        xTr, yTr, w0,
        sigma=0.05,
        lr=0.05,
        iters=50,
        popsize=64,
        activation=activation
    )
    es_loss_te = mse_loss(xTe, yTe, es_w, activation=activation)
    print(f"[ES] Final train loss: {es_loss_tr:.4f}, test loss: {es_loss_te:.4f}")

    preds = forward_batch(xTe, es_w, activation=activation)
    print("\n[ES] Sample predictions (test):")
    for i in range(5):
        print(f"  pred={preds[i]:.3f}, label={yTe[i]}, error={preds[i]-yTe[i]:.3f}")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    # 1) Gradient-based baseline
    torch_baseline()

    # 2) Zero-order experiments
    zero_order_experiments()
