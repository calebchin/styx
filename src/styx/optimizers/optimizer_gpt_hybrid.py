from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
#  DATA HELPERS (with optional normalization)
# ============================================================

def load_digits_split(normalize=True, train_size=100, test_size=50):
    digits = load_digits()
    X = digits.data.astype("float32")
    y = digits.target.astype("float32")

    xTr, xTe, yTr, yTe = train_test_split(X, y, test_size=0.2, random_state=1)
    xTr = xTr[:train_size]
    yTr = yTr[:train_size]
    xTe = xTe[:test_size]
    yTe = yTe[:test_size]

    if normalize:
        X_mean = xTr.mean(axis=0, keepdims=True)
        X_std = xTr.std(axis=0, keepdims=True) + 1e-8
        xTr = (xTr - X_mean) / X_std
        xTe = (xTe - X_mean) / X_std

        y_mean = yTr.mean()
        y_std = yTr.std() + 1e-8
        yTr = (yTr - y_mean) / y_std
        yTe = (yTe - y_mean) / y_std

        norm_params = (X_mean, X_std, y_mean, y_std)
    else:
        norm_params = None

    return xTr, xTe, yTr, yTe, norm_params


# ============================================================
#  TORCH BASELINE MODEL (with optional nonlinearity)
# ============================================================

class BaselineModel(nn.Module):
    def __init__(self, use_nonlinear=True):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.use_nonlinear = use_nonlinear
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        if self.use_nonlinear:
            x = self.act(x)
        x = self.fc2(x)
        if self.use_nonlinear:
            x = self.act(x)
        x = self.fc3(x)
        return x


def torch_mse_loss(output, target):
    # target is float (possibly normalized)
    return ((output.flatten() - target) ** 2).mean()


def make_torch_loaders(batch_size=16, normalize=True):
    xTr, xTe, yTr, yTe, norm_params = load_digits_split(normalize=normalize)

    xTr_t = torch.from_numpy(xTr)
    xTe_t = torch.from_numpy(xTe)
    yTr_t = torch.from_numpy(yTr)
    yTe_t = torch.from_numpy(yTe)

    train_loader = DataLoader(TensorDataset(xTr_t, yTr_t), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(xTe_t, yTe_t), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, norm_params


# ============================================================
#  PURE SGD BASELINE (TORCH)
# ============================================================

def torch_baseline_sgd(epochs=20, lr=1e-3, use_nonlinear=True):
    print("\n========== TORCH SGD BASELINE ==========")
    train_loader, test_loader, norm_params = make_torch_loaders(batch_size=16, normalize=True)

    model = BaselineModel(use_nonlinear=use_nonlinear)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = torch_mse_loss(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * data.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = torch_mse_loss(output, target)
                total_test_loss += loss.item() * data.size(0)
        avg_test_loss = total_test_loss / len(test_loader.dataset)

        if (epoch + 1) % 10 == 0:
          print(f"[SGD] Epoch {epoch+1:02d} | Train: {avg_train_loss:.4f} | Test: {avg_test_loss:.4f}")

    # Show a few predictions (denormalized for intuition)
    X_mean, X_std, y_mean, y_std = norm_params
    print("\n[SGD] Sample predictions (denormalized):")
    with torch.no_grad():
        for i in range(5):
            x, y_true_n = train_loader.dataset[i]
            y_pred_n = model(x).item()
            # denormalize
            y_pred = y_pred_n * y_std + y_mean
            y_true = y_true_n.item() * y_std + y_mean
            print(f"  pred={y_pred:.3f}, label={y_true:.3f}, error={y_pred-y_true:.3f}")


# ============================================================
#  TORCH PARAMETER FLATTEN / UNFLATTEN UTILITIES
# ============================================================

def get_param_vector(model):
    """
    Flatten all parameters into a single 1D torch tensor.
    """
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def set_param_vector(model, vec):
    """
    Set model parameters from a single 1D torch tensor 'vec'.
    """
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        new_vals = vec[idx: idx + numel].view_as(p)
        p.data.copy_(new_vals)
        idx += numel


# ============================================================
#  HYBRID OPTIMIZER: SGD + ES-STYLE ZEROTH-ORDER STEP
# ============================================================

def es_refinement_step(
    model,
    train_loader,
    sigma=0.01,
    lr=1e-3,
    popsize=32,
    max_batches=2,
    device="cpu"
):
    """
    One ES step:
      - sample eps ~ N(0, I)
      - evaluate fitness (negative loss) for w +/- sigma * eps
      - normalize fitness
      - estimate gradient in param space
      - take a small step in that direction
    Uses only a small subset of train batches (max_batches) to keep it cheap.
    """
    model.to(device)
    base_params = get_param_vector(model).to(device)
    dim = base_params.numel()

    # Collect a small subset of data for ES evaluation
    batches = []
    for i, (data, target) in enumerate(train_loader):
        batches.append((data.to(device), target.to(device)))
        if i + 1 >= max_batches:
            break

    # Helper for loss under a given parameter vector
    def eval_loss_for_params(param_vec):
        set_param_vector(model, param_vec)
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in batches:
                output = model(data)
                loss = torch_mse_loss(output, target)
                total_loss += loss.item() * data.size(0)
                count += data.size(0)
        return total_loss / max(count, 1)

    # Sample perturbations
    eps = torch.randn(popsize, dim, device=device)

    # Evaluate plus/minus
    losses_plus = []
    losses_minus = []
    for i in range(popsize):
        w_plus = base_params + sigma * eps[i]
        w_minus = base_params - sigma * eps[i]
        loss_p = eval_loss_for_params(w_plus)
        loss_m = eval_loss_for_params(w_minus)
        losses_plus.append(loss_p)
        losses_minus.append(loss_m)

    losses_plus = torch.tensor(losses_plus, device=device)
    losses_minus = torch.tensor(losses_minus, device=device)

    # Fitness = -loss (we want to maximize fitness)
    fitness_plus = -losses_plus
    fitness_minus = -losses_minus

    fitness = torch.cat([fitness_plus, fitness_minus], dim=0)  # (2 * popsize)
    eps_all = torch.cat([eps, -eps], dim=0)                     # (2 * popsize, dim)

    # Normalize fitness
    fitness = (fitness - fitness.mean()) / (fitness.std() + 1e-8)

    # Gradient estimate in parameter space
    grad_est = (fitness.view(-1, 1) * eps_all).mean(dim=0) / sigma

    # Optional gradient clipping
    max_norm = 1.0
    gnorm = torch.norm(grad_est)
    if gnorm > max_norm:
        grad_est = grad_est * (max_norm / (gnorm + 1e-8))

    # Ascend in fitness (descend in loss)
    new_params = base_params + lr * grad_est
    set_param_vector(model, new_params)


def hybrid_torch_optimizer(
    sgd_epochs=10,
    sgd_lr=1e-3,
    es_after_each_epoch=True,
    es_sigma=0.01,
    es_lr=1e-3,
    es_popsize=32,
    es_max_batches=2,
    use_nonlinear=True
):
    """
    Train with SGD, and after each epoch perform one ES refinement step.
    """
    print("\n========== HYBRID OPTIMIZER (SGD + ES) ==========")
    train_loader, test_loader, norm_params = make_torch_loaders(batch_size=16, normalize=True)

    device = "cpu"
    model = BaselineModel(use_nonlinear=use_nonlinear).to(device)
    optimizer = optim.SGD(model.parameters(), lr=sgd_lr)

    for epoch in range(sgd_epochs):
        # ----- SGD phase -----
        model.train()
        total_train_loss = 0.0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch_mse_loss(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * data.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # ----- ES refinement phase -----
        if es_after_each_epoch:
            es_refinement_step(
                model,
                train_loader,
                sigma=es_sigma,
                lr=es_lr,
                popsize=es_popsize,
                max_batches=es_max_batches,
                device=device
            )

        # Evaluate
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = torch_mse_loss(output, target)
                total_test_loss += loss.item() * data.size(0)
        avg_test_loss = total_test_loss / len(test_loader.dataset)

        if (epoch + 1) % 10 == 0:
          print(f"[Hybrid] Epoch {epoch+1:02d} | Train: {avg_train_loss:.4f} | Test: {avg_test_loss:.4f}")

    # Show a few denormalized predictions
    X_mean, X_std, y_mean, y_std = norm_params
    print("\n[Hybrid] Sample predictions (denormalized):")
    with torch.no_grad():
        for i in range(5):
            x, y_true_n = train_loader.dataset[i]
            x = x.to(device)
            y_pred_n = model(x.unsqueeze(0)).item()
            y_pred = y_pred_n * y_std + y_mean
            y_true = y_true_n.item() * y_std + y_mean
            print(f"  pred={y_pred:.3f}, label={y_true:.3f}, error={y_pred-y_true:.3f}")


# ============================================================
#  NUMPY MODEL (WITH NONLINEARITY) + ZERO-ORDER EXPERIMENTS
# ============================================================

INPUT_DIM = 64
H1 = 128
H2 = 128
OUT = 1
NUM_PARAMS = H1 * INPUT_DIM + H2 * H1 + OUT * H2


def init_weights_numpy(scale=0.1):
    return np.random.randn(NUM_PARAMS) * scale


def unpack_weights_numpy(w):
    idx1 = H1 * INPUT_DIM
    idx2 = idx1 + H2 * H1
    W1 = w[:idx1].reshape(H1, INPUT_DIM)
    W2 = w[idx1:idx2].reshape(H2, H1)
    W3 = w[idx2:].reshape(OUT, H2)
    return W1, W2, W3


def relu_np(z):
    return np.maximum(z, 0.0)


def forward_batch_numpy(X, w, activation=relu_np):
    W1, W2, W3 = unpack_weights_numpy(w)
    h1 = W1 @ X.T        # (H1, N)
    if activation is not None:
        h1 = activation(h1)
    h2 = W2 @ h1         # (H2, N)
    if activation is not None:
        h2 = activation(h2)
    out = W3 @ h2        # (1, N)
    return out.flatten()


def mse_loss_numpy(X, y, w, activation=relu_np):
    preds = forward_batch_numpy(X, w, activation=activation)
    return np.mean((preds - y) ** 2)


def random_search_optimize_numpy(X, y, w0, step_scale=0.05, iters=30, candidates=32, activation=relu_np):
    best_w = w0.copy()
    best_loss = mse_loss_numpy(X, y, best_w, activation=activation)
    print(f"[RandomSearch] Initial loss: {best_loss:.4f}")
    dim = w0.size
    for t in range(iters):
        noises = np.random.randn(candidates, dim) * step_scale
        ws = best_w[None, :] + noises
        losses = np.array([mse_loss_numpy(X, y, w, activation=activation) for w in ws])
        idx_best = np.argmin(losses)
        if losses[idx_best] < best_loss:
            best_loss = losses[idx_best]
            best_w = ws[idx_best].copy()
        if (t + 1) % 5 == 0 or t == 0:
            print(f"[RandomSearch] Iter {t+1:03d} | best_loss = {best_loss:.4f}")
    return best_w, best_loss


def zero_order_numpy_experiments():
    print("\n========== NUMPY ZERO-ORDER (RELU) ==========")
    xTr, xTe, yTr, yTe, _ = load_digits_split(normalize=True)

    w0 = init_weights_numpy(scale=0.1)
    rs_w, rs_loss_tr = random_search_optimize_numpy(
        xTr, yTr, w0,
        step_scale=0.05,
        iters=30,
        candidates=32,
        activation=relu_np
    )
    rs_loss_te = mse_loss_numpy(xTe, yTe, rs_w, activation=relu_np)
    print(f"[RandomSearch] Final train loss: {rs_loss_tr:.4f}, test loss: {rs_loss_te:.4f}")
    preds = forward_batch_numpy(xTe, rs_w, activation=relu_np)
    print("\n[RandomSearch] Sample predictions (normalized y):")
    for i in range(5):
        print(f"  pred={preds[i]:.3f}, label={yTe[i]:.3f}, error={preds[i]-yTe[i]:.3f}")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    # 1) Pure SGD baseline (toggle use_nonlinear)
    torch_baseline_sgd(epochs=700, lr=1e-3, use_nonlinear=True)

    # 2) Hybrid optimizer: SGD + ES refinement
    hybrid_torch_optimizer(
        sgd_epochs=700,
        sgd_lr=1e-3,
        es_after_each_epoch=True,
        es_sigma=0.01,
        es_lr=1e-3,
        es_popsize=32,
        es_max_batches=2,
        use_nonlinear=True
    )

    # 3) NumPy zero-order with ReLU (simple random search)
    # zero_order_numpy_experiments()




# AVI NOTES (don't try to do all at same time! Discuss approaches and ideas)

# 1. Adding randomization can help, but it seems like it is not good towards the end of runs (running 700 epochs)
   # a. implication, small gradient steps once nearing local minima may be good to switch back
   # to standard sgd for more stabile descent into minima, then revert back to hybrid approach
# 2. Having multiple starting points optimized at the same time (evolutionarily choose better starting points)
   # b. idea: any point could be the random starting point, so iteratively maintain 10 locations to optimize
   # optimize from those points, preserve a certain subset (e.g. 5), and then spawn 5 new points
   # near the 5 that are currently doing well (don't necessarily have to do 5 and 5, probably better 
   # to do something like best 5 out of 100 and respawn 95)
# 3. For randomized approach, SGD provides some sort of eqiuvalent idea, we need to leverage what makes 
#  a randomized approach potentially better
  # a. SGD udpates are noisy but still require backpropagation, randomization has benefit that you can
  # try many more possible directions in same computation, but the downside is that it's not 
  # necessarily a better update than SGD
# 4. We should try this idea on smoother loss landscape and a more jagged loss landscape
  # a. Not entirely sure how to measure "smoothness" of the loss landscape given a dataset
  # but if it is at all possible, my intuition is that this approach works better on jagged landscapes
# 5. Another huge benefit of a zero-grad approach is that you can utilize models that are non-differentiable
  # a. What sort of models are non-differentiable? 
  # b. If non-differentiable models exist, how can we use zero-grad (and optimize) to make them work?

