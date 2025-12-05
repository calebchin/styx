import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
#  Data helpers (normalized)
# ============================================================

def load_digits_regression(normalize=True, train_size=100, test_size=50):
    digits = load_digits()
    X = digits.data.astype("float32")
    y = digits.target.astype("float32")  # regression target 0–9

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
#  Torch model and utilities
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
    return ((output.flatten() - target) ** 2).mean()


def get_param_vector(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def set_param_vector(model, vec):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        new_vals = vec[idx: idx + numel].view_as(p)
        p.data.copy_(new_vals)
        idx += numel


def make_dataset_torch(batch_size=32, normalize=True):
    xTr, xTe, yTr, yTe, norm_params = load_digits_regression(normalize=normalize)
    xTr_t = torch.from_numpy(xTr)
    xTe_t = torch.from_numpy(xTe)
    yTr_t = torch.from_numpy(yTr)
    yTe_t = torch.from_numpy(yTe)

    train_loader = DataLoader(TensorDataset(xTr_t, yTr_t), batch_size=batch_size, shuffle=True)
    full_train_loader = DataLoader(TensorDataset(xTr_t, yTr_t), batch_size=batch_size, shuffle=False)
    full_test_loader = DataLoader(TensorDataset(xTe_t, yTe_t), batch_size=batch_size, shuffle=False)
    return train_loader, full_train_loader, full_test_loader, norm_params


# ============================================================
#  Common: evaluate loss for a given parameter vector
# ============================================================

def eval_loss_for_params(model, param_vec, data_loader, device="cpu"):
    set_param_vector(model, param_vec)
    model.to(device)
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = torch_mse_loss(output, target)
            total_loss += loss.item() * data.size(0)
            count += data.size(0)
    return total_loss / max(count, 1)


# ============================================================
#  1) Annealed zero-grad population ES
# ============================================================

def annealed_pop_es(
    pop_size=50,
    elite_frac=0.2,
    sigma_start=0.1,
    sigma_end=0.005,
    generations=100,
    use_nonlinear=True,
    device="cpu"
):
    print("\n========== Annealed Zero-Grad Population ES ==========")
    # Dataset + model
    _, full_train_loader, full_test_loader, norm_params = make_dataset_torch(
        batch_size=32, normalize=True
    )
    model = BaselineModel(use_nonlinear=use_nonlinear).to(device)

    # Initial population in parameter space
    base_params = get_param_vector(model)
    dim = base_params.numel()
    pop = []
    for _ in range(pop_size):
        # small random init around base_params
        individual = base_params + 0.1 * torch.randn_like(base_params)
        pop.append(individual)
    pop = torch.stack(pop, dim=0)  # (pop_size, dim)

    for gen in range(generations):
        # Anneal sigma linearly (you can try log/exp too)
        t = gen / max(generations - 1, 1)
        sigma = sigma_start * (1 - t) + sigma_end * t

        # Evaluate losses for each individual
        losses = []
        for i in range(pop_size):
            loss_i = eval_loss_for_params(model, pop[i], full_train_loader, device=device)
            losses.append(loss_i)
        losses = np.array(losses)
        order = np.argsort(losses)
        pop = pop[order]
        losses = losses[order]

        best_loss = losses[0]
        median_loss = np.median(losses)

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"[AnnealedES] Gen {gen+1:03d} | sigma={sigma:.4f} | best={best_loss:.4f} | median={median_loss:.4f}")

        # Elitism + mutation
        elite_num = max(1, int(pop_size * elite_frac))
        elites = pop[:elite_num]

        new_pop = [elite.clone() for elite in elites]
        while len(new_pop) < pop_size:
            parent = elites[np.random.randint(elite_num)]
            child = parent + sigma * torch.randn(dim)
            new_pop.append(child)
        pop = torch.stack(new_pop, dim=0)

    # Final best individual
    final_losses = []
    for i in range(pop_size):
        loss_i = eval_loss_for_params(model, pop[i], full_train_loader, device=device)
        final_losses.append(loss_i)
    final_losses = np.array(final_losses)
    best_idx = np.argmin(final_losses)
    best_params = pop[best_idx]
    best_train_loss = final_losses[best_idx]

    # Evaluate test loss
    test_loss = eval_loss_for_params(model, best_params, full_test_loader, device=device)
    print(f"[AnnealedES] Final train loss: {best_train_loss:.4f}, test loss: {test_loss:.4f}")

    # Sample predictions (denormalized for intuition)
    X_mean, X_std, y_mean, y_std = norm_params
    set_param_vector(model, best_params)
    model.eval()
    with torch.no_grad():
        x_sample, y_sample = next(iter(full_test_loader))
        preds_n = model(x_sample).flatten().numpy()
        ys_n = y_sample.numpy()
    print("\n[AnnealedES] Sample predictions (denormalized):")
    for i in range(5):
        y_pred = preds_n[i] * y_std + y_mean
        y_true = ys_n[i] * y_std + y_mean
        print(f"  pred={y_pred:.3f}, label={y_true:.3f}, error={y_pred-y_true:.3f}")


# ============================================================
#  2) Population + local SGD per entity (multi-optimizer)
# ============================================================

def local_sgd_refine(
    model,
    init_params,
    train_loader,
    local_epochs=1,
    lr=1e-3,
    device="cpu"
):
    """
    Run a small number of SGD epochs starting from init_params.
    Returns new param vector.
    """
    set_param_vector(model, init_params)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for _ in range(local_epochs):
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch_mse_loss(output, target)
            loss.backward()
            optimizer.step()

    # Return the updated parameter vector
    return get_param_vector(model)

def population_with_local_sgd(
    pop_size=20,
    elite_frac=0.3,
    generations=50,
    local_epochs=1,
    local_lr=1e-3,
    sigma_mutation=0.02,
    use_nonlinear=True,
    device="cpu"
):
    print("\n========== Population + Local SGD (Multi-Optimizer) ==========")
    train_loader, full_train_loader, full_test_loader, norm_params = make_dataset_torch(
        batch_size=32, normalize=True
    )
    model = BaselineModel(use_nonlinear=use_nonlinear).to(device)

    base_params = get_param_vector(model)
    dim = base_params.numel()

    # Initial population around base params
    pop = []
    for _ in range(pop_size):
        individual = base_params + 0.1 * torch.randn_like(base_params)
        pop.append(individual)
    pop = torch.stack(pop, dim=0)

    for gen in range(generations):
        # 1) Local SGD step for each individual (exploitation)
        refined_pop = []
        for i in range(pop_size):
            refined = local_sgd_refine(
                model,
                pop[i],
                train_loader,
                local_epochs=local_epochs,
                lr=local_lr,
                device=device
            )
            refined_pop.append(refined)
        pop = torch.stack(refined_pop, dim=0)

        # 2) Evaluate fitness (train loss)
        losses = []
        for i in range(pop_size):
            loss_i = eval_loss_for_params(model, pop[i], full_train_loader, device=device)
            losses.append(loss_i)
        losses = np.array(losses)
        order = np.argsort(losses)
        pop = pop[order]
        losses = losses[order]

        best_loss = losses[0]
        median_loss = np.median(losses)
        if (gen + 1) % 5 == 0 or gen == 0:
            print(f"[Pop+SGD] Gen {gen+1:03d} | best={best_loss:.4f} | median={median_loss:.4f}")

        # 3) Selection + mutation (exploration)
        elite_num = max(1, int(pop_size * elite_frac))
        elites = pop[:elite_num]

        new_pop = [elite.clone() for elite in elites]
        while len(new_pop) < pop_size:
            parent = elites[np.random.randint(elite_num)]
            child = parent + sigma_mutation * torch.randn(dim)
            new_pop.append(child)
        pop = torch.stack(new_pop, dim=0)

    # Final best
    final_losses = []
    for i in range(pop_size):
        loss_i = eval_loss_for_params(model, pop[i], full_train_loader, device=device)
        final_losses.append(loss_i)
    final_losses = np.array(final_losses)
    best_idx = np.argmin(final_losses)
    best_params = pop[best_idx]
    best_train_loss = final_losses[best_idx]

    test_loss = eval_loss_for_params(model, best_params, full_test_loader, device=device)
    print(f"[Pop+SGD] Final train loss: {best_train_loss:.4f}, test loss: {test_loss:.4f}")

    X_mean, X_std, y_mean, y_std = norm_params
    set_param_vector(model, best_params)
    model.eval()
    with torch.no_grad():
        x_sample, y_sample = next(iter(full_test_loader))
        preds_n = model(x_sample).flatten().numpy()
        ys_n = y_sample.numpy()
    print("\n[Pop+SGD] Sample predictions (denormalized):")
    for i in range(5):
        y_pred = preds_n[i] * y_std + y_mean
        y_true = ys_n[i] * y_std + y_mean
        print(f"  pred={y_pred:.3f}, label={y_true:.3f}, error={y_pred-y_true:.3f}")


# ============================================================
#  Main: run both and compare
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # 1) Annealed zero-grad PopES
    annealed_pop_es(
        pop_size=40,
        elite_frac=0.2,
        sigma_start=0.05,
        sigma_end=0.005,
        generations=300,
        use_nonlinear=True,
        device="cpu"
    )

    # 2) Population + local SGD (multi-optimizer)
    population_with_local_sgd(
        pop_size=50,          # start smaller; you can crank this later
        elite_frac=0.3,
        generations=30,
        local_epochs=1,       # tweak: 1–3 local epochs per generation
        local_lr=1e-3,
        sigma_mutation=0.02,
        use_nonlinear=True,
        device="cpu"
    )
