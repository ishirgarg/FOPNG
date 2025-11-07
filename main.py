import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
import numpy as np

# -----------------------------
# Task Gradient Storage
# -----------------------------
class TaskGradientBuffer:
    def __init__(self, scheme='full', rank=None):
        self.scheme = scheme
        self.rank = rank
        self.gradients = []

    def add(self, grad):
        if self.scheme == 'low_rank':
            if len(self.gradients) == 0:
                self.gradients.append(grad.clone().detach())
            else:
                G = torch.stack(self.gradients + [grad])
                U, S, V = torch.svd(G)
                self.gradients = [U[:, :self.rank] @ torch.diag(S[:self.rank])]
        else:
            self.gradients.append(grad.clone().detach())

    def get_matrix(self):
        if len(self.gradients) == 0:
            return None
        return torch.stack(self.gradients, dim=1)

# -----------------------------
# Simple MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# Dataset: Split MNIST
# -----------------------------
def get_split_mnist(digit_pair=(0,1), train=True):
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    indices = [i for i, (_, y) in enumerate(mnist) if y in digit_pair]
    return Subset(mnist, indices)

# -----------------------------
# Empirical Fisher
# -----------------------------
def compute_fisher(model, dataloader, criterion, device='mps', diagonal=False):
    if diagonal:
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.clone() ** 2
        for n in fisher:
            fisher[n] /= len(dataloader)
        return torch.cat([fisher[n].view(-1) for n in fisher])
    else:
        p = sum(p.numel() for p in model.parameters())
        fisher = torch.zeros(p, p, device=device)
        model.eval()
        n_samples = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            for i in range(data.size(0)):
                model.zero_grad()
                output = model(data[i:i+1])
                loss = criterion(output, target[i:i+1])
                loss.backward()
                grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
                fisher += torch.outer(grad, grad)
                n_samples += 1
        return fisher / n_samples

def compute_fisher_diagonal_approx(model, dataloader, criterion, device='mps'):
    return compute_fisher(model, dataloader, criterion, device, diagonal=True)

# -----------------------------
# FOPNG Optimizer
# -----------------------------
class FOPNGOptimizer:
    def __init__(self, model, lambda_reg=1e-3, fisher_type='diagonal', epsilon=1e-10):
        self.model = model
        self.lambda_reg = lambda_reg
        self.fisher_type = fisher_type
        self.epsilon = epsilon

    def step(self, gradient, F_new, F_old, G):
        device = gradient.device
        lam = self.lambda_reg

        if self.fisher_type == 'diagonal':
            F_new_inv_diag = 1.0 / (F_new + lam)
            F_old_diag = F_old.view(-1, 1)
            F_old_G = F_old_diag * G
            weighted_G = F_old_diag * (F_new_inv_diag.view(-1,1) * F_old_G)
            A = G.T @ weighted_G + lam * torch.eye(G.size(1), device=device)
            A_inv = torch.inverse(A)

            F_old_g = F_old * gradient
            G_T_F_old_g = G.T @ F_old_g
            A_inv_G_T_F_old_g = A_inv @ G_T_F_old_g
            correction = (G @ A_inv_G_T_F_old_g).view(-1) * F_old.squeeze()
            P_g = gradient - correction
            F_new_inv_P_g = P_g * F_new_inv_diag
            denom = torch.sqrt((P_g * F_new_inv_P_g).sum() + 1e-8)
            v_star = -self.epsilon * F_new_inv_P_g / (denom + 1e-8)
        else:
            F_new_inv = torch.inverse(F_new + lam * torch.eye(F_new.size(0), device=device))
            temp = F_old @ F_new_inv @ F_old @ G
            A = G.T @ temp + lam * torch.eye(G.size(1), device=device)
            A_inv = torch.inverse(A)
            P = torch.eye(gradient.size(0), device=device) - F_old @ G @ A_inv @ G.T @ F_old
            P_g = P @ gradient
            F_new_inv_P_g = F_new_inv @ P_g
            denom = torch.sqrt(P_g @ F_new_inv_P_g + 1e-8)
            v_star = -self.epsilon * F_new_inv_P_g / denom
        return v_star

# -----------------------------
# Adam Training (with stats)
# -----------------------------
def train(model, dataloader, optimizer, criterion, n_epochs, device='mps'):
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        grad_norms, upd_norms = [], []

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            grad_norms.append(grad.norm().item())

            old_params = [p.clone() for p in model.parameters()]
            optimizer.step()

            with torch.no_grad():
                upd = torch.cat([(p - o).view(-1) for p, o in zip(model.parameters(), old_params)])
                upd_norms.append(upd.norm().item())

            epoch_loss += loss.item()

        mean_g, std_g = np.mean(grad_norms), np.std(grad_norms)
        mean_u, std_u = np.mean(upd_norms), np.std(upd_norms)
        ratio = mean_u / (mean_g + 1e-8)
        print(f"[Adam ] Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss/len(dataloader):.4f} "
              f"| ||∇θ|| mean={mean_g:.3e}±{std_g:.3e} | ||Δθ|| mean={mean_u:.3e}±{std_u:.3e} | ratio={ratio:.3e}")

# -----------------------------
# FOPNG Training (with stats)
# -----------------------------
def train_with_fopng(model, dataloader, fopng_optimizer, criterion, F_new, F_old, G, n_epochs, device='mps'):
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        grad_norms, upd_norms = [], []

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()

            grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            grad_norms.append(grad.norm().item())

            update = fopng_optimizer.step(grad, F_new, F_old, G)
            upd_norms.append(update.norm().item())

            idx = 0
            with torch.no_grad():
                for p in model.parameters():
                    n = p.numel()
                    p.add_(update[idx:idx+n].view_as(p))
                    idx += n

            epoch_loss += loss.item()

        mean_g, std_g = np.mean(grad_norms), np.std(grad_norms)
        mean_u, std_u = np.mean(upd_norms), np.std(upd_norms)
        ratio = mean_u / (mean_g + 1e-8)
        print(f"[FOPNG] Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss/len(dataloader):.4f} "
              f"| ||∇θ|| mean={mean_g:.3e}±{std_g:.3e} | ||Δθ|| mean={mean_u:.3e}±{std_u:.3e} | ratio={ratio:.3e}")

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, dataloader, device='mps'):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

# -----------------------------
# Continual Learning Routine
# -----------------------------
def run_experiment(use_fopng=True, fisher_type='diagonal', lr=1e-3, n_epochs=5):
    device = 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Using FOPNG: {use_fopng}, Fisher type: {fisher_type}")

    task_pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    task_gradients = TaskGradientBuffer(scheme='full', rank=200000)
    all_task_accuracies = []

    if use_fopng:
        fopng_optimizer = FOPNGOptimizer(model, lambda_reg=1e-3, fisher_type=fisher_type, epsilon=1e-4)

    for task_idx, digit_pair in enumerate(task_pairs):
        print(f"\n=== Task {task_idx+1}: digits {digit_pair} ===")
        train_loader = DataLoader(get_split_mnist(digit_pair, train=True), batch_size=64, shuffle=True)
        test_loader = DataLoader(get_split_mnist(digit_pair, train=False), batch_size=64, shuffle=False)

        if task_idx == 0:
            print("Training with Adam (first task)...")
            opt = Adam(model.parameters(), lr=lr)
            train(model, train_loader, opt, criterion, n_epochs, device=device)
        else:
            F_new = compute_fisher_diagonal_approx(model, train_loader, criterion, device)
            old_pairs = task_pairs[:task_idx]
            old_data = []
            for op in old_pairs:
                old_data.extend(list(get_split_mnist(op, train=True)))
            old_loader = DataLoader(old_data, batch_size=64, shuffle=True)
            F_old = compute_fisher_diagonal_approx(model, old_loader, criterion, device)
            G = task_gradients.get_matrix()

            if use_fopng:
                print("Fine-tuning with FOPNG...")
                train_with_fopng(model, train_loader, fopng_optimizer, criterion, F_new, F_old, G, n_epochs, device=device)
            else:
                print("Fine-tuning with Adam...")
                opt = Adam(model.parameters(), lr=lr)
                train(model, train_loader, opt, criterion, n_epochs, device=device)

        # Store gradient
        model.eval(); model.zero_grad()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            loss = criterion(model(data), target)
            loss.backward()
            break
        flat_grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        task_gradients.add(flat_grad)

        # Evaluation
        accs = []
        for eval_idx, pair in enumerate(task_pairs[:task_idx+1]):
            loader = DataLoader(get_split_mnist(pair, train=False), batch_size=64, shuffle=False)
            acc = evaluate(model, loader, device)
            accs.append(acc)
            print(f"  Task {eval_idx+1} ({pair}): {acc*100:.2f}%")
        all_task_accuracies.append(accs)
        print(f"Mean accuracy after Task {task_idx+1}: {np.mean(accs)*100:.2f}%")

    return all_task_accuracies

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("# Running FOPNG")
    run_experiment(use_fopng=True, fisher_type='diagonal', lr=1e-3, n_epochs=15)

    print("\n# Running Adam baseline")
    run_experiment(use_fopng=False, fisher_type='diagonal', lr=1e-3, n_epochs=5)