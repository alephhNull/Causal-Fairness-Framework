import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# Define MLP Model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        # Shared layers (latent representation Z)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        # Task head (income prediction)
        self.head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_z=False):
        z = self.shared(x)
        y_pred = self.head(z)
        if return_z:
            return y_pred, z  # Return both prediction and latent Z
        return y_pred


class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 8),  # Input: latent Z (dim=16)
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()       # Predict sensitive attribute S (gender)
        )

    def forward(self, z):
        return self.network(z)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# Update Adversary with GRL
class Adversary(nn.Module):
    def __init__(self, lambda_=1.0):
        super(Adversary, self).__init__()
        self.grl = GradientReversal(lambda_)
        self.network = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.grl(z)  # Reverse gradients here
        return self.network(z)


def train_nn_adversarial(X_train, y_train, s_train, feature_cols, n_epochs=500, lambda_=1.0):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])

    main_model = MLPClassifier(input_dim=X_train_scaled.shape[1])
    adversary = Adversary(lambda_=lambda_)

    criterion_main = nn.BCELoss()  # For income prediction
    criterion_adv = nn.BCELoss()  # For gender prediction

    # Separate optimizers for main model and adversary
    optimizer_main = optim.Adam(main_model.parameters(), lr=0.01)
    optimizer_adv = optim.Adam(adversary.parameters(), lr=0.01)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    s_train_tensor = torch.tensor(s_train.values, dtype=torch.float32).view(-1, 1)

    for epoch in tqdm(range(n_epochs), "Adversarial Training..."):
        # Forward pass
        y_pred, z = main_model(X_train_tensor, return_z=True)
        s_pred = adversary(z)

        # Compute losses
        loss_main = criterion_main(y_pred, y_train_tensor)
        loss_adv = criterion_adv(s_pred, s_train_tensor)

        # Total loss = main loss - λ * adversary loss
        # (Minimize main loss, maximize adversary loss)
        total_loss = loss_main - lambda_ * loss_adv

        # Update main model (freeze adversary)
        optimizer_main.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer_main.step()

        # Update adversary (freeze main model)
        optimizer_adv.zero_grad()
        s_pred_adv = adversary(z.detach())
        loss_adv_detached = criterion_adv(s_pred_adv, s_train_tensor)
        loss_adv_detached.backward()
        optimizer_adv.step()

    print("======= Training Completed ========")

    with torch.no_grad():
        _, z = main_model(X_train_tensor, return_z=True)
        s_pred = adversary(z)
        adv_accuracy = ((s_pred.round() == s_train_tensor).float().mean().item())
    print(f"Adversary accuracy: {adv_accuracy:.3f}")

    return main_model, scaler


def predict_nn(model, X_test, feature_cols, scaler):
    X_test_scaled = scaler.transform(X_test[feature_cols])
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    return model(X_test_tensor).detach().numpy().round()
