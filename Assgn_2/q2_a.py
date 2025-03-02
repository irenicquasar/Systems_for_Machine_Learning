import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
import math

# 1) RBF Layer
class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # x: (B, in_features)
        # Compute squared Euclidean distances between x and each center.
        x_expanded = x.unsqueeze(1).expand(-1, self.out_features, -1)
        centers_expanded = self.centers.unsqueeze(0).expand(x.size(0), -1, -1)
        distances = torch.sum((x_expanded - centers_expanded) ** 2, dim=2)
        return distances

# 2) MAPLoss (Equation 9)
class MAPLoss(nn.Module):
    def __init__(self, j=1.0):
        super().__init__()
        self.j = j
    
    def forward(self, outputs, targets):
        """
        outputs: (B, num_classes) - RBF distances
        targets: (B,) ground-truth class indices
        """
        batch_size = outputs.size(0)
        # Get the distance for the correct class for each sample
        correct_class_distances = outputs[torch.arange(batch_size), targets]

        # Compute e^(-distance) for all classes
        exp_neg_distances = torch.exp(-outputs)
        # Convert self.j to a tensor on the same device as outputs
        j_tensor = torch.tensor(self.j, dtype=outputs.dtype, device=outputs.device)
        exp_neg_j = torch.exp(-j_tensor)

        # Sum over classes and add the constant term
        sum_exp_terms = exp_neg_distances.sum(dim=1) + exp_neg_j
        log_sum_exp_term = torch.log(sum_exp_terms)

        # Final MAP loss: mean over batch
        loss = (correct_class_distances + log_sum_exp_term).mean()
        return loss

# 3) LeNet-5 Architecture
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers:
        # C1: Input 1x32x32 -> Conv -> 6x28x28 -> Pool -> 6x14x14
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 6x14x14 -> Conv -> 16x10x10 -> Pool -> 16x5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers:
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.rbf = RBFLayer(84, 10)

    def forward(self, x):
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.rbf(x)
        return x

# 4) Training/Evaluation Utilities
def train_model(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience, device):
    """
    Train the model using train_loader. Use val_loader for early stopping.
    Stop training if validation accuracy does not improve for 'patience' epochs.
    """
    model.to(device)

    train_losses = []
    train_accs   = []
    val_accs     = []

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # For RBF, predicted class = argmin(distance)
            _, predicted = torch.min(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (len(pbar)),
                'acc':  100. * correct / total
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc  = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Evaluate on validation set for early stopping
        val_acc = evaluate_model(model, val_loader, device)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{max_epochs}] "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Check early stopping criteria
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_lenet5.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nValidation accuracy did not improve for {patience} consecutive epochs. Stopping early.")
            break
    
    return train_losses, train_accs, val_accs

def evaluate_model(model, data_loader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # For RBF, predicted = argmin(distance)
            _, predicted = torch.min(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total

def plot_training_history(train_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    
    # Loss vs. Epochs
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs epochs for LeNet-5')
    plt.legend()
    
    # Accuracy vs. Epochs
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'o-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'o-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('accuracy vs epochs for LeNet-5')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 5) Main Script
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # MNIST with images resized to 32x32 (LeNet-5 expects 32x32 input)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    # Download full MNIST training set
    full_train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Split full training set into training (90%) and validation (10%)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # Initialize model, criterion, optimizer
    model = LeNet5()
    criterion = MAPLoss(j=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Print model summary
    summary(model, input_size=(1, 32, 32))
    
    # Training parameters
    max_epochs = 50
    patience = 5  # Stop if no improvement on validation set for 5 consecutive epochs
    
    train_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, max_epochs, patience, device
    )
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_accs)
    
    # Final evaluation on the test set using the best model saved via early stopping
    model.load_state_dict(torch.load("best_lenet5.pth"))
    final_test_acc = evaluate_model(model, test_loader, device)
    print(f"\nBest model test accuracy: {final_test_acc:.2f}%")

if __name__ == "__main__":
    main()
