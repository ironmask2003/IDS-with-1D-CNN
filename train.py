import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from network import Dataset, CNN
from tqdm import tqdm

# Parametri
csv_path = './Dataset/UNSW_NB15_training-set.csv'
batch_size = 64
epochs = 10
lr = 1e-3
num_classes = 2

# Dataset e DataLoader
train_dataset = Dataset(csv_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Modello
model = CNN(num_classes)
model.train()

# Loss e ottimizzatore
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Per il plot
loss_history = []
acc_history = []

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit='batch')
    for inputs, labels in loop:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        bs = inputs.size(0)
        running_loss += loss.item() * bs
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += bs
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        loop.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc:.4f}")
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    loss_history.append(epoch_loss)
    acc_history.append(epoch_acc)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

# Plot finale
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.subplot(1,2,2)
plt.plot(acc_history, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()
plt.tight_layout()
plt.show()
