import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_from_disk
from sklearn.metrics import classification_report

# Setting the path and loading datasets
path = os.getcwd()
print(f"Current Working Directory: {path}")

train_data = load_from_disk('../../data_loader/data/full_train')
test_data = load_from_disk('../../data_loader/data/full_test')
print(train_data, test_data)

# Define TextCNN model
class TextCNN(nn.Module):
    def __init__(self, num_classes):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(100, 128, k) for k in [3, 4, 5, 6, 7]])
        self.pools = nn.ModuleList([nn.MaxPool1d(5) for _ in range(2)] + [nn.MaxPool1d(35)])
        self.fc_layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(3)])
        self.dropout = nn.Dropout(0.25)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=2)
        for pool in self.pools:
            x = pool(x)
        x = torch.flatten(x, start_dim=1)
        for fc in self.fc_layers:
            x = self.dropout(fc(x).relu())
        x = self.output(x)
        return x

# Preprocessing function to pad and stack data
def pad_and_stack(data, max_length=1000):
    padded_data = []
    for d in data:
        padded_vector = d['vectorized_data']
        pad_size = max_length - len(padded_vector)
        padded_vector += [torch.zeros(100)] * pad_size
        padded_data.append(torch.tensor(padded_vector))
    return torch.stack(padded_data)

# Preprocessing datasets
train_data = pad_and_stack(train_data)
train_labels = torch.tensor([d['label_level_1'] for d in train_data])
test_data = pad_and_stack(test_data)
test_labels = torch.tensor([d['label_level_1'] for d in test_data])

# Create datasets and dataloaders
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize model, loss, and optimizer
num_classes = torch.unique(train_labels).size(0)
model = TextCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Training and evaluation functions
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')

# Run training and evaluation
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
evaluate_model(model, test_loader)

# Save and load model
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Prediction example
with torch.no_grad():
    inputs = torch.tensor(test_data[0]["vectorized_data"]).float()
    outputs = model(inputs.unsqueeze(0))
    _, predicted = torch.max(outputs.data, 1)
    print(f"Predicted class: {predicted}")

# Classification report
def classification_report(predicted, true):
    print(classification_report(true, predicted))

classification_report(predicted, test_labels)
