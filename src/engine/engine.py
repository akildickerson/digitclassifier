import torch
import torch.nn as nn
import torch.optim as optim
from torch.serialization import validate_cuda_device


def train(model, train_loader, val_loader, epochs, learning_rate, device):
    print(f"-------- Training --------")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0.0, 0.0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            ### FORWARD PASS ####
            logits = model(img)

            ### CALCULATE LOSS ###
            loss = loss_fn(logits, label)

            # 1. Zero the gradients
            optimizer.zero_grad()

            # 2. Compute gradients
            loss.backward()

            # 3. Update the parameters
            optimizer.step()

            running_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                pred = logits.argmax(dim=1)
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)
        val_accuracy = val_correct / val_total

        avg_loss = running_loss / len(train_loader)
        print(
            f"Epoch {epoch:02d}:  Loss={avg_loss:.4f}     Train Accuracy={(100 * (correct/total)):.2f}%    Validation Accuracy={100 *val_accuracy:.2f}%"
        )

        if avg_loss < 0.01:
            break


def predict(model, test_loader, device):
    print("\n-------- Test Accuracy --------")
    model.eval()
    correct, total = 0.0, 0.0

    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            logits = model(img)
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

    print(f"{100*(correct / total):.2f}%")
