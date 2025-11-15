import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models.digitclassifier import DigitClassifier
from engine.engine import train, predict
import itertools


def split_data(TEST_MNIST):
    test_size = int(0.8 * len(TEST_MNIST))
    val_size = len(TEST_MNIST) - test_size

    TEST_MNIST, VAL_MNIST = random_split(
        TEST_MNIST, [test_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    return TEST_MNIST, VAL_MNIST


def evaluate_batch_configs(model_class, TRAIN_MNIST, VAL_MNIST, device):
    batch_sizes = [64, 128, 256, 512]
    num_workers = [2, 4, 8]

    results = []
    for batch_size, num_worker in itertools.product(batch_sizes, num_workers):
        print(
            f"\n-------- Testing batch size {batch_size}, workers {num_worker} --------"
        )
        try:
            train_loader = DataLoader(
                TRAIN_MNIST, batch_size=batch_size, shuffle=True, num_workers=num_worker
            )

            val_loader = DataLoader(
                VAL_MNIST, batch_size=batch_size, shuffle=True, num_workers=num_worker
            )

            model = model_class(in_features=784, out_features=10).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            loss_fn = torch.nn.CrossEntropyLoss()

            for _ in range(5):
                model.train()
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
                model.eval()
                correct, total = 0.0, 0.0

                with torch.no_grad():
                    for img, label in val_loader:
                        img, label = img.to(device), label.to(device)
                        logits = model(img)
                        pred = logits.argmax(dim=1)
                        correct += (pred == label).sum().item()
                        total += label.size(0)
            acc = correct / total
            print(f"Validation Accuracy: {acc*100:.2f}%")
            results.append((batch_size, num_worker, acc))
        except RuntimeError as e:
            print(
                f"Skipped config (batch={batch_size}, workers={num_worker}) due to error: {e}"
            )
            continue
    return results


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: Metal (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    TRAIN_MNIST = datasets.MNIST(
        "/Users/akildickerson/Projects/DigitClassifier_v2/src/data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    TEST_MNIST = datasets.MNIST(
        "/Users/akildickerson/Projects/DigitClassifier_v2/src/data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    TEST_MNIST, VAL_MNIST = split_data(TEST_MNIST)

    results = evaluate_batch_configs(DigitClassifier, TRAIN_MNIST, VAL_MNIST, device)
    results.sort(key=lambda x: x[2], reverse=True)

    print("\n-------- batches, workers, and validation accuracies --------")
    for bs, nw, acc in results:
        print(f"batch={bs:3d} workers={nw:2d} -> validation accuracy={acc*100:.2f}%")

    batch_size, num_workers, _ = results[0]
    model = DigitClassifier(in_features=784, out_features=10).to(device)

    train_loader = DataLoader(
        TRAIN_MNIST, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        VAL_MNIST, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        TEST_MNIST, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        learning_rate=0.001,
        device=device,
    )
    predict(model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
