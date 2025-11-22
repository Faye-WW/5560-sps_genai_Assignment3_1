import torch
from helper_lib.model import get_model
from helper_lib.data_loader import get_cifar10_loaders
from helper_lib.trainer import train_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader = get_cifar10_loaders(batch_size=128, img_size=64)
    model = get_model("CNN", num_classes=10)

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=3,                 
        save_path="models/cnn.pt"
    )
    print("Training summary:", result)

if __name__ == "__main__":
    main()
