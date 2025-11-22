import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from gan.models import Generator, Discriminator, LATENT_DIM

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    # macOS/MPS: 这里很关键 → num_workers=0, pin_memory=False
    loader = DataLoader(ds, batch_size=128, shuffle=True,
                        num_workers=0, pin_memory=False)

    G, D = Generator().to(DEVICE), Discriminator().to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    os.makedirs("artifacts", exist_ok=True)
    fixed_z = torch.randn(64, LATENT_DIM, device=DEVICE)

    EPOCHS = 6
    for epoch in range(EPOCHS):
        for x, _ in loader:
            x = x.to(DEVICE)

            # ---- Train D ----
            z = torch.randn(x.size(0), LATENT_DIM, device=DEVICE)
            with torch.no_grad():
                fake = G(z)
            lossD = bce(D(x), torch.ones(x.size(0), device=DEVICE)) + \
                    bce(D(fake), torch.zeros(x.size(0), device=DEVICE))
            optD.zero_grad(set_to_none=True); lossD.backward(); optD.step()

            # ---- Train G ----
            z = torch.randn(x.size(0), LATENT_DIM, device=DEVICE)
            fake = G(z)
            lossG = bce(D(fake), torch.ones(x.size(0), device=DEVICE))
            optG.zero_grad(set_to_none=True); lossG.backward(); optG.step()

        with torch.no_grad():
            grid = (G(fixed_z).to("cpu") + 1) / 2
        utils.save_image(grid, f"artifacts/samples_epoch_{epoch:03d}.png", nrow=8)
        torch.save(G.state_dict(), "artifacts/generator.pt")
        print(f"[epoch {epoch}] lossD={lossD.item():.4f} lossG={lossG.item():.4f}")

    print("Saved: artifacts/generator.pt")

if __name__ == "__main__":
    main()

