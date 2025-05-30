from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn

def train_model(model, train_loader, device, epochs=10, lr=1e-3, save_path="vdsr.pth"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for lr_batch, hr_batch in progress:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()
            output = model(lr_batch)
            loss = criterion(output, hr_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"Loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dataset = PatchSRDataset("path_to_LR", "path_to_HR", patch_size=48, scale=4, augment=True)
#     loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

#     model = VDSR()
#     train_model(model, loader, device, epochs=20, lr=1e-3, save_path="vdsr_final.pth")
