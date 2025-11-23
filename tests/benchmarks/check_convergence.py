
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vit_pytorch.fractal_vit import NextGenerationFractalViT

def check_convergence():
    print("Starting Convergence Check for Fractal ViT...")
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple synthetic dataset (Overfitting test)
    # Task: Classify images based on the color of the top-left pixel
    # If top-left is red -> Class 0, if blue -> Class 1
    BATCH_SIZE = 8
    NUM_SAMPLES = 100
    IMAGE_SIZE = (32, 32)
    
    print("Generating synthetic dataset...")
    images = torch.randn(NUM_SAMPLES, 3, *IMAGE_SIZE)
    labels = torch.zeros(NUM_SAMPLES, dtype=torch.long)
    
    for i in range(NUM_SAMPLES):
        if i % 2 == 0:
            images[i, 0, 0:16, 0:16] = 2.0 # Red patch
            labels[i] = 0
        else:
            images[i, 2, 0:16, 0:16] = 2.0 # Blue patch
            labels[i] = 1
            
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    model = NextGenerationFractalViT(
        image_size=IMAGE_SIZE,
        num_classes=2,
        dim=64,
        depth=2,
        heads=4,
        mlp_dim=128,
        min_patch_size=(4, 4),
        max_level=3
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    losses = []
    accuracies = []
    EPOCHS = 20
    
    print(f"Training for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_imgs, batch_labels in dataloader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            
            # Handle tuple output if model returns aux info
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
        
        if accuracy > 99:
            print("Converged early!")
            break
            
    # Plotting
    try:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Accuracy', color='orange')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        output_path = Path("workspace") / "convergence_check.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path)
        print(f"Convergence plot saved to {output_path}")
    except Exception as e:
        print(f"Could not save plot: {e}")

    if accuracies[-1] > 90:
        print("\n✅ SUCCESS: Model successfully learned the synthetic task.")
    else:
        print("\n❌ FAILURE: Model failed to converge.")

if __name__ == "__main__":
    check_convergence()
