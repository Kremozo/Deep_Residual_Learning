import matplotlib.pyplot as plt
import torch
from ResNetCIFAR import device, ResNetCIFAR,ResidualBlock,PlainBlock,train_and_validate
import os

def get_or_train_model(model, model_name, epochs=15):
    filename = f"{model_name}_checkpoint.pth"
    
    # 1. Check if the file already exists
    if os.path.exists(filename):
        print(f"\n[CACHE] Found saved model for {model_name}. Loading...")
        
        # Load the dictionary from the file
        checkpoint = torch.load(filename, map_location=device)
        
        # Restore the model weights
        model.load_state_dict(checkpoint['model_state'])
        
        # Return the saved history lists so we can plot them
        return checkpoint['train_loss'], checkpoint['test_acc']
    
    else:
        # 2. If no file, run the training normally
        print(f"\n[CACHE] No cache found for {model_name}. Training from scratch...")
        train_loss, test_acc = train_and_validate(model, model_name, epochs)
        
        # 3. Save everything into one file
        print(f"[CACHE] Saving {model_name} to disk...")
        torch.save({
            'model_state': model.state_dict(),
            'train_loss': train_loss,
            'test_acc': test_acc
        }, filename)
        
        return train_loss, test_acc
def visualize_comparison(p20, r20, p56, r56):
    plt.figure(figsize=(10, 6))
    
    # Shallow Networks (Dashed Lines)
    plt.plot(p20, label='Plain-20', color='red', linestyle='--', alpha=0.6)
    plt.plot(r20, label='ResNet-20', color='blue', linestyle='--', alpha=0.6)
    
    # Deep Networks (Solid Lines)
    plt.plot(p56, label='Plain-56', color='darkred', linewidth=2)
    plt.plot(r56, label='ResNet-56', color='darkblue', linewidth=2)
    
    plt.title('The Degradation Problem: 20 vs 56 Layers')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
def main():
    
    # --- Experiment 1: Shallow Networks (20 Layers) ---
    print("\n--- Running Experiment 1: Shallow (20 Layers) ---")
    shallow_blocks = [3, 3, 3]

    plain_20 = ResNetCIFAR(PlainBlock, shallow_blocks).to(device)
    resnet_20 = ResNetCIFAR(ResidualBlock, shallow_blocks).to(device)
    
    # Train or Load
    p20_loss, p20_acc = get_or_train_model(plain_20, "PlainNet_20")
    r20_loss, r20_acc = get_or_train_model(resnet_20, "ResNet_20")

    # --- Experiment 2: Deep Networks (56 Layers) ---
    print("\n--- Running Experiment 2: Deep (56 Layers) ---")
    deep_blocks = [9, 9, 9] 

    plain_56 = ResNetCIFAR(PlainBlock, deep_blocks).to(device)
    resnet_56 = ResNetCIFAR(ResidualBlock, deep_blocks).to(device)
    
    # Train or Load
    p56_loss, p56_acc = get_or_train_model(plain_56, "PlainNet_56")
    r56_loss, r56_acc = get_or_train_model(resnet_56, "ResNet_56")

    # --- Visualization ---
    visualize_comparison(p20_loss, r20_loss, p56_loss, r56_loss)

if __name__ == "__main__":
    main()