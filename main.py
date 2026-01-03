import matplotlib.pyplot as plt
import torch
from ResNetCIFAR import device, ResNetCIFAR,ResidualBlock,PlainBlock,train_and_validate
import os

import os

def get_or_train_model(model, model_name, epochs=15):
    filename = f"{model_name}_checkpoint.pth"
    
    if os.path.exists(filename):
        print(f"\n[CACHE] Found saved model for {model_name}. Loading...")
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        return checkpoint['history'] # Returns the dictionary we saved
    
    else:
        print(f"\n[CACHE] No cache found for {model_name}. Training...")
        history = train_and_validate(model, model_name, epochs)
        
        print(f"[CACHE] Saving {model_name} to disk...")
        torch.save({
            'model_state': model.state_dict(),
            'history': history
        }, filename)
        
        return history
def visualize_comparison(h_p20, h_r20, h_p56, h_r56):
    plt.figure(figsize=(12, 8))
    
    # Helper to convert Acc to Error
    def get_err(hist_dict, key):
        return [100 - x for x in hist_dict[key]]

    # --- PLOT SETTINGS ---
    # Colors
    c_p20 = 'salmon'      # Light Red
    c_p56 = 'darkred'     # Dark Red
    c_r20 = 'skyblue'     # Light Blue
    c_r56 = 'darkblue'    # Dark Blue
    
    # Styles
    style_train = '--'    # Dashed
    style_test = '-'      # Solid
    width_train = 1.5
    width_test = 3.0      # Bold

    # --- PLOTTING ---
    
    # 1. Plain-20
    plt.plot(get_err(h_p20, 'train_acc'), color=c_p20, linestyle=style_train, linewidth=width_train, label='Plain-20 Train Error')
    plt.plot(get_err(h_p20, 'test_acc'),  color=c_p20, linestyle=style_test,  linewidth=width_test,  label='Plain-20 Test Error')

    # 2. ResNet-20
    plt.plot(get_err(h_r20, 'train_acc'), color=c_r20, linestyle=style_train, linewidth=width_train, label='ResNet-20 Train Error')
    plt.plot(get_err(h_r20, 'test_acc'),  color=c_r20, linestyle=style_test,  linewidth=width_test,  label='ResNet-20 Test Error')

    # 3. Plain-56
    plt.plot(get_err(h_p56, 'train_acc'), color=c_p56, linestyle=style_train, linewidth=width_train, label='Plain-56 Train Error')
    plt.plot(get_err(h_p56, 'test_acc'),  color=c_p56, linestyle=style_test,  linewidth=width_test,  label='Plain-56 Test Error')

    # 4. ResNet-56
    plt.plot(get_err(h_r56, 'train_acc'), color=c_r56, linestyle=style_train, linewidth=width_train, label='ResNet-56 Train Error')
    plt.plot(get_err(h_r56, 'test_acc'),  color=c_r56, linestyle=style_test,  linewidth=width_test,  label='ResNet-56 Test Error')

    plt.title('CIFAR-10 Error Rates: Plain vs ResNet')
    plt.xlabel('Epochs')
    plt.ylabel('Error (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def main():
    shallow_blocks = [3, 3, 3] # 20 layers
    deep_blocks = [9, 9, 9]    # 56 layers

    print("Initializing models...")
    plain_20 = ResNetCIFAR(PlainBlock, shallow_blocks).to(device)
    resnet_20 = ResNetCIFAR(ResidualBlock, shallow_blocks).to(device)
    plain_56 = ResNetCIFAR(PlainBlock, deep_blocks).to(device)
    resnet_56 = ResNetCIFAR(ResidualBlock, deep_blocks).to(device)
    
    h_p20 = get_or_train_model(plain_20, "PlainNet_20")
    h_r20 = get_or_train_model(resnet_20, "ResNet_20")
    h_p56 = get_or_train_model(plain_56, "PlainNet_56")
    h_r56 = get_or_train_model(resnet_56, "ResNet_56")

    # 4. Visualize
    visualize_comparison(h_p20, h_r20, h_p56, h_r56)

if __name__ == "__main__":
    main()