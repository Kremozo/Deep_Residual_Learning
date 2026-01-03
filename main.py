import matplotlib.pyplot as plt
from ResNetCIFAR import device, ResNetCIFAR,ResidualBlock,PlainBlock,train_and_validate
def visualize(plain_loss, resnet_loss,plain_acc,resnet_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Optimization (Training Loss)
    ax1.plot(plain_loss, label='PlainNet', color='red', linestyle='--')
    ax1.plot(resnet_loss, label='ResNet', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Generalization (Test Accuracy)
    ax2.plot(plain_acc, label='PlainNet', color='red', linestyle='--')
    ax2.plot(resnet_acc, label='ResNet', color='blue')
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()

    plt.show()
def main():
    
    num_blocks = [3, 3, 3]

    plain_net = ResNetCIFAR(PlainBlock, num_blocks).to(device)
    resnet_net = ResNetCIFAR(ResidualBlock, num_blocks).to(device)
    
    plain_loss, plain_acc = train_and_validate(plain_net, "PlainNet")
    resnet_loss, resnet_acc = train_and_validate(resnet_net, "ResNet")

    visualize(plain_loss,plain_acc,resnet_loss,resnet_acc)

if __name__ == "__main__":
    main()