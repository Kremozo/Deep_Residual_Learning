# Deep Residual Learning: Reproducing the Degradation Problem
This project is a PyTorch implementation of the concepts introduced in the paper ["Deep Residual Learning for Image Recognition" (He et al., 2015)](https://arxiv.org/abs/1512.03385).

I implemented both **Plain Networks** and **Residual Networks (ResNets)** of varying depths (20 layers vs. 56 layers) on the CIFAR-10 dataset to test two hypotheses:
1.  **Plain Nets:** Deeper networks (56 layers) should perform *worse* than shallow ones (20 layers) due to optimization difficulties.
2.  **ResNets:** Residual connections should resolve this, allowing the 56-layer model to outperform the 20-layer model.

##Results
![Experiment Results](error_rates.png)
References
    K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2015.
