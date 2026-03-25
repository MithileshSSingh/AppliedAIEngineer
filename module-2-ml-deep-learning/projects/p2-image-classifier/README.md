# P2: Image Classifier with Transfer Learning

## Objective
Build an image classification model using both a custom CNN and transfer learning (pretrained ResNet). Compare approaches on a real-world dataset.

## Dataset
CIFAR-10 via torchvision (60,000 32x32 color images, 10 classes):
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training + 10,000 test images

## Architecture
```
Approach 1: Custom CNN (from scratch)
    Conv blocks → FC layers → Softmax

Approach 2: Transfer Learning (ResNet18)
    Pretrained ResNet18 → Replace FC → Fine-tune
```

## Key Skills
- Building CNNs with PyTorch
- Data augmentation (transforms)
- Transfer learning with pretrained models
- Training with GPU/MPS acceleration
- Model comparison and error analysis

## Suggested Approach
1. Load and explore CIFAR-10
2. Build a custom CNN, train from scratch
3. Load pretrained ResNet18, fine-tune on CIFAR-10
4. Compare accuracy, training time, convergence
5. Error analysis: which classes are hardest?
6. Visualize learned features (optional)

## How to Run
```bash
jupyter lab notebook.ipynb
```
