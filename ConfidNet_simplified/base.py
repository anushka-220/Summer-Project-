import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt # For visualizing images

# --- 1. Improved Model Definition ---
class ImprovedMNISTClassifier(nn.Module):
    def __init__(self, num_classes, input_channels=1): # num_classes is now dynamic
        super().__init__()
        self.num_classes = num_classes # Store the actual number of classes for this model
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1), # Output: 32x28x28 (maintains size with padding)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: 64x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64x14x14
            nn.Dropout(0.25),

            # Second convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Output: 128x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128x7x7
            nn.Dropout(0.25)
        )
        # Calculate input features for the linear layer
        # After two max-pooling layers, a 28x28 image becomes 7x7.
        # With 128 output channels from the last conv layer: 128 * 7 * 7
        self.classifier = nn.Sequential(
            nn.Flatten(), # Flatten the output of the convolutional layers
            nn.Linear(128 * 7 * 7, 256), # Fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes) # Output layer for classification - NOW USES DYNAMIC num_classes
        )

    def forward(self, x):
        """
        Performs a forward pass through the feature extractor and classifier.
        """
        x = self.features(x)
        logits = self.classifier(x)
        return logits

# --- 2. Training Logic ---
def train_classifier_epoch(model, device, train_loader, optimizer, epoch, log_interval):
    """
    Trains the base classifier for one epoch.
    """
    model.train() # Set model to training mode
    criterion = nn.CrossEntropyLoss() # Standard loss for classification
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # Clear gradients from previous step
        logits = model(data) # Forward pass
        loss = criterion(logits, target) # Calculate loss
        loss.backward() # Backpropagation
        optimizer.step() # Update model parameters

        if batch_idx % log_interval == 0:
            print(f"Classifier Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

# --- 3. Evaluation Logic ---
def evaluate_model(model, device, test_loader, ood_exclude_digits=None, train_class_names=None, test_class_names=None):
    """
    Evaluates the base classifier's accuracy on the test set.
    Reports overall accuracy and separate accuracy for in-distribution and OOD samples.
    """
    model.eval() # Set model to evaluation mode
    correct_cls = 0
    total = 0
    
    correct_id = 0
    total_id = 0
    correct_ood = 0
    total_ood = 0

    print("\n--- Evaluation ---")
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            logits = model(data) # Forward pass
            
            # For evaluation, we need to map the predicted class index (0-7 for training classes)
            # back to the full 0-9 digit space for comparison with the test_dataset's true labels.
            # This is where it gets tricky: The model outputs 8 logits, corresponding to 2-9.
            # We need to map these predicted indices (0-7) back to the actual digit values (2-9).
            # The test_dataset's target labels are 0-9.

            # Get the predicted class index from the model's 8 outputs
            _, predicted_train_class_idx = torch.max(logits.data, 1)

            # Map the predicted train_class_idx (0-7) to the actual digit value (2-9)
            # This assumes train_class_names is like ['2', '3', ..., '9']
            # If predicted_train_class_idx is 0, it means the model predicted '2'
            # If predicted_train_class_idx is 1, it means the model predicted '3'
            # etc.
            predicted_actual_digits = torch.tensor([int(train_class_names[idx.item()]) for idx in predicted_train_class_idx], device=device)


            total += target.size(0)
            # Now compare the mapped predicted digit with the true target digit from the test set
            correct_cls += (predicted_actual_digits == target).sum().item()


            # Separate evaluation for In-Distribution (ID) and Out-of-Distribution (OOD)
            for i in range(target.size(0)):
                sample_true_digit_label = target[i].item() # This is the actual digit (0-9)
                is_ood = ood_exclude_digits and sample_true_digit_label in ood_exclude_digits

                if is_ood:
                    total_ood += 1
                    # For OOD samples, the model is expected to be wrong, as it never saw these classes.
                    # We are checking if it *happens* to predict the correct OOD label (which is unlikely
                    # and not the goal of OOD detection, but included for completeness of accuracy calc).
                    if predicted_actual_digits[i] == sample_true_digit_label:
                        correct_ood += 1
                else: # In-Distribution (digits 2-9)
                    total_id += 1
                    if predicted_actual_digits[i] == sample_true_digit_label:
                        correct_id += 1

    overall_accuracy = 100 * correct_cls / total
    print(f'\nOverall Accuracy on {total} test images: {overall_accuracy:.2f}%')

    if ood_exclude_digits:
        print("\n--- Detailed OOD Evaluation ---")
        if total_id > 0:
            accuracy_id = 100 * correct_id / total_id
            print(f"In-Distribution Samples (digits not in {ood_exclude_digits}) - Accuracy: {accuracy_id:.2f}% ({correct_id}/{total_id})")
        else:
            print("No In-Distribution samples found in test set based on ood_exclude_digits.")
        
        if total_ood > 0:
            accuracy_ood = 100 * correct_ood / total_ood
            print(f"Out-of-Distribution Samples (digits {ood_exclude_digits}) - Accuracy: {accuracy_ood:.2f}% ({correct_ood}/{total_ood})")
        else:
            print("No Out-of-Distribution samples found in test set based on ood_exclude_digits.")
    
    return overall_accuracy

# --- 4. Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Fixed Improved Base Classifier Training and Testing")
    parser.add_argument('--data-path', type=str, default='./processed_mnist',
                        help='path to the root dataset directory (containing train/test subfolders)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N', # Increased default epochs
                        help='number of epochs to train the classifier (default: 15)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for classifier (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum for classifier (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    # Removed --num-classes from args as it's now dynamically determined for the model
    parser.add_argument('--input-channels', type=int, default=1,
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--ood-exclude-digits', type=int, nargs='+', default=[0, 1], # Default to 0 and 1
                        help='List of digit labels to exclude from training to simulate OOD (e.g., 0 1).')
    parser.add_argument('--inspect-data', action='store_true', default=False,
                        help='Enable visual inspection of a few training batch images.')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Normalization statistics for MNIST (grayscale)
    dataset_mean = (0.1307,)
    dataset_std = (0.3081,)
    if args.input_channels == 3:
        dataset_mean = (0.5, 0.5, 0.5)
        dataset_std = (0.5, 0.5, 0.5)
        print("WARNING: Using placeholder 3-channel mean/std. Calculate for your dataset.")

    transform_list = []
    if args.input_channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    transform_list.extend([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])
    transform = transforms.Compose(transform_list)

    train_data_path = os.path.join(args.data_path, 'train')
    test_data_path = os.path.join(args.data_path, 'test')

    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError(
            f"Train path ({train_data_path}) or Test path ({test_data_path}) not found. "
            f"Please ensure your data_path ('{args.data_path}') contains 'train' and 'test' subdirectories."
        )

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
    
    print(f"Training with {len(train_dataset.classes)} classes: {train_dataset.classes}")
    print(f"Testing with {len(test_dataset.classes)} classes: {test_dataset.classes}")

    train_loader_kwargs = {'batch_size': args.batch_size}
    test_loader_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_loader_kwargs.update(cuda_kwargs)
        test_loader_kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': False})
    else:
        train_loader_kwargs.update({'shuffle': True})
        test_loader_kwargs.update({'shuffle': False})

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    test_loader = DataLoader(test_dataset, **test_loader_kwargs)

    # --- Data Inspection ---
    if args.inspect_data:
        print("\n--- Inspecting a batch of training data ---")
        data_iter = iter(train_loader)
        images, labels = next(data_iter)

        print(f"Batch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")
        print(f"Min pixel value in batch: {images.min().item():.4f}")
        print(f"Max pixel value in batch: {images.max().item():.4f}")
        print(f"Mean pixel value in batch: {images.mean().item():.4f}")
        print(f"Std pixel value in batch: {images.std().item():.4f}")

        fig = plt.figure(figsize=(10, 4))
        for i in range(min(10, images.shape[0])):
            ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
            img = images[i].cpu().numpy().squeeze()
            img = img * dataset_std[0] + dataset_mean[0] # Unnormalize
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap='gray')
            # Map internal ImageFolder label (0-7) back to actual digit (2-9) for display
            true_digit_label = train_dataset.classes[labels[i].item()]
            ax.set_title(f"True: {true_digit_label}")
        plt.suptitle("Sample Training Images (Unnormalized)")
        plt.show()
        print("Please close the image window to continue training.")

    # Instantiate the model with the CORRECT number of output classes
    # This is the key fix for the "Incorrect Number of Classes" problem
    model = ImprovedMNISTClassifier(num_classes=len(train_dataset.classes), input_channels=args.input_channels).to(device)

    print("\n--- Training Base Classifier ---")
    optimizer_cls = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train_classifier_epoch(model, device, train_loader, optimizer_cls, epoch, args.log_interval)
        # Evaluate after each epoch to see progress
        # Pass train_dataset.classes to evaluate_model for correct label mapping in output
        evaluate_model(model, device, test_loader, ood_exclude_digits=args.ood_exclude_digits,
                       train_class_names=train_dataset.classes, test_class_names=test_dataset.classes)

    print("\n--- Final Evaluation of Base Classifier ---")
    final_accuracy = evaluate_model(model, device, test_loader, ood_exclude_digits=args.ood_exclude_digits,
                                    train_class_names=train_dataset.classes, test_class_names=test_dataset.classes)
    print(f"Final overall accuracy of improved base classifier: {final_accuracy:.2f}%")

if __name__ == '__main__':
    main()
