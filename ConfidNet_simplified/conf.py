import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np # For calculating averages

# --- 1. Model Definition ---
class CustomConfidNet(nn.Module):
    # num_classes will now be passed dynamically based on train_dataset.classes
    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        self.num_classes = num_classes # Store the actual number of classes for the classifier head
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout_feat_ext = nn.Dropout(0.25)
        # Calculate the input dimension for the first fully connected layer
        # For MNIST (28x28) after two 3x3 convs and 2x2 maxpool:
        # (28 - 3 + 1) = 26 -> (26 - 3 + 1) = 24 -> (24 / 2) = 12
        self.fc_feat_input_dim = 64 * 12 * 12
        self.fc_feat = nn.Linear(self.fc_feat_input_dim, 128)
        self.dropout_fc_feat_ext = nn.Dropout(0.5)
        self.classifier_fc = nn.Linear(128, self.num_classes) # Use dynamic num_classes here
        self.confid_fc1 = nn.Linear(128, 64)
        self.confid_fc2 = nn.Linear(64, 32)
        self.confid_fc3 = nn.Linear(32, 1)
        self._feature_extractor_modules = nn.ModuleList([self.conv1, self.conv2, self.fc_feat])
        self._classifier_head_modules = nn.ModuleList([self.classifier_fc])
        self._confidence_head_modules = nn.ModuleList([self.confid_fc1, self.confid_fc2, self.confid_fc3])

    def get_features(self, x):
        """
        Extracts features from the input image using convolutional and fully connected layers.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout_feat_ext(x)
        x = x.view(x.size(0), -1) # Flatten the tensor
        if x.shape[1] != self.fc_feat_input_dim:
            # This check helps debug issues with input dimensions if image size changes
            raise ValueError(
                f"Unexpected feature shape after conv layers. Expected {self.fc_feat_input_dim}, got {x.shape[1]}."
            )
        x = F.relu(self.fc_feat(x))
        x = self.dropout_fc_feat_ext(x)
        return x

    def forward(self, x):
        """
        Performs a forward pass through the feature extractor, classifier head, and confidence head.
        """
        features = self.get_features(x)
        classification_logits = self.classifier_fc(features)
        confid_hidden = F.relu(self.confid_fc1(features))
        confid_hidden = F.relu(self.confid_fc2(confid_hidden))
        predicted_confidence = torch.sigmoid(self.confid_fc3(confid_hidden))
        return classification_logits, predicted_confidence

    def _set_module_list_grad(self, module_list, requires_grad_bool):
        """
        Helper function to set requires_grad for parameters within a list of module lists.
        """
        for modules in module_list:
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = requires_grad_bool

    def configure_train_phase(self, phase="classification_train"):
        """
        Configures the model for different training phases by setting `train()`/`eval()`
        and `requires_grad` for different module groups.
        """
        if phase == "classification_train":
            self.train() # Set model to training mode
            # Enable gradients for feature extractor and classifier head
            self._set_module_list_grad([self._feature_extractor_modules, self._classifier_head_modules], True)
            # Disable gradients for confidence head
            self._set_module_list_grad([self._confidence_head_modules], False)
        elif phase == "confidence_train":
            # Set feature extractor and classifier head to eval mode (freeze their weights)
            for module in self._feature_extractor_modules: module.eval()
            self.dropout_feat_ext.eval(); self.dropout_fc_feat_ext.eval() # Dropouts in eval mode
            for module in self._classifier_head_modules: module.eval()
            # Set confidence head to train mode (enable its gradients)
            for module in self._confidence_head_modules: module.train()
            self._set_module_list_grad([self._feature_extractor_modules, self._classifier_head_modules], False)
            self._set_module_list_grad([self._confidence_head_modules], True)
        elif phase == "confidence_finetune_convnet":
            self.train() # Set entire model to training mode
            # Enable gradients for feature extractor and confidence head
            self._set_module_list_grad([self._feature_extractor_modules, self._confidence_head_modules], True)
            # Disable gradients for classifier head
            self._set_module_list_grad([self._classifier_head_modules], False)
            # Keep dropouts in eval mode as per original ConfidNet paper during fine-tuning
            self.dropout_feat_ext.eval(); self.dropout_fc_feat_ext.eval()
        elif phase == "eval":
            self.eval() # Set entire model to evaluation mode
            # Disable gradients for all parameters
            self._set_module_list_grad(
                [self._feature_extractor_modules, self._classifier_head_modules, self._confidence_head_modules], False)
        elif phase == "mc_dropout_eval":
            self.eval() # Set model to eval mode, but enable dropouts for MC sampling
            self.dropout_feat_ext.train(); self.dropout_fc_feat_ext.train()
        else:
            raise ValueError(f"Unknown training phase: {phase}")

# --- 2. Training Logic ---
def train_classifier_epoch(model, device, train_loader, optimizer, epoch, log_interval):
    """
    Trains the base classifier (feature extractor + classifier head) for one epoch.
    """
    model.configure_train_phase("classification_train")
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, _ = model(data) # Only use logits for classification training
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Classifier Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

def train_confidnet_epoch(model, device, train_loader, optimizer, epoch, log_interval, fine_tune_convnet=False):
    """
    Trains the confidence head (and optionally fine-tunes the ConvNet) for one epoch.
    The target for confidence is True Class Probability (TCP).
    """
    if fine_tune_convnet:
        model.configure_train_phase("confidence_finetune_convnet")
    else:
        model.configure_train_phase("confidence_train")
    criterion_confid = nn.MSELoss() # MSE loss for confidence prediction
    for batch_idx, (data, target_labels_internal) in enumerate(train_loader):
        data, target_labels_internal = data.to(device), target_labels_internal.to(device)
        optimizer.zero_grad()
        classification_logits, predicted_confidence = model(data)
        
        with torch.no_grad():
            # Calculate True Class Probability (TCP)
            # F.softmax(classification_logits, dim=1) gives probabilities for the 8 classes (2-9)
            # target_labels_internal are the ImageFolder internal indices (0-7)
            probs = F.softmax(classification_logits, dim=1)
            tcp_target = probs[torch.arange(probs.size(0)), target_labels_internal]
        
        # Ensure predicted_confidence is squeezed to match tcp_target shape (N,)
        loss = criterion_confid(predicted_confidence.squeeze(), tcp_target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            phase_str = "ConfidNet FineTune ConvNet" if fine_tune_convnet else "ConfidNet Train"
            print(f"{phase_str} Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tConfidLoss: {loss.item():.6f}")

# --- 3. Evaluation Logic ---
def evaluate_model(model, device, test_loader, ood_exclude_digits=None, use_mc_dropout=False, num_mc_samples=10,
                   train_class_names=None, test_class_names=None):
    """
    Evaluates the model's classification accuracy and confidence scores.
    Correctly maps ImageFolder's internal labels to actual digit values for reporting.
    """
    if use_mc_dropout:
        model.configure_train_phase("mc_dropout_eval")
        eval_type_str = f"MC Dropout ({num_mc_samples} samples)"
    else:
        model.configure_train_phase("eval")
        eval_type_str = "Standard"
    print(f"\n--- Evaluation ({eval_type_str}) ---")

    correct_cls = 0
    total = 0
    
    id_mcp_scores, id_confidnet_scores = [], []
    ood_mcp_scores, ood_confidnet_scores = [], []
    
    misclassified_detail_batches_to_show = 2
    misclassified_examples_per_batch_to_show = 3
    misclassified_batches_shown = 0

    ood_detail_print_max = 5 # Max number of OOD sample details to print overall
    ood_printed_count = 0

    print("\n--- Evaluation Details (First few batches & Misclassified & OOD if applicable) ---")
    with torch.no_grad():
        for batch_idx, (data, target_actual_digits) in enumerate(test_loader):
            data, target_actual_digits = data.to(device), target_actual_digits.to(device)

            if use_mc_dropout:
                mc_logits_sum = torch.zeros(data.size(0), model.num_classes).to(device)
                mc_confidence_sum = torch.zeros(data.size(0), 1).to(device)
                for _ in range(num_mc_samples):
                    logits_sample, confidence_scores_sample = model(data)
                    mc_logits_sum += logits_sample
                    mc_confidence_sum += confidence_scores_sample
                logits = mc_logits_sum / num_mc_samples
                confidence_scores = mc_confidence_sum / num_mc_samples
            else:
                logits, confidence_scores = model(data) # confidence_scores from model is (N,1)

            # Get the predicted class index from the model's outputs (0-7 for digits 2-9)
            _, predicted_train_class_idx = torch.max(logits.data, 1)

            # Map the predicted train_class_idx (0-7) back to the actual digit value (2-9)
            # This is crucial for comparing with the test_loader's target_actual_digits (0-9)
            predicted_actual_digits = torch.tensor([int(train_class_names[idx.item()]) for idx in predicted_train_class_idx], device=device)

            total += target_actual_digits.size(0)
            # Compare the mapped predicted digit with the true target digit from the test set
            correct_cls += (predicted_actual_digits == target_actual_digits).sum().item()
            
            mcp_scores = F.softmax(logits, dim=1).max(dim=1)[0] # MCP is still based on model's 8 outputs

            # Print general examples for the first batch only
            if batch_idx == 0:
                print(f"\nBatch {batch_idx + 1} (General Samples):")
                for i in range(min(data.size(0), 5)): # Print up to 5 samples
                    is_correct_str = "Correct" if predicted_actual_digits[i] == target_actual_digits[i] else "INCORRECT"
                    print(f"  Sample {i}: True={target_actual_digits[i].item()}, Pred={predicted_actual_digits[i].item()} ({is_correct_str}), "
                          f"MCP={mcp_scores[i].item():.3f}, ConfidNet={confidence_scores.squeeze()[i].item():.3f}")

            # Log misclassified examples (based on actual digit values)
            misclassified_mask = (predicted_actual_digits != target_actual_digits)
            num_misclassified_in_batch = misclassified_mask.sum().item()
            if num_misclassified_in_batch > 0 and misclassified_batches_shown < misclassified_detail_batches_to_show:
                if misclassified_batches_shown == 0: print("\n--- Misclassified Example Details ---") # Header once
                misclassified_indices = torch.where(misclassified_mask)[0]
                for i in range(min(num_misclassified_in_batch, misclassified_examples_per_batch_to_show)):
                    idx = misclassified_indices[i]
                    print(f"  Batch {batch_idx}, Sample (idx {idx.item()}): True={target_actual_digits[idx].item()}, Predicted={predicted_actual_digits[idx].item()}, "
                          f"MCP={mcp_scores[idx].item():.3f}, ConfidNet={confidence_scores.squeeze()[idx].item():.3f}")
                misclassified_batches_shown +=1
            
            # OOD Sample Analysis (based on actual digit values)
            if ood_exclude_digits:
                for i in range(data.size(0)):
                    sample_true_digit_label = target_actual_digits[i].item() # This is the actual digit (0-9)
                    sample_mcp = mcp_scores[i].item()
                    sample_confidnet = confidence_scores.squeeze()[i].item()
                    is_ood = sample_true_digit_label in ood_exclude_digits

                    if is_ood:
                        ood_mcp_scores.append(sample_mcp)
                        ood_confidnet_scores.append(sample_confidnet)
                        if ood_printed_count < ood_detail_print_max:
                            if ood_printed_count == 0: print("\n--- OOD Sample Details ---") # Header once
                            print(f"  Batch {batch_idx}, Sample (idx {i}): OOD (True Label={sample_true_digit_label}, Pred={predicted_actual_digits[i].item()}): "
                                  f"MCP={sample_mcp:.3f}, ConfidNet={sample_confidnet:.3f}")
                            ood_printed_count += 1
                    else: # In-Distribution
                        id_mcp_scores.append(sample_mcp)
                        id_confidnet_scores.append(sample_confidnet)

    accuracy = 100 * correct_cls / total
    print(f'\nAccuracy of the base classifier on {total} test images: {accuracy:.2f}% (Note: accuracy is on all test samples, including OOD if present)')

    if ood_exclude_digits:
        print("\n--- OOD Analysis Summary ---")
        if id_confidnet_scores:
            avg_id_confidnet = np.mean(id_confidnet_scores)
            avg_id_mcp = np.mean(id_mcp_scores)
            print(f"In-Distribution Samples ({len(id_confidnet_scores)}): Avg ConfidNet={avg_id_confidnet:.3f}, Avg MCP={avg_id_mcp:.3f}")
        else:
            print("No In-Distribution samples found based on ood_exclude_digits (check test set labels).")
        if ood_confidnet_scores:
            avg_ood_confidnet = np.mean(ood_confidnet_scores)
            avg_ood_mcp = np.mean(ood_mcp_scores)
            print(f"Out-of-Distribution Samples ({len(ood_confidnet_scores)}): Avg ConfidNet={avg_ood_confidnet:.3f}, Avg MCP={avg_ood_mcp:.3f}")
        else:
            print("No Out-of-Distribution samples found in the test set based on ood_exclude_digits.")
    return accuracy

# --- 4. Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="ConfidNet Training with OOD Simulation")
    parser.add_argument('--data-path', type=str, default='./processed_mnist',
                        help='path to the root dataset directory (containing train/test subfolders)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--classifier-epochs', type=int, default=15, metavar='N', # Increased default epochs
                        help='number of epochs to train the classifier (default: 15)')
    parser.add_argument('--confidnet-epochs', type=int, default=10, metavar='N', # Increased default epochs
                        help='number of epochs to train ConfidNet head (default: 10)')
    parser.add_argument('--confidnet-finetune-epochs', type=int, default=5, metavar='N', # Increased default epochs
                        help='number of epochs to fine-tune ConvNet with ConfidNet (default: 5, set to 0 to disable)')
    parser.add_argument('--lr-classifier', type=float, default=0.01, metavar='LR',
                        help='learning rate for classifier (default: 0.01)')
    parser.add_argument('--lr-confidnet', type=float, default=0.001, metavar='LR',
                        help='learning rate for ConfidNet head (default: 0.001)')
    parser.add_argument('--lr-confidnet-finetune', type=float, default=0.0001, metavar='LR',
                        help='learning rate for ConfidNet ConvNet fine-tuning (default: 0.0001, typically smaller)')
    parser.add_argument('--momentum-classifier', type=float, default=0.5, metavar='M',
                        help='SGD momentum for classifier (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    # Removed --num-classes from argparse, as it will be inferred from train_dataset
    parser.add_argument('--input-channels', type=int, default=1,
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--mc-dropout', action='store_true', default=False,
                        help='enable Monte Carlo Dropout during evaluation')
    parser.add_argument('--num-mc-samples', type=int, default=10,
                        help='number of Monte Carlo samples for dropout during evaluation (default: 10)')
    parser.add_argument('--ood-exclude-digits', type=int, nargs='+', default=[0, 1], # Default to 0 and 1
                        help='List of digit labels to exclude from training to simulate OOD (e.g., 0 1).')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    dataset_mean = (0.1307,) # MNIST mean for 1 channel
    dataset_std = (0.3081,)  # MNIST std for 1 channel
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

    # Instantiate the model with the CORRECT number of output classes based on training data
    model = CustomConfidNet(num_classes=len(train_dataset.classes), input_channels=args.input_channels).to(device)

    # --- Phase 1: Train Classifier ---
    print("\n--- Training Classifier ---")
    optimizer_cls = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr_classifier, momentum=args.momentum_classifier)
    for epoch in range(1, args.classifier_epochs + 1):
        train_classifier_epoch(model, device, train_loader, optimizer_cls, epoch, args.log_interval)
        # Pass train_dataset.classes for correct label mapping in evaluation output
        evaluate_model(model, device, test_loader, ood_exclude_digits=args.ood_exclude_digits,
                       use_mc_dropout=args.mc_dropout, num_mc_samples=args.num_mc_samples,
                       train_class_names=train_dataset.classes, test_class_names=test_dataset.classes)

    # --- Phase 2: Train ConfidNet Head ---
    print("\n--- Training ConfidNet Head (ConvNet Frozen) ---")
    optimizer_conf = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_confidnet)
    for epoch in range(1, args.confidnet_epochs + 1):
        train_confidnet_epoch(model, device, train_loader, optimizer_conf, epoch, args.log_interval, fine_tune_convnet=False)
        # Pass train_dataset.classes for correct label mapping in evaluation output
        evaluate_model(model, device, test_loader, ood_exclude_digits=args.ood_exclude_digits,
                       use_mc_dropout=args.mc_dropout, num_mc_samples=args.num_mc_samples,
                       train_class_names=train_dataset.classes, test_class_names=test_dataset.classes)
    
    # --- Phase 3: Fine-tuning ConvNet with ConfidNet ---
    if args.confidnet_finetune_epochs > 0:
        print("\n--- Fine-tuning ConvNet with ConfidNet ---")
        optimizer_finetune = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_confidnet_finetune)
        for epoch in range(1, args.confidnet_finetune_epochs + 1):
            train_confidnet_epoch(model, device, train_loader, optimizer_finetune, epoch, args.log_interval, fine_tune_convnet=True)
            # Pass train_dataset.classes for correct label mapping in evaluation output
            evaluate_model(model, device, test_loader, ood_exclude_digits=args.ood_exclude_digits,
                           use_mc_dropout=args.mc_dropout, num_mc_samples=args.num_mc_samples,
                           train_class_names=train_dataset.classes, test_class_names=test_dataset.classes)

    print("\n--- Final Overall Evaluation ---")
    final_accuracy = evaluate_model(model, device, test_loader, ood_exclude_digits=args.ood_exclude_digits,
                                    use_mc_dropout=args.mc_dropout, num_mc_samples=args.num_mc_samples,
                                    train_class_names=train_dataset.classes, test_class_names=test_dataset.classes)
    print(f"Final accuracy after all phases: {final_accuracy:.2f}%")

if __name__ == '__main__':
    main()



#methods for imbalanced regression
#run tcp new on the new dataset
#check why confidnet is not working for this one