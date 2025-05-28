import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- 1. Model Definition ---
class CustomConfidNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=1): 
        super().__init__()
        #COnvolutional Feature Extractor 
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout_feat_ext = nn.Dropout(0.25)
     
        #Fully connected layers
        self.fc_feat_input_dim = 64 * 12 * 12 # Calculated for 28x28 input after convs/pool
        self.fc_feat = nn.Linear(self.fc_feat_input_dim, 128)
        self.dropout_fc_feat_ext = nn.Dropout(0.5)

        # Classification Head outputs logits for each class
        self.classifier_fc = nn.Linear(128, num_classes)

        # Confidence Head : 3 layer Multi layer perceptron (MLP) to predict confidence score
        # This head takes the same features as the classifier
        # and outputs a single confidence score in [0, 1] range
        self.confid_fc1 = nn.Linear(128, 64) 
        self.confid_fc2 = nn.Linear(64, 32)
        self.confid_fc3 = nn.Linear(32, 1)  

        # Store layer groups for easier management
        self._feature_extractor_modules = nn.ModuleList([
            self.conv1, self.conv2, self.fc_feat
        ])
        self._classifier_head_modules = nn.ModuleList([self.classifier_fc])
        self._confidence_head_modules = nn.ModuleList([
            self.confid_fc1, self.confid_fc2, self.confid_fc3
        ])

    def get_features(self, x): #takes input image x and returns features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout_feat_ext(x)
        x = x.view(x.size(0), -1) # Flatten
        if x.shape[1] != self.fc_feat_input_dim:
             raise ValueError(
                f"Unexpected feature shape after conv layers. "
                f"Expected {self.fc_feat_input_dim}, got {x.shape[1]}. "
                f"Check image dimensions and conv architecture."
            )
        x = F.relu(self.fc_feat(x))
        x = self.dropout_fc_feat_ext(x)
        return x

    def forward(self, x): # x is a batch of images
        features = self.get_features(x)
        classification_logits = self.classifier_fc(features) # gives raw clas logits for each class
        # Confidence score prediction
        # This is a single value in [0, 1] indicating confidence in the predicted class
        confid_hidden = F.relu(self.confid_fc1(features))
        confid_hidden = F.relu(self.confid_fc2(confid_hidden))
        predicted_confidence = torch.sigmoid(self.confid_fc3(confid_hidden)) # Sigmoid for [0,1] score [cite: 69]
        return classification_logits, predicted_confidence

    #helper function to freeze or unfreeze modules
    # This is used to set requires_grad for all parameters in a list of modules
    def _set_module_list_grad(self, module_list, requires_grad_bool):
        for modules in module_list:
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = requires_grad_bool

    def configure_train_phase(self, phase="classification_train"):
        if phase == "classification_train":
            # train feature extractor and classifier head
            self.train()
            self._set_module_list_grad([self._feature_extractor_modules, self._classifier_head_modules], True)
            self._set_module_list_grad([self._confidence_head_modules], False)
        elif phase == "confidence_train":
            # Freeze feature extractor and classifier head
            for module in self._feature_extractor_modules: module.eval() # Freeze feature extractor
            self.dropout_feat_ext.eval() # Important for freezing
            self.dropout_fc_feat_ext.eval() # Important for freezing
            for module in self._classifier_head_modules: module.eval() 
            # Train only confidence head
            for module in self._confidence_head_modules: module.train()
            self._set_module_list_grad([self._feature_extractor_modules, self._classifier_head_modules], False) # Freeze them
            self._set_module_list_grad([self._confidence_head_modules], True)
        elif phase == "confidence_finetune_convnet": 
            self.train() # Set the whole model to train mode
            # But only ConvNet encoder and ConfidNet layers are actually updated
            self._set_module_list_grad([self._feature_extractor_modules, self._confidence_head_modules], True)
            self._set_module_list_grad([self._classifier_head_modules], False) # Classifier remains fixed [cite: 80]
            # Deactivate dropout during this fine-tuning phase as per paper [cite: 81]
            self.dropout_feat_ext.eval()
            self.dropout_fc_feat_ext.eval()
        elif phase == "eval":
            self.eval()
            self._set_module_list_grad(
                [self._feature_extractor_modules, self._classifier_head_modules, self._confidence_head_modules],
                False
            )
        elif phase == "mc_dropout_eval":
            self.eval() # Set all modules to eval mode (disables batch norm updates, no gradients)
            # Explicitly set dropout layers to train mode to activate dropout during inference
            self.dropout_feat_ext.train()
            self.dropout_fc_feat_ext.train()
            # Parameters will not have requires_grad=True, as we are still in inference context
        
        else:
            raise ValueError(f"Unknown training phase: {phase}")

# --- 2. Training Logic ---
def train_classifier_epoch(model, device, train_loader, optimizer, epoch, log_interval):
    model.configure_train_phase("classification_train")
    criterion = nn.CrossEntropyLoss() 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Classifier Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

def train_confidnet_epoch(model, device, train_loader, optimizer, epoch, log_interval, fine_tune_convnet=False):
    if fine_tune_convnet:
        model.configure_train_phase("confidence_finetune_convnet")
    else:
        model.configure_train_phase("confidence_train")

    criterion_confid = nn.MSELoss() # L_conf (MSE loss) for regressing TCP 
    for batch_idx, (data, target_labels) in enumerate(train_loader):
        data, target_labels = data.to(device), target_labels.to(device)
        optimizer.zero_grad()
        classification_logits, predicted_confidence = model(data)

        with torch.no_grad(): # TCP target should not require gradients
            probs = F.softmax(classification_logits, dim=1) #compute class probabilities
            # True Class Probability (TCP) target c*(x, y*) = P(Y=y*|w,x) 
            tcp_target = probs[torch.arange(probs.size(0)), target_labels] # This is the probability of the true class

        loss = criterion_confid(predicted_confidence.squeeze(), tcp_target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            phase_str = "ConfidNet FineTune ConvNet" if fine_tune_convnet else "ConfidNet Train"
            print(f"{phase_str} Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tConfidLoss: {loss.item():.6f}")

# --- 3. Evaluation Logic ---
def evaluate_model(model, device, test_loader, use_mc_dropout=False, num_mc_samples=10):
    if use_mc_dropout:
        model.configure_train_phase("mc_dropout_eval")
        print(f"\n--- Evaluation with MC Dropout ({num_mc_samples} samples) ---")
    else:
        model.configure_train_phase("eval")
        print("\n--- Standard Evaluation ---")

    correct_cls = 0
    total = 0
    all_targets = []
    all_predicted_labels = []
    all_mcp_scores = []
    all_confidnet_scores = []

    print("\n--- Evaluation Examples ---")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

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
                logits, confidence_scores = model(data)

            _, predicted_labels = torch.max(logits.data, 1)

            total += target.size(0)
            correct_cls += (predicted_labels == target).sum().item()

            mcp_scores = F.softmax(logits, dim=1).max(dim=1)[0] # Maximum Class Probability (MCP)

            all_targets.extend(target.cpu().numpy())
            all_predicted_labels.extend(predicted_labels.cpu().numpy())
            all_mcp_scores.extend(mcp_scores.cpu().numpy())
            all_confidnet_scores.extend(confidence_scores.squeeze().cpu().numpy())

            if batch_idx < 2: # Print for the first few batches
                print(f"\nBatch {batch_idx + 1}:")
                print(f"True Labels:      {target[:8].cpu().numpy()}")
                print(f"Predicted Labels: {predicted_labels[:8].cpu().numpy()}")
                print(f"Max Class Prob:   {[f'{x:.3f}' for x in mcp_scores[:8].cpu().numpy()]}")
                print(f"ConfidNet Scores: {[f'{x:.3f}' for x in confidence_scores[:8].squeeze().cpu().numpy()]}")

    accuracy = 100 * correct_cls / total
    print(f'\nAccuracy of the base classifier on {total} test images: {accuracy:.2f}%')
    
    return accuracy

# --- 4. Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="ConfidNet Training for PNG Dataset")
    parser.add_argument('--data-path', type=str, default='./processed_mnist',
                        help='path to the root dataset directory (containing train/test subfolders)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--classifier-epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train the classifier (default: 3)')
    parser.add_argument('--confidnet-epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train ConfidNet head (default: 3)')
    parser.add_argument('--confidnet-finetune-epochs', type=int, default=2, metavar='N',
                        help='number of epochs to fine-tune ConvNet with ConfidNet (default: 2, set to 0 to disable)')
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
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes in the dataset')
    parser.add_argument('--input-channels', type=int, default=1,
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--mc-dropout', action='store_true', default=False,
                        help='enable Monte Carlo Dropout during evaluation')
    parser.add_argument('--num-mc-samples', type=int, default=10,
                        help='number of Monte Carlo samples for dropout during evaluation (default: 10)')


    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    dataset_mean = (0.1307,)
    dataset_std = (0.3081,)

    transform_list = []
    if args.input_channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1)) # Ensure 1 channel
    transform_list.extend([
        transforms.Resize((28, 28)), # Ensure consistent input size for the network
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

    model = CustomConfidNet(num_classes=args.num_classes, input_channels=args.input_channels).to(device)

    # --- Phase 1: Train Classifier ---
    print("--- Training Classifier ---")
    model.configure_train_phase("classification_train")
    optimizer_cls = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr_classifier, momentum=args.momentum_classifier)
    for epoch in range(1, args.classifier_epochs + 1):
        train_classifier_epoch(model, device, train_loader, optimizer_cls, epoch, args.log_interval)
    
    # Evaluate after classifier training, potentially with MC Dropout
    evaluate_model(model, device, test_loader, use_mc_dropout=args.mc_dropout, num_mc_samples=args.num_mc_samples) 

    # --- Phase 2: Train ConfidNet Head ---
    print("\n--- Training ConfidNet Head (ConvNet Frozen) ---")
    model.configure_train_phase("confidence_train")
    optimizer_conf = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_confidnet)
    for epoch in range(1, args.confidnet_epochs + 1):
        train_confidnet_epoch(model, device, train_loader, optimizer_conf, epoch, args.log_interval, fine_tune_convnet=False)
    
    # Evaluate after ConfidNet head training, potentially with MC Dropout
    evaluate_model(model, device, test_loader, use_mc_dropout=args.mc_dropout, num_mc_samples=args.num_mc_samples)

    if args.confidnet_finetune_epochs > 0:
        print("\n--- Fine-tuning ConvNet with ConfidNet ---")
        model.configure_train_phase("confidence_finetune_convnet")
        optimizer_finetune = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_confidnet_finetune)
        for epoch in range(1, args.confidnet_finetune_epochs + 1):
            train_confidnet_epoch(model, device, train_loader, optimizer_finetune, epoch, args.log_interval, fine_tune_convnet=True)
        
        # Evaluate after fine-tuning, potentially with MC Dropout
        evaluate_model(model, device, test_loader, use_mc_dropout=args.mc_dropout, num_mc_samples=args.num_mc_samples)


    # --- Final Evaluation ---
    print("\n--- Final Evaluation After All Training Phases ---")
    final_accuracy = evaluate_model(model, device, test_loader, use_mc_dropout=args.mc_dropout, num_mc_samples=args.num_mc_samples)
    print(f"Final accuracy: {final_accuracy:.2f}%")

if __name__ == '__main__':
    main()