import pandas as pd
import numpy as np
import torch
import yaml
import argparse
import random
import matplotlib.pyplot as plt
import os

from torch.optim import AdamW, Adam, SGD
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# --- Utility Functions ---

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_metrics(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.title("Training and Validation Loss")
    plt.legend(); plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.title("Validation Accuracy")
    plt.legend(); plt.show()

# --- Core Classes and Functions ---

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

class CustomModel(torch.nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate):
        super(CustomModel, self).__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
    def forward(self, **kwargs):
        output = self.base_model(**kwargs)
        logits = self.dropout(output.logits)
        return SequenceClassifierOutput(logits=logits, hidden_states=output.hidden_states, attentions=output.attentions)

def tokenize_data(data, tokenizer, max_length):
    return tokenizer(data["title"].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

def evaluate_model(model, data_loader, device, loss_fn):
    model.eval()
    predictions, true_labels, total_loss = [], [], 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch["labels"])
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, axis=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    return predictions, true_labels, avg_loss

def train_model(model, train_loader, val_loader, optimizer, device, loss_fn, config, best_model_path):
    best_accuracy = 0
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Training Loss: {avg_train_loss:.4f}")

        predictions, true_labels, avg_val_loss = evaluate_model(model, val_loader, device, loss_fn)
        accuracy = accuracy_score(true_labels, predictions)
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            
    return train_losses, val_losses, val_accuracies

# --- Main Execution ---

def main(config):
    """Main function to run the training and evaluation pipeline."""
    set_seed(config['seed'])
    
    best_model_path = "best_model.pth"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data, tokenizer, and model
    df = pd.read_csv(config['data_path'])
    df_cropped = df[["title", "movement"]].copy()
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = CustomModel(config['model_name'], config['num_labels'], config['dropout'])
    model.to(device)

    # --- Mode-Specific Execution ---
    mode = config.get('operation_mode', 'train')
    print(f"--- Running in {mode.upper()} mode ---")

    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(df_cropped, test_size=config['test_split_size'], stratify=df_cropped["movement"], random_state=config['seed'])
    val_df, test_df = train_test_split(temp_df, test_size=config['validation_split_size'], stratify=temp_df["movement"], random_state=config['seed'])

    if mode == 'train':
        # Tokenize all data splits
        train_encodings = tokenize_data(train_df, tokenizer, config['max_len'])
        val_encodings = tokenize_data(val_df, tokenizer, config['max_len'])
        
        # Create Datasets and DataLoaders for training
        train_dataset = SentimentDataset(train_encodings, torch.tensor(train_df["movement"].values))
        val_dataset = SentimentDataset(val_encodings, torch.tensor(val_df["movement"].values))
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # Setup loss function with class weights
        class_weights = compute_class_weight("balanced", classes=np.unique(train_df["movement"]), y=train_df["movement"])
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
        
        # Setup optimizer based on config
        optimizer_name = config.get('optimizer_name', 'AdamW').lower()
        lr = config['learning_rate']
        weight_decay = config.get('weight_decay', 0.0)

        if optimizer_name == 'adamw':
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: '{config['optimizer_name']}'. Please use 'AdamW', 'Adam', or 'SGD'.")
        
        print(f"Using optimizer: {optimizer_name.upper()}")

        # Train the model
        print("\nFine-Tuning the Model...")
        train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, optimizer, device, loss_fn, config, best_model_path)
        plot_metrics(train_losses, val_losses, val_accuracies)
        
        # Load the best performing model for final evaluation
        model.load_state_dict(torch.load(best_model_path))
        print(f"\nLoaded best model from {best_model_path} for final evaluation.")

    elif mode == 'evaluate':
        if not os.path.exists(best_model_path):
            print(f"Error: Model file not found at '{best_model_path}'. Please run in 'train' mode first.")
            return
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded model from {best_model_path} for evaluation.")

    # --- Final Evaluation on Test Set (runs after training or by itself in evaluate mode) ---
    print("\nEvaluating Model on Test Set...")
    test_encodings = tokenize_data(test_df, tokenizer, config['max_len'])
    test_dataset = SentimentDataset(test_encodings, torch.tensor(test_df["movement"].values))
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Use a non-weighted loss for final evaluation reporting if desired, or keep the weighted one
    eval_loss_fn = torch.nn.CrossEntropyLoss().to(device)

    test_predictions, test_true_labels, test_loss = evaluate_model(model, test_loader, device, eval_loss_fn)
    test_accuracy = accuracy_score(test_true_labels, test_predictions)
    
    print(f"Test Set Loss: {test_loss:.4f}")
    print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")
    print(classification_report(test_true_labels, test_predictions, target_names=['down', 'up']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate a sentiment analysis model using a config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)
```# filepath: main.py
import pandas as pd
import numpy as np
import torch
import yaml
import argparse
import random
import matplotlib.pyplot as plt
import os

from torch.optim import AdamW, Adam, SGD
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# --- Utility Functions ---

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_metrics(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.title("Training and Validation Loss")
    plt.legend(); plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.title("Validation Accuracy")
    plt.legend(); plt.show()

# --- Core Classes and Functions ---

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

class CustomModel(torch.nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate):
        super(CustomModel, self).__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
    def forward(self, **kwargs):
        output = self.base_model(**kwargs)
        logits = self.dropout(output.logits)
        return SequenceClassifierOutput(logits=logits, hidden_states=output.hidden_states, attentions=output.attentions)

def tokenize_data(data, tokenizer, max_length):
    return tokenizer(data["title"].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

def evaluate_model(model, data_loader, device, loss_fn):
    model.eval()
    predictions, true_labels, total_loss = [], [], 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch["labels"])
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, axis=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    return predictions, true_labels, avg_loss

def train_model(model, train_loader, val_loader, optimizer, device, loss_fn, config, best_model_path):
    best_accuracy = 0
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Training Loss: {avg_train_loss:.4f}")

        predictions, true_labels, avg_val_loss = evaluate_model(model, val_loader, device, loss_fn)
        accuracy = accuracy_score(true_labels, predictions)
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            
    return train_losses, val_losses, val_accuracies

# --- Main Execution ---

def main(config):
    """Main function to run the training and evaluation pipeline."""
    set_seed(config['seed'])
    
    best_model_path = "best_model.pth"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data, tokenizer, and model
    df = pd.read_csv(config['data_path'])
    df_cropped = df[["title", "movement"]].copy()
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = CustomModel(config['model_name'], config['num_labels'], config['dropout'])
    model.to(device)

    # --- Mode-Specific Execution ---
    mode = config.get('operation_mode', 'train')
    print(f"--- Running in {mode.upper()} mode ---")

    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(df_cropped, test_size=config['test_split_size'], stratify=df_cropped["movement"], random_state=config['seed'])
    val_df, test_df = train_test_split(temp_df, test_size=config['validation_split_size'], stratify=temp_df["movement"], random_state=config['seed'])

    if mode == 'train':
        # Tokenize all data splits
        train_encodings = tokenize_data(train_df, tokenizer, config['max_len'])
        val_encodings = tokenize_data(val_df, tokenizer, config['max_len'])
        
        # Create Datasets and DataLoaders for training
        train_dataset = SentimentDataset(train_encodings, torch.tensor(train_df["movement"].values))
        val_dataset = SentimentDataset(val_encodings, torch.tensor(val_df["movement"].values))
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # Setup loss function with class weights
        class_weights = compute_class_weight("balanced", classes=np.unique(train_df["movement"]), y=train_df["movement"])
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
        
        # Setup optimizer based on config
        optimizer_name = config.get('optimizer_name', 'AdamW').lower()
        lr = config['learning_rate']
        weight_decay = config.get('weight_decay', 0.0)

        if optimizer_name == 'adamw':
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: '{config['optimizer_name']}'. Please use 'AdamW', 'Adam', or 'SGD'.")
        
        print(f"Using optimizer: {optimizer_name.upper()}")

        # Train the model
        print("\nFine-Tuning the Model...")
        train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, optimizer, device, loss_fn, config, best_model_path)
        plot_metrics(train_losses, val_losses, val_accuracies)
        
        # Load the best performing model for final evaluation
        model.load_state_dict(torch.load(best_model_path))
        print(f"\nLoaded best model from {best_model_path} for final evaluation.")

    elif mode == 'evaluate':
        if not os.path.exists(best_model_path):
            print(f"Error: Model file not found at '{best_model_path}'. Please run in 'train' mode first.")
            return
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded model from {best_model_path} for evaluation.")

    # --- Final Evaluation on Test Set (runs after training or by itself in evaluate mode) ---
    print("\nEvaluating Model on Test Set...")
    test_encodings = tokenize_data(test_df, tokenizer, config['max_len'])
    test_dataset = SentimentDataset(test_encodings, torch.tensor(test_df["movement"].values))
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Use a non-weighted loss for final evaluation reporting if desired, or keep the weighted one
    eval_loss_fn = torch.nn.CrossEntropyLoss().to(device)

    test_predictions, test_true_labels, test_loss = evaluate_model(model, test_loader, device, eval_loss_fn)
    test_accuracy = accuracy_score(test_true_labels, test_predictions)
    
    print(f"Test Set Loss: {test_loss:.4f}")
    print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")
    print(classification_report(test_true_labels, test_predictions, target_names=['down', 'up']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate a sentiment analysis model using a config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)