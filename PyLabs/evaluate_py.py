#!/usr/bin/env python3
"""
ASL Model Evaluation Script
Evaluate trained PBN model on test dataset
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
import json

from train import PatchBasedNetwork, ASLDataset

def evaluate_model(model_path, test_data_dir, device='cuda'):
    """
    Evaluate the trained model on test dataset
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = PatchBasedNetwork(num_classes=26)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Test data transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = ASLDataset(test_data_dir, 'test', test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Class names (A-Z)
    class_names = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    # Classification report
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ASL Recognition')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Plot per-class accuracy
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, per_class_acc, color='skyblue', alpha=0.7)
    plt.axhline(y=accuracy, color='red', linestyle='--', 
                label=f'Overall Accuracy: {accuracy:.3f}')
    plt.xlabel('ASL Letters')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=300)
    plt.show()
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'precision': float(report['weighted avg']['precision']),
        'recall': float(report['weighted avg']['recall']),
        'f1_score': float(report['weighted avg']['f1-score']),
        'per_class_accuracy': {
            class_names[i]: float(acc) for i, acc in enumerate(per_class_acc)
        },
        'classification_report': report
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to 'evaluation_results.json'")
    print("Confusion matrix saved to 'confusion_matrix.png'")
    print("Per-class accuracy plot saved to 'per_class_accuracy.png'")
    
    return results

def plot_training_history(history_file='training_history.json'):
    """
    Plot training history
    """
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training loss
    ax1.plot(history['train_losses'], label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Validation accuracy
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()

def main():
    # Evaluate model
    results = evaluate_model(
        model_path='pbn_asl_model.pth',
        test_data_dir='data/ASL-A'
    )
    
    # Plot training history if available
    try:
        plot_training_history()
    except FileNotFoundError:
        print("Training history not found. Train the model first.")

if __name__ == "__main__":
    main()