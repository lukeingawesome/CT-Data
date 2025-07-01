import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.image.data.transforms import get_chest_xray_transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import the dataset and model classes from classification.py
from finetune import CXRClassificationDataset, ClassificationModel

def load_model(checkpoint_path, device):
    # Initialize image encoder
    image_encoder = get_biovil_t_image_encoder()
    image_encoder = image_encoder.to(torch.bfloat16)

    # Initialize model
    model = ClassificationModel(image_encoder, num_classes=3, mode='full')
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate(model, test_loader, device):
    all_preds, all_labels, all_probs = [], [], []
    all_switch_preds, all_switch_probs = [], []
    all_combined_preds, all_combined_probs = [], []

    with torch.no_grad():
        for current_images, previous_images, labels in tqdm(test_loader, desc='Evaluating'):
            current_images = current_images.to(device)
            previous_images = previous_images.to(device)
            labels = labels.to(device).long()

            # Forward pass - original order
            outputs = model(current_images, previous_images)
            probs = torch.softmax(outputs.to(torch.float32), dim=1)
            preds = torch.argmax(probs, dim=1)

            # Forward pass - switched order
            switch_outputs = model(previous_images, current_images)
            switch_probs = torch.softmax(switch_outputs.to(torch.float32), dim=1)
            switch_preds = torch.argmax(switch_probs, dim=1)

            # Reverse the switch probabilities (swap class 0 <-> 2)
            reversed_switch_probs = switch_probs.clone()
            reversed_switch_probs[:, 0] = switch_probs[:, 2]
            reversed_switch_probs[:, 2] = switch_probs[:, 0]
            # TILA Inference
            combined_probs = (probs + reversed_switch_probs) / 2
            combined_preds = torch.argmax(combined_probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_switch_preds.extend(switch_preds.cpu().numpy())
            all_switch_probs.extend(switch_probs.cpu().numpy())
            all_combined_preds.extend(combined_preds.cpu().numpy())
            all_combined_probs.extend(combined_probs.cpu().numpy())

    return (np.array(all_preds), np.array(all_labels), np.array(all_probs),
            np.array(all_switch_preds), np.array(all_switch_probs),
            np.array(all_combined_preds), np.array(all_combined_probs))

def plot_confusion_matrix(y_true, y_pred, output_dir):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_metrics(y_true, y_pred, y_prob, y_switch_pred, y_switch_prob, y_combined_pred, y_combined_prob):
    # Original prediction metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    class_report = classification_report(y_true, y_pred, output_dict=True)

    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    macro_accuracy = np.mean(per_class_accuracy)

    # Switched predictions: swap 0<->2
    switch_labels = 2 - y_true
    switch_accuracy = accuracy_score(switch_labels, y_switch_pred)
    switch_cm = confusion_matrix(switch_labels, y_switch_pred)
    switch_per_class_accuracy = switch_cm.diagonal() / switch_cm.sum(axis=1)
    switch_macro_accuracy = np.mean(switch_per_class_accuracy)

    # Combined prediction metrics
    combined_accuracy = accuracy_score(y_true, y_combined_pred)
    combined_cm = confusion_matrix(y_true, y_combined_pred)
    combined_per_class_accuracy = combined_cm.diagonal() / combined_cm.sum(axis=1)
    combined_macro_accuracy = np.mean(combined_per_class_accuracy)

    # Row-wise (both predictions correct)
    row_success = (y_pred == y_true) & (y_switch_pred == switch_labels)
    combined_row_accuracy = np.mean(row_success)

    labels = np.unique(y_true)
    per_label_success = []
    for label in labels:
        label_mask = (y_true == label)
        if np.sum(label_mask) == 0:
            label_success_rate = 0
        else:
            label_success_rate = np.sum(row_success & label_mask) / np.sum(label_mask)
        per_label_success.append(label_success_rate)
    macro_row_accuracy = np.mean(per_label_success)

    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Macro Accuracy: {macro_accuracy:.4f}")
    print(f"Switched Macro Accuracy: {switch_macro_accuracy:.4f}")
    print(f"Combined Macro Accuracy: {combined_macro_accuracy:.4f}")
    print(f"Macro Row Accuracy: {macro_row_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return y_pred

def save_predictions(df, y_pred, y_prob):
    # Create results dataframe
    results_df = df.copy()
    results_df['predicted_label'] = y_pred
    results_df['prediction_confidence'] = np.max(y_prob, axis=1)
    for i in range(y_prob.shape[1]):
        results_df[f'prob_class_{i}'] = y_prob[:, i]
    return results_df

def main():
    checkpoint_path = 'your_finetuned_checkpoint'
    test_csv = 'pneumonia_test_mscxrt.csv'
    img_key = 'img_path'
    prev_img_key = 'previous_img_path'
    label_key = 'label'
    batch_size = 16

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    _, test_transform = get_chest_xray_transforms(size=448, crop_size=448)

    test_dataset = CXRClassificationDataset(
        test_csv,
        img_key,
        prev_img_key,
        label_key,
        test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    print("Running evaluation...")
    y_pred, y_true, y_prob, y_switch_pred, y_switch_prob, y_combined_pred, y_combined_prob = evaluate(
        model, test_loader, device
    )

    # Save metrics
    _ = save_metrics(y_true, y_pred, y_prob, y_switch_pred, y_switch_prob, y_combined_pred, y_combined_prob)

if __name__ == "__main__":
    main()
