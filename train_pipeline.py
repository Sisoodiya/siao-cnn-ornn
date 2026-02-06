"""
Complete SIAO-CNN-ORNN Training Pipeline

Integrates all components:
1. Data Pipeline - Load and preprocess reactor data
2. Window Processor - Create sliding windows
3. CNN Feature Extractor - Extract spatial features
4. ORNN - SIAO-optimized RNN for classification

Author: SIAO-CNN-ORNN Integration
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_complete_pipeline(
    data_dir: str = 'data/',
    window_size: int = 100, # Increased from 50 to match specs
    stride: int = 25,
    cnn_embedding_dim: int = 256,
    wks_dim: int = 43,  # WKS features dimension (num signals)
    rnn_hidden_size: int = 128,
    num_classes: int = 6,
    test_size: float = 0.2,
    wks_pop_size: int = 15,
    wks_max_iter: int = 30,
    siao_pop_size: int = 20,
    siao_max_iter: int = 50,
    bp_epochs: int = 100,  # Scaled to match iteration count
    batch_size: int = 163, # Updated to 163 as per research specs
    use_class_weights: bool = True
) -> Dict:
    """
    Run the complete SIAO-CNN-ORNN training pipeline.
    
    Args:
        data_dir: Path to data directory
        window_size: Sliding window size
        stride: Window stride
        cnn_embedding_dim: CNN output dimension
        rnn_hidden_size: RNN hidden size
        num_classes: Number of output classes
        test_size: Validation split ratio
        siao_pop_size: SIAO population size
        siao_max_iter: SIAO iterations
        bp_epochs: Backpropagation epochs
        batch_size: Training batch size
    
    Returns:
        Dictionary with results and history
    """
    
    # Rich Imports
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    
    console = Console()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold blue]Device:[/bold blue] {device}")
    
    # =========================================================================
    # Step 1: Load and Preprocess Data
    # =========================================================================
    console.print(Panel("[bold green]Step 1: Loading and Preprocessing Data[/bold green]", box=box.DOUBLE))
    
    from stage1_data.data_pipeline import IP200DataPipeline
    
    with console.status("[bold green]Ingesting data...[/bold green]", spinner="dots"):
        pipeline = IP200DataPipeline(
            data_dir=data_dir,
            normalization='zscore',
            outlier_window=5,
            handle_missing='interpolate'
        )
        
        # Use caching to speed up subsequent runs
        X_raw, y_raw = pipeline.run(use_cache=True)
    
    console.print(f" [bold]Raw data loaded:[/bold] X={X_raw.shape}, y={y_raw.shape}")
    
    # =========================================================================
    # Step 2: Create Sliding Windows
    # =========================================================================
    console.print(Panel("[bold green]Step 2: Creating Sliding Windows[/bold green]", box=box.DOUBLE))
    
    from stage1_data.window_processor import SlidingWindowProcessor
    
    with console.status("[bold green]Processing windows...[/bold green]", spinner="dots"):
        window_proc = SlidingWindowProcessor(
            window_size=window_size,
            stride=stride,
            padding='zero'
        )
        
        X_windows, y_windows = window_proc.transform(X_raw, y_raw)
    
    console.print(f" [bold]Windowed data:[/bold] X={X_windows.shape}, y={y_windows.shape}")
    
    # =========================================================================
    # Step 3: Train/Validation Split
    # =========================================================================
    console.print(Panel("[bold green]Step 3: Train/Validation Split[/bold green]", box=box.DOUBLE))
    
    # Stratified split
    train_idx, val_idx = train_test_split(
        np.arange(len(y_windows)),
        test_size=test_size,
        stratify=y_windows,
        random_state=42
    )
    
    X_train = X_windows[train_idx]
    y_train = y_windows[train_idx]
    X_val = X_windows[val_idx]
    y_val = y_windows[val_idx]
    
    console.print(f" [bold]Train:[/bold] X={X_train.shape}, y={y_train.shape}")
    console.print(f" [bold]Val:[/bold]   X={X_val.shape}, y={y_val.shape}")
    
    # =========================================================================
    # Step 4: CNN Feature Extraction
    # =========================================================================
    console.print(Panel("[bold green]Step 4: CNN Feature Extraction[/bold green]", box=box.DOUBLE))
    
    from stage3_models.cnn_model import CNNFeatureExtractorForSequence, create_cnn_extractor
    
    # Input shape: [samples, time_steps, features]
    input_channels = X_train.shape[2]  # Features (43)
    
    # Initialize CNN
    cnn = create_cnn_extractor(
        input_shape=(window_size, input_channels),
        embedding_dim=cnn_embedding_dim,
        dropout=0.2
    ).to(device)
    
    # Convert to tensors for CNN feature extraction
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    # Extract features in batches meant for sequence processing
    # Note: CNNFeatureExtractorForSequence isn't strictly needed unless we treat 
    # the window itself as a sequence of smaller chunks. 
    # Here, we treat the window as an image (1, H, W) or signal (C, L).
    # Since our CNN is 2D, we likely need to reshape: [batch, 1, time, features]
    
    # For efficiency, we'll process in batches
    def extract_cnn_features(model, X_tensor, batch_size=64):
        model.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                # Reshape: [batch, 1, time, features]
                batch = batch.unsqueeze(1)
                emb = model(batch)
                embeddings.append(emb.cpu().numpy())
        return np.vstack(embeddings)

    with console.status("[bold green]Extracting CNN features...[/bold green]", spinner="dots"):
        X_train_cnn = extract_cnn_features(cnn, X_train_t)
        X_val_cnn = extract_cnn_features(cnn, X_val_t)
    
    console.print(f"[bold]CNN embeddings:[/bold] Train={X_train_cnn.shape}, Val={X_val_cnn.shape}")
    
    # =========================================================================
    # Step 4.5: WKS Feature Extraction (Aquila-optimized)
    # =========================================================================
    console.print(Panel("[bold green]Step 4.5: WKS Feature Extraction (Aquila-optimized)[/bold green]", box=box.DOUBLE))
    
    from stage4_optimizers.aquila_optimizer import WKSOptimizer
    from stage2_features.feature_extractor import extract_statistical_features
    
    # Extract Standard Statistical Features
    console.print("[bold green]Extracting Standard Statistical Features...[/bold green]")
    X_train_stat = extract_statistical_features(X_train)
    X_val_stat = extract_statistical_features(X_val)
    console.print(f"[bold]Statistical features:[/bold] Train={X_train_stat.shape}, Val={X_val_stat.shape}")

    # We need to find optimal omega using Training data
    # Create optimizer
    wks_opt = WKSOptimizer(
        pop_size=wks_pop_size,
        max_iter=wks_max_iter
    )
    
    console.print("[bold cyan]Starting Aquila Optimizer...[/bold cyan]")
    
    # Run optimization
    with console.status("[bold cyan]Optimizing WKS parameters...[/bold cyan]", spinner="simpleDotsScrolling"):
        optimal_omega, fitness, history = wks_opt.optimize(X_train, y_train)
    
    console.print(f"[bold]Optimal Omega:[/bold] {optimal_omega:.4f}")
    console.print(f"[bold]Best Fitness:[/bold] {fitness:.4f}")
    
    # Extract features using optimal omega
    console.print("Extracting WKS features...")
    X_train_wks = wks_opt.extract_wks_features(X_train, omega=optimal_omega)
    X_val_wks = wks_opt.extract_wks_features(X_val, omega=optimal_omega)
    
    console.print(f"[bold]WKS features:[/bold] Train={X_train_wks.shape}, Val={X_val_wks.shape}")
    
    # Combine CNN + Statistical + WKS features
    # Order: [CNN Features, Statistical Features, WKS Features]
    X_train_combined = np.hstack([X_train_cnn, X_train_stat, X_train_wks])
    X_val_combined = np.hstack([X_val_cnn, X_val_stat, X_val_wks])
    
    console.print(f"[bold]Combined features:[/bold] Train={X_train_combined.shape}, Val={X_val_combined.shape}")
    
    # Convert for RNN [batch, 1, combined_features] 
    X_train_rnn = torch.tensor(X_train_combined, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_rnn = torch.tensor(X_val_combined, dtype=torch.float32).unsqueeze(1).to(device)
    
    # console.print(f"[dim]RNN Input: {X_train_rnn.shape}[/dim]")
    
    # =========================================================================
    # Step 5: ORNN Training (SIAO + Backprop)
    # =========================================================================
    console.print(Panel("[bold green]Step 5: ORNN Training (SIAO + Backprop)[/bold green]", box=box.DOUBLE))
    
    from stage3_models.ornn_model import ORNN, SIAOORNNTrainer, plot_ornn_training
    
    # Combined input size = CNN embedding + WKS features
    combined_input_size = X_train_combined.shape[1]
    
    ornn = ORNN(
        input_size=combined_input_size,
        hidden_size=rnn_hidden_size,
        num_layers=1,
        cell_type='gru'
    )
    
    trainer = SIAOORNNTrainer(
        ornn=ornn,
        output_size=num_classes,
        device=device,
        siao_pop_size=siao_pop_size,
        siao_max_iter=siao_max_iter,
        bp_epochs=100,
        bp_lr=0.001,
        weight_bounds=(-1.0, 1.0),
        fc_dropout=0.2,
        weight_decay=1e-5,
        patience=20
    )

    # -------------------------------------------------------------------------
    # Display Parameter Counts
    # -------------------------------------------------------------------------
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    cnn_params = count_parameters(cnn) # These might be 'trainable' in the model object, but we are treating them as fixed extractors here
    ornn_params = count_parameters(ornn)
    fc_params = count_parameters(trainer.fc)
    total_trainable = ornn_params + fc_params
    
    param_table = Table(title="Model Parameters", box=box.ROUNDED)
    param_table.add_column("Component", style="cyan")
    param_table.add_column("Parameters", justify="right")
    param_table.add_column("Status", style="yellow")
    
    param_table.add_row("CNN (Feature Extractor)", f"{cnn_params:,}", "Fixed/Frozen")
    param_table.add_row("ORNN (Recurrent Layer)", f"{ornn_params:,}", "[green]Trainable[/green]")
    param_table.add_row("Classifier Head (FC)", f"{fc_params:,}", "[green]Trainable[/green]")
    param_table.add_section()
    param_table.add_row("[bold]Total Trainable[/bold]", f"[bold]{total_trainable:,}[/bold]", "")
    
    console.print(param_table)
    
    # Calculate Class Weights
    if use_class_weights:
        class_counts = np.bincount(y_train, minlength=num_classes)
        total_samples = len(y_train)
        # Inverse frequency weights: total / (num_classes * count)
        class_weights = total_samples / (num_classes * (class_counts + 1))
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        console.print(f"[bold]Class counts:[/bold] {class_counts}")
        console.print(f"[bold]Class weights:[/bold] {class_weights}")
        
        # Update trainer to use class weights
        trainer.criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    else:
        console.print("[bold yellow]Using standard CrossEntropyLoss (no class weights)[/bold yellow]")
        trainer.criterion = nn.CrossEntropyLoss()
    
    console.print("[bold cyan]Starting Hybrid Training...[/bold cyan]")
    history = trainer.train(
        X_train_rnn, y_train,
        X_val_rnn, y_val,
        batch_size=batch_size
    )
    
    # =========================================================================
    # Step 6: Final Evaluation
    # =========================================================================
    console.print(Panel("[bold green]Step 6: Final Evaluation[/bold green]", box=box.DOUBLE))
    
    ornn.eval()
    trainer.fc.eval()
    
    with torch.no_grad():
        output, _ = ornn(X_train_rnn)
        last_hidden = output[:, -1, :]
        logits = trainer.fc(last_hidden)
        _, predicted_train = logits.max(1)
        
        output_val, _ = ornn(X_val_rnn)
        last_hidden_val = output_val[:, -1, :]
        logits_val = trainer.fc(last_hidden_val)
        _, predicted_val = logits_val.max(1)
        val_preds = predicted_val.cpu().numpy()
    
    # Calculate basic accuracy
    train_acc = (predicted_train.cpu().numpy() == y_train).mean()
    val_acc = (val_preds == y_val).mean()
    
    console.print(f"[bold]Final Train Accuracy:[/bold] [green]{train_acc:.4f}[/green]")
    console.print(f"[bold]Final Val Accuracy:[/bold]   [green]{val_acc:.4f}[/green]")
    
    # Detailed report
    from sklearn.metrics import classification_report
    report = classification_report(y_val, predicted_val.cpu().numpy(), output_dict=True)
    
    # Create Table
    table = Table(title="Classification Report", box=box.ROUNDED)
    table.add_column("Class", style="cyan", no_wrap=True)
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right")
    table.add_column("Support", justify="right")
    
    for cls_id in sorted([k for k in report.keys() if k.isdigit()]):
        metrics = report[cls_id]
        table.add_row(
            str(cls_id),
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1-score']:.3f}",
            str(metrics['support'])
        )
    
    # Add averages
    table.add_section()
    table.add_row(
        "Accuracy", "", "", f"{report['accuracy']:.3f}", str(report['macro avg']['support'])
    )
    
    console.print(table)
    
    console.print(Panel("[bold green]Training Complete![/bold green]", box=box.DOUBLE))
    console.print(f"Final Validation Accuracy: [bold green]{val_acc:.4f}[/bold green]")
    
    # Plot training history
    plot_ornn_training(history)
    
    # =========================================================================
    # Return Results
    # =========================================================================
    results = {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'history': history,
        'cnn_model': cnn,
        'ornn_trainer': trainer,
        'y_val': y_val,
        'val_preds': val_preds
    }
    
    return results


# =============================================================================
# Quick Start Function
# =============================================================================

def quick_start():
    """
    Quick start with default parameters.
    
    Usage in Colab:
        from train_pipeline import quick_start
        results = quick_start()
    """
    return run_complete_pipeline(
        data_dir='data/',
        window_size=50,
        stride=25,
        cnn_embedding_dim=256,
        wks_dim=43,
        rnn_hidden_size=128,
        num_classes=6,
        test_size=0.2,
        wks_pop_size=15,
        wks_max_iter=30,
        siao_pop_size=15,
        siao_max_iter=30,
        bp_epochs=50,
        batch_size=32
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("SIAO-CNN-ORNN Complete Training Pipeline")
    print("=" * 60)
    
    results = quick_start()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Validation Accuracy: {results['val_accuracy']:.4f}")
