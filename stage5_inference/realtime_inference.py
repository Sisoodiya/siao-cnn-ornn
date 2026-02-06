"""
Real-Time Inference Module for SIAO-CNN-ORNN

Enables real-time fault detection from streaming sensor data.

Features:
- Sliding window buffer for streaming data
- Real-time predictions with low latency
- Alert system for fault detection
- Model checkpoint loading
- Thread-safe design for production

Author: Deployment Specialist
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Dict, Optional, Callable, List, Tuple, Any
import threading
import time
import json
import pickle
from datetime import datetime
from dataclasses import dataclass, field
import logging
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# =============================================================================
# Constants
# =============================================================================

FAULT_CLASSES = {
    0: {'name': 'Steady State', 'severity': 0, 'alert': False, 'color': 'green'},
    1: {'name': 'Transient', 'severity': 1, 'alert': True, 'color': 'yellow'},
    2: {'name': 'PORV', 'severity': 3, 'alert': True, 'color': 'orange'},
    3: {'name': 'SGTR', 'severity': 4, 'alert': True, 'color': 'red'},
    4: {'name': 'FWLB', 'severity': 4, 'alert': True, 'color': 'red'},
    5: {'name': 'RCP Failure', 'severity': 5, 'alert': True, 'color': 'darkred'}
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Prediction:
    """Container for a single prediction."""
    timestamp: datetime
    class_id: int
    class_name: str
    confidence: float
    probabilities: np.ndarray
    is_fault: bool
    severity: int
    latency_ms: float


@dataclass
class Alert:
    """Alert for fault detection."""
    timestamp: datetime
    fault_type: str
    severity: int
    confidence: float
    message: str
    action: str


@dataclass
class SensorReading:
    """Single sensor reading from the reactor."""
    timestamp: datetime
    values: np.ndarray  # [n_features]


# =============================================================================
# Normalization Statistics
# =============================================================================

@dataclass
class NormalizationStats:
    """Statistics for Z-score normalization."""
    mean: np.ndarray
    std: np.ndarray
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize data using stored statistics."""
        return (x - self.mean) / (self.std + 1e-8)
    
    def save(self, path: str):
        """Save stats to file."""
        np.savez(path, mean=self.mean, std=self.std)
        logger.info(f"Normalization stats saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'NormalizationStats':
        """Load stats from file."""
        data = np.load(path)
        return cls(mean=data['mean'], std=data['std'])
    
    @classmethod
    def from_data(cls, X: np.ndarray) -> 'NormalizationStats':
        """Compute stats from training data."""
        # Flatten time dimension
        X_flat = X.reshape(-1, X.shape[-1])
        return cls(mean=X_flat.mean(axis=0), std=X_flat.std(axis=0))


# =============================================================================
# Sliding Window Buffer
# =============================================================================

class SlidingWindowBuffer:
    """
    Thread-safe sliding window buffer for streaming data.
    
    Maintains a fixed-size window of recent sensor readings.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        n_features: int = 43
    ):
        self.window_size = window_size
        self.n_features = n_features
        
        self.buffer = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
        logger.info(f"Buffer initialized: window={window_size}, features={n_features}")
    
    def add(self, reading: np.ndarray):
        """Add a sensor reading to the buffer."""
        with self.lock:
            self.buffer.append(reading)
    
    def add_batch(self, readings: np.ndarray):
        """Add multiple readings at once."""
        with self.lock:
            for r in readings:
                self.buffer.append(r)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough data for prediction."""
        return len(self.buffer) >= self.window_size
    
    def get_window(self) -> Optional[np.ndarray]:
        """Get current window as numpy array."""
        with self.lock:
            if not self.is_ready():
                return None
            return np.array(list(self.buffer))
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Real-Time Predictor
# =============================================================================

class RealTimePredictor:
    """
    Real-time fault detection predictor.
    
    Processes streaming sensor data and outputs predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        window_size: int = 50,
        n_features: int = 43,
        device: Optional[torch.device] = None,
        norm_stats: Optional[NormalizationStats] = None,
        confidence_threshold: float = 0.5,
        alert_callback: Optional[Callable[[Alert], None]] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained SIAO-CNN-ORNN model
            window_size: Size of sliding window
            n_features: Number of sensor features
            device: GPU/CPU device
            norm_stats: Normalization statistics from training
            confidence_threshold: Minimum confidence for alerts
            alert_callback: Function to call when alert is triggered
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        
        self.window_size = window_size
        self.n_features = n_features
        self.norm_stats = norm_stats
        self.confidence_threshold = confidence_threshold
        self.alert_callback = alert_callback
        
        # Buffer
        self.buffer = SlidingWindowBuffer(window_size, n_features)
        
        # History
        self.prediction_history: List[Prediction] = []
        self.alert_history: List[Alert] = []
        
        # Running state
        self.is_running = False
        self._lock = threading.Lock()
        
        logger.info(f"Predictor initialized on {self.device}")
    
    def preprocess(self, window: np.ndarray) -> torch.Tensor:
        """Preprocess window for model input."""
        # Normalize if stats available
        if self.norm_stats:
            window = self.norm_stats.normalize(window)
        
        # Convert to tensor
        x = torch.tensor(window, dtype=torch.float32, device=self.device)
        
        # Add batch dimension
        x = x.unsqueeze(0)  # [1, window_size, n_features]
        
        return x
    
    def predict(self, window: np.ndarray) -> Prediction:
        """
        Make prediction on a window of sensor data.
        
        Args:
            window: Sensor window [window_size, n_features]
        
        Returns:
            Prediction object
        """
        start_time = time.time()
        
        # Preprocess
        x = self.preprocess(window)
        
        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            class_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, class_id].item()
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Create prediction
        fault_info = FAULT_CLASSES[class_id]
        prediction = Prediction(
            timestamp=datetime.now(),
            class_id=class_id,
            class_name=fault_info['name'],
            confidence=confidence,
            probabilities=probs.cpu().numpy().flatten(),
            is_fault=class_id > 0,
            severity=fault_info['severity'],
            latency_ms=latency_ms
        )
        
        return prediction
    
    def process_reading(self, reading: np.ndarray) -> Optional[Prediction]:
        """
        Process a single sensor reading.
        
        Args:
            reading: Sensor values [n_features]
        
        Returns:
            Prediction if buffer is ready, None otherwise
        """
        self.buffer.add(reading)
        
        if not self.buffer.is_ready():
            return None
        
        window = self.buffer.get_window()
        prediction = self.predict(window)
        
        with self._lock:
            self.prediction_history.append(prediction)
        
        # Check for alerts
        if prediction.is_fault and prediction.confidence >= self.confidence_threshold:
            self._trigger_alert(prediction)
        
        return prediction
    
    def _trigger_alert(self, prediction: Prediction):
        """Trigger alert for fault detection."""
        fault_info = FAULT_CLASSES[prediction.class_id]
        
        # Determine action based on severity
        actions = {
            1: "Continue monitoring, log event",
            2: "Notify operator, increase monitoring frequency",
            3: "Alert operator, prepare for potential intervention",
            4: "IMMEDIATE operator action required, initiate safety protocols",
            5: "EMERGENCY: Execute emergency shutdown procedures"
        }
        
        alert = Alert(
            timestamp=prediction.timestamp,
            fault_type=prediction.class_name,
            severity=prediction.severity,
            confidence=prediction.confidence,
            message=f"FAULT DETECTED: {prediction.class_name} "
                   f"(Confidence: {prediction.confidence:.1%})",
            action=actions.get(prediction.severity, "Monitor")
        )
        
        with self._lock:
            self.alert_history.append(alert)
        
        logger.warning(f"⚠️ ALERT: {alert.message}")
        logger.warning(f"   Action: {alert.action}")
        
        if self.alert_callback:
            self.alert_callback(alert)
    
    def get_recent_predictions(self, n: int = 10) -> List[Prediction]:
        """Get n most recent predictions."""
        with self._lock:
            return list(self.prediction_history[-n:])
    
    def get_statistics(self) -> Dict:
        """Get prediction statistics."""
        with self._lock:
            if not self.prediction_history:
                return {}
            
            latencies = [p.latency_ms for p in self.prediction_history]
            fault_count = sum(1 for p in self.prediction_history if p.is_fault)
            
            return {
                'total_predictions': len(self.prediction_history),
                'fault_count': fault_count,
                'fault_rate': fault_count / len(self.prediction_history),
                'avg_latency_ms': np.mean(latencies),
                'max_latency_ms': np.max(latencies),
                'total_alerts': len(self.alert_history)
            }
    
    def reset(self):
        """Reset predictor state."""
        self.buffer.clear()
        with self._lock:
            self.prediction_history.clear()
            self.alert_history.clear()
        logger.info("Predictor reset")


# =============================================================================
# Streaming Inference Engine
# =============================================================================

class StreamingInferenceEngine:
    """
    Production-ready streaming inference engine.
    
    Handles:
    - Continuous data ingestion
    - Real-time predictions
    - Alert management
    - Performance monitoring
    """
    
    def __init__(
        self,
        predictor: RealTimePredictor,
        prediction_interval: float = 0.1  # seconds
    ):
        self.predictor = predictor
        self.prediction_interval = prediction_interval
        
        self.is_running = False
        self._thread = None
        self._stop_event = threading.Event()
        
        # Callbacks
        self.on_prediction: Optional[Callable[[Prediction], None]] = None
        self.on_alert: Optional[Callable[[Alert], None]] = None
    
    def start(self):
        """Start the inference engine."""
        if self.is_running:
            logger.warning("Engine already running")
            return
        
        self.is_running = True
        self._stop_event.clear()
        logger.info("Streaming inference engine started")
    
    def stop(self):
        """Stop the inference engine."""
        self.is_running = False
        self._stop_event.set()
        logger.info("Streaming inference engine stopped")
    
    def ingest_reading(self, reading: np.ndarray) -> Optional[Prediction]:
        """
        Ingest a sensor reading and return prediction if available.
        
        Args:
            reading: Sensor values [n_features]
        
        Returns:
            Prediction or None
        """
        if not self.is_running:
            return None
        
        prediction = self.predictor.process_reading(reading)
        
        if prediction and self.on_prediction:
            self.on_prediction(prediction)
        
        return prediction
    
    def ingest_batch(self, readings: np.ndarray) -> List[Prediction]:
        """
        Ingest batch of sensor readings.
        
        Args:
            readings: [n_readings, n_features]
        
        Returns:
            List of predictions
        """
        predictions = []
        
        for reading in readings:
            pred = self.ingest_reading(reading)
            if pred:
                predictions.append(pred)
        
        return predictions
    
    def simulate_realtime(
        self,
        data: np.ndarray,
        delay: float = 0.01  # seconds between readings
    ) -> List[Prediction]:
        """
        Simulate real-time inference on historical data.
        
        Args:
            data: Historical sensor data [n_samples, n_features]
            delay: Delay between simulated readings
        
        Returns:
            All predictions made
        """
        logger.info(f"Simulating real-time on {len(data)} readings...")
        
        self.start()
        predictions = []
        
        for i, reading in enumerate(data):
            pred = self.ingest_reading(reading)
            if pred:
                predictions.append(pred)
            
            if delay > 0:
                time.sleep(delay)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{len(data)} readings")
        
        self.stop()
        
        return predictions


# =============================================================================
# Model Checkpoint Utility
# =============================================================================

class ModelCheckpoint:
    """Utility for saving and loading model checkpoints."""
    
    @staticmethod
    def save(
        model: nn.Module,
        path: str,
        norm_stats: Optional[NormalizationStats] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            model: Trained model
            path: Save path
            norm_stats: Normalization statistics
            metadata: Additional metadata
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        if norm_stats:
            checkpoint['norm_mean'] = norm_stats.mean
            checkpoint['norm_std'] = norm_stats.std
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")
    
    @staticmethod
    def load(
        path: str,
        model_class: Any,
        device: torch.device,
        **model_kwargs
    ) -> Tuple[nn.Module, Optional[NormalizationStats], Dict]:
        """
        Load model checkpoint.
        
        Args:
            path: Checkpoint path
            model_class: Model class to instantiate
            device: Device to load to
            **model_kwargs: Model initialization arguments
        
        Returns:
            Tuple of (model, norm_stats, metadata)
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        norm_stats = None
        if 'norm_mean' in checkpoint:
            norm_stats = NormalizationStats(
                mean=checkpoint['norm_mean'],
                std=checkpoint['norm_std']
            )
        
        metadata = checkpoint.get('metadata', {})
        
        logger.info(f"Model loaded from {path}")
        
        return model, norm_stats, metadata


# =============================================================================
# Visualization
# =============================================================================

def plot_realtime_dashboard(predictions: List[Prediction], save_path: Optional[str] = None):
    """Plot real-time monitoring dashboard."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Timeline
        times = [i for i in range(len(predictions))]
        classes = [p.class_id for p in predictions]
        confidences = [p.confidence for p in predictions]
        latencies = [p.latency_ms for p in predictions]
        
        # Class predictions over time
        axes[0, 0].scatter(times, classes, c=classes, cmap='RdYlGn_r', alpha=0.6)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Predicted Class')
        axes[0, 0].set_title('Fault Detection Timeline')
        axes[0, 0].set_yticks(range(6))
        axes[0, 0].set_yticklabels([FAULT_CLASSES[i]['name'][:10] for i in range(6)])
        
        # Confidence
        axes[0, 1].plot(times, confidences, 'b-', alpha=0.7)
        axes[0, 1].axhline(y=0.9, color='g', linestyle='--', label='High Conf')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].set_title('Prediction Confidence')
        axes[0, 1].legend()
        
        # Latency
        axes[1, 0].plot(times, latencies, 'r-', alpha=0.7)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].set_title('Inference Latency')
        
        # Class distribution
        class_counts = [sum(1 for p in predictions if p.class_id == i) for i in range(6)]
        colors = [FAULT_CLASSES[i]['color'] for i in range(6)]
        axes[1, 1].bar(range(6), class_counts, color=colors)
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Class Distribution')
        axes[1, 1].set_xticks(range(6))
        axes[1, 1].set_xticklabels([FAULT_CLASSES[i]['name'][:10] for i in range(6)], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available")


# =============================================================================
# Quick Start Functions
# =============================================================================

def create_realtime_predictor(
    model: nn.Module,
    X_train: np.ndarray,
    window_size: int = 50
) -> RealTimePredictor:
    """
    Create real-time predictor from trained model.
    
    Args:
        model: Trained model
        X_train: Training data for normalization stats
        window_size: Sliding window size
    
    Returns:
        Configured RealTimePredictor
    """
    # Compute normalization stats
    norm_stats = NormalizationStats.from_data(X_train)
    
    # Create predictor
    predictor = RealTimePredictor(
        model=model,
        window_size=window_size,
        n_features=X_train.shape[-1],
        norm_stats=norm_stats,
        confidence_threshold=0.7
    )
    
    return predictor


def simulate_sensor_stream(
    model: nn.Module,
    test_data: np.ndarray,
    X_train: np.ndarray,
    window_size: int = 50
) -> Tuple[List[Prediction], Dict]:
    """
    Simulate real-time sensor stream processing.
    
    Args:
        model: Trained model
        test_data: Test sensor data [n_samples, n_features]
        X_train: Training data for normalization
        window_size: Window size
    
    Returns:
        Tuple of (predictions, statistics)
    """
    # Create predictor
    predictor = create_realtime_predictor(model, X_train, window_size)
    
    # Create engine
    engine = StreamingInferenceEngine(predictor)
    
    # Simulate
    predictions = engine.simulate_realtime(test_data, delay=0)
    
    # Get stats
    stats = predictor.get_statistics()
    
    logger.info(f"Simulation complete: {stats['total_predictions']} predictions, "
               f"{stats['fault_count']} faults detected")
    
    return predictions, stats


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Real-Time Inference Module - Demo")
    print("=" * 60)
    
    # Create dummy model (for demo)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(43 * 50, 6)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = DummyModel()
    
    # Simulate sensor data
    n_readings = 200
    sensor_data = np.random.randn(n_readings, 43).astype(np.float32)
    train_data = np.random.randn(100, 50, 43).astype(np.float32)
    
    # Create predictor
    predictor = create_realtime_predictor(model, train_data, window_size=50)
    
    # Process readings
    print(f"Processing {n_readings} sensor readings...")
    
    predictions = []
    for i, reading in enumerate(sensor_data):
        pred = predictor.process_reading(reading)
        if pred:
            predictions.append(pred)
            if (i + 1) % 50 == 0:
                print(f"  Step {i+1}: {pred.class_name} ({pred.confidence:.1%})")
    
    # Statistics
    stats = predictor.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Faults detected: {stats['fault_count']}")
    print(f"  Avg latency: {stats['avg_latency_ms']:.2f} ms")
    
    # Plot
    if predictions:
        plot_realtime_dashboard(predictions)
