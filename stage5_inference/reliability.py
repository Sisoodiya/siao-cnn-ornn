"""
Reliability Engineering Module for IP-200 Reactor Fault Detection

Computes reliability metrics based on fault predictions from SIAO-CNN-ORNN model.

Metrics:
1. Failure Rate: λ = n / [d * T + Σxi]
2. Mean Time To Failure: MTTF = 1 / λ
3. Reliability: R(t) = exp(-t / MTTF)

Features:
- Dynamic time-based computation
- Per-fault reliability tracking
- Reliability decay curves visualization
- Maintenance planning recommendations

Author: Reliability Engineering Specialist
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

FAULT_CLASSES = {
    0: {'name': 'Steady State', 'severity': 0, 'is_fault': False},
    1: {'name': 'Transient', 'severity': 1, 'is_fault': True},
    2: {'name': 'PORV', 'severity': 3, 'is_fault': True},
    3: {'name': 'SGTR', 'severity': 4, 'is_fault': True},
    4: {'name': 'FWLB', 'severity': 4, 'is_fault': True},
    5: {'name': 'RCP Failure', 'severity': 5, 'is_fault': True}
}

# Severity levels
SEVERITY_LEVELS = {
    0: 'Normal Operation',
    1: 'Minor - Monitor',
    2: 'Moderate - Plan Maintenance',
    3: 'Significant - Schedule Repair',
    4: 'Critical - Immediate Attention',
    5: 'Severe - Emergency Response'
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ReliabilityMetrics:
    """Container for reliability metrics."""
    failure_rate: float  # λ (failures per hour)
    mttf: float  # Mean Time To Failure (hours)
    reliability: float  # R(t) at given time
    availability: float  # System availability
    fault_count: int
    total_samples: int
    observation_time: float


@dataclass
class FaultStats:
    """Statistics for a specific fault type."""
    fault_id: int
    fault_name: str
    count: int
    proportion: float
    failure_rate: float
    mttf: float
    severity: int


# =============================================================================
# Reliability Calculator
# =============================================================================

class ReliabilityCalculator:
    """
    Computes reliability metrics from fault predictions.
    
    Uses:
    - Failure rate: λ = n / [d * T + Σxi]
    - MTTF: 1 / λ
    - Reliability: R(t) = exp(-t / MTTF)
    """
    
    def __init__(
        self,
        time_step_hours: float = 1.0,
        repair_rate: float = 0.1  # μ for availability calculation
    ):
        """
        Initialize calculator.
        
        Args:
            time_step_hours: Duration of each time step in hours
            repair_rate: Average repair rate (repairs per hour)
        """
        self.time_step_hours = time_step_hours
        self.repair_rate = repair_rate
        
        self.predictions = None
        self.timestamps = None
        self.metrics_history = []
    
    def fit(
        self,
        predictions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        observation_time: Optional[float] = None
    ) -> 'ReliabilityCalculator':
        """
        Fit calculator with prediction data.
        
        Args:
            predictions: Class predictions [n_samples]
            timestamps: Optional timestamps for each prediction
            observation_time: Total observation time in hours
        
        Returns:
            Self for method chaining
        """
        self.predictions = np.array(predictions)
        
        if timestamps is not None:
            self.timestamps = np.array(timestamps)
        else:
            # Create synthetic timestamps
            n = len(predictions)
            self.timestamps = np.arange(n) * self.time_step_hours
        
        if observation_time is None:
            self.observation_time = len(predictions) * self.time_step_hours
        else:
            self.observation_time = observation_time
        
        logger.info(f"Fitted with {len(predictions)} predictions over {self.observation_time:.1f} hours")
        
        return self
    
    def compute_failure_rate(
        self,
        predictions: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute failure rate λ.
        
        λ = n / [d * T + Σxi]
        
        Where:
        - n = number of failures
        - d = number of operating (non-failed) components
        - T = total observation time
        - xi = time to failure for each failed component
        
        Args:
            predictions: Optional prediction array (uses stored if None)
        
        Returns:
            Failure rate (failures per hour)
        """
        if predictions is None:
            predictions = self.predictions
        
        if predictions is None:
            raise ValueError("No predictions available. Call fit() first.")
        
        # Count faults (classes 1-5 are faults)
        fault_mask = predictions > 0
        n_faults = np.sum(fault_mask)
        n_normal = np.sum(~fault_mask)
        
        if n_faults == 0:
            return 0.0  # No failures
        
        # Calculate cumulative operating time
        # d * T: normal operation time contribution
        # Σxi: time to failure for each fault
        
        T = self.observation_time
        d = n_normal / len(predictions)  # Proportion of normal operation
        
        # Time to failure: sum of time indices where faults occurred
        fault_indices = np.where(fault_mask)[0]
        sum_xi = np.sum(fault_indices * self.time_step_hours)
        
        # Total exposure time
        total_exposure = d * T + sum_xi if sum_xi > 0 else T
        
        # Failure rate
        lambda_rate = n_faults / total_exposure if total_exposure > 0 else 0.0
        
        return lambda_rate
    
    def compute_mttf(self, failure_rate: Optional[float] = None) -> float:
        """
        Compute Mean Time To Failure.
        
        MTTF = 1 / λ
        
        Args:
            failure_rate: Optional failure rate (computed if None)
        
        Returns:
            MTTF in hours
        """
        if failure_rate is None:
            failure_rate = self.compute_failure_rate()
        
        if failure_rate <= 0:
            return float('inf')  # No failures = infinite MTTF
        
        return 1.0 / failure_rate
    
    def compute_reliability(
        self,
        t: float,
        mttf: Optional[float] = None
    ) -> float:
        """
        Compute reliability at time t.
        
        R(t) = exp(-t / MTTF)
        
        Args:
            t: Time in hours
            mttf: Optional MTTF (computed if None)
        
        Returns:
            Reliability (0 to 1)
        """
        if mttf is None:
            mttf = self.compute_mttf()
        
        if mttf == float('inf') or mttf <= 0:
            return 1.0  # Perfect reliability if no failures
        
        return np.exp(-t / mttf)
    
    def compute_availability(
        self,
        failure_rate: Optional[float] = None
    ) -> float:
        """
        Compute system availability.
        
        A = μ / (λ + μ)
        
        Where:
        - λ = failure rate
        - μ = repair rate
        
        Returns:
            Availability (0 to 1)
        """
        if failure_rate is None:
            failure_rate = self.compute_failure_rate()
        
        if failure_rate <= 0:
            return 1.0  # Perfect availability
        
        return self.repair_rate / (failure_rate + self.repair_rate)
    
    def compute_all_metrics(self) -> ReliabilityMetrics:
        """
        Compute all reliability metrics.
        
        Returns:
            ReliabilityMetrics object
        """
        if self.predictions is None:
            raise ValueError("No predictions. Call fit() first.")
        
        failure_rate = self.compute_failure_rate()
        mttf = self.compute_mttf(failure_rate)
        reliability = self.compute_reliability(self.observation_time, mttf)
        availability = self.compute_availability(failure_rate)
        
        fault_count = np.sum(self.predictions > 0)
        
        metrics = ReliabilityMetrics(
            failure_rate=failure_rate,
            mttf=mttf,
            reliability=reliability,
            availability=availability,
            fault_count=fault_count,
            total_samples=len(self.predictions),
            observation_time=self.observation_time
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def compute_per_fault_stats(self) -> List[FaultStats]:
        """
        Compute statistics for each fault type.
        
        Returns:
            List of FaultStats for each fault class
        """
        if self.predictions is None:
            raise ValueError("No predictions. Call fit() first.")
        
        stats = []
        total = len(self.predictions)
        
        for fault_id in range(6):
            count = np.sum(self.predictions == fault_id)
            proportion = count / total if total > 0 else 0
            
            fault_info = FAULT_CLASSES[fault_id]
            
            # Compute fault-specific failure rate
            if fault_info['is_fault'] and count > 0:
                failure_rate = count / self.observation_time
                mttf = 1.0 / failure_rate if failure_rate > 0 else float('inf')
            else:
                failure_rate = 0.0
                mttf = float('inf')
            
            stats.append(FaultStats(
                fault_id=fault_id,
                fault_name=fault_info['name'],
                count=count,
                proportion=proportion,
                failure_rate=failure_rate,
                mttf=mttf,
                severity=fault_info['severity']
            ))
        
        return stats
    
    def get_reliability_curve(
        self,
        t_max: Optional[float] = None,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate reliability decay curve.
        
        Args:
            t_max: Maximum time (uses observation_time if None)
            n_points: Number of points in curve
        
        Returns:
            Tuple of (time_array, reliability_array)
        """
        if t_max is None:
            t_max = self.observation_time * 2
        
        mttf = self.compute_mttf()
        
        t = np.linspace(0, t_max, n_points)
        r = np.array([self.compute_reliability(ti, mttf) for ti in t])
        
        return t, r
    
    def get_per_fault_reliability_curves(
        self,
        t_max: Optional[float] = None,
        n_points: int = 100
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate reliability curves for each fault type.
        
        Returns:
            Dict mapping fault name to (time, reliability) tuples
        """
        if t_max is None:
            t_max = self.observation_time * 2
        
        fault_stats = self.compute_per_fault_stats()
        curves = {}
        
        t = np.linspace(0, t_max, n_points)
        
        for stat in fault_stats:
            if stat.mttf != float('inf') and stat.mttf > 0:
                r = np.exp(-t / stat.mttf)
            else:
                r = np.ones_like(t)
            
            curves[stat.fault_name] = (t, r)
        
        return curves


# =============================================================================
# Maintenance Planner
# =============================================================================

class MaintenancePlanner:
    """
    Provides maintenance planning recommendations based on reliability metrics.
    """
    
    # Reliability thresholds
    THRESHOLDS = {
        'normal': 0.99,
        'warning': 0.95,
        'caution': 0.90,
        'critical': 0.80
    }
    
    def __init__(self, calculator: ReliabilityCalculator):
        self.calculator = calculator
    
    def get_maintenance_schedule(
        self,
        reliability_target: float = 0.90,
        max_interval_hours: float = 1000
    ) -> Dict:
        """
        Compute recommended maintenance intervals.
        
        Args:
            reliability_target: Target reliability level
            max_interval_hours: Maximum maintenance interval
        
        Returns:
            Maintenance schedule recommendations
        """
        mttf = self.calculator.compute_mttf()
        
        if mttf == float('inf'):
            interval = max_interval_hours
        else:
            # Solve R(t) = target for t
            # t = -MTTF * ln(target)
            interval = -mttf * np.log(reliability_target)
            interval = min(interval, max_interval_hours)
        
        return {
            'maintenance_interval_hours': interval,
            'maintenance_interval_days': interval / 24,
            'target_reliability': reliability_target,
            'current_mttf': mttf,
            'recommended_inspections_per_month': 720 / interval if interval > 0 else 0
        }
    
    def get_priority_ranking(self) -> List[Dict]:
        """
        Rank fault types by maintenance priority.
        
        Returns:
            List of faults sorted by priority (highest first)
        """
        fault_stats = self.calculator.compute_per_fault_stats()
        
        priorities = []
        for stat in fault_stats:
            if not FAULT_CLASSES[stat.fault_id]['is_fault']:
                continue
            
            # Priority score = severity * failure_rate
            priority_score = stat.severity * stat.failure_rate
            
            priorities.append({
                'fault_name': stat.fault_name,
                'severity': stat.severity,
                'severity_level': SEVERITY_LEVELS[stat.severity],
                'failure_rate': stat.failure_rate,
                'mttf_hours': stat.mttf,
                'priority_score': priority_score,
                'action': self._get_action(stat.severity, stat.failure_rate)
            })
        
        # Sort by priority score (descending)
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return priorities
    
    def _get_action(self, severity: int, failure_rate: float) -> str:
        """Get recommended action based on severity and failure rate."""
        if failure_rate > 0.1:
            return "IMMEDIATE: High failure rate requires urgent intervention"
        elif severity >= 4:
            return "CRITICAL: Schedule emergency maintenance within 24 hours"
        elif severity >= 3:
            return "HIGH: Schedule maintenance within 1 week"
        elif severity >= 2:
            return "MODERATE: Plan maintenance during next scheduled outage"
        else:
            return "LOW: Monitor and include in routine maintenance"
    
    def generate_report(self) -> str:
        """
        Generate comprehensive reliability report.
        
        Returns:
            Formatted report string
        """
        metrics = self.calculator.compute_all_metrics()
        fault_stats = self.calculator.compute_per_fault_stats()
        priorities = self.get_priority_ranking()
        schedule = self.get_maintenance_schedule()
        
        report = []
        report.append("=" * 70)
        report.append("RELIABILITY ENGINEERING REPORT - IP-200 REACTOR")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        report.append("\n1. SYSTEM RELIABILITY METRICS")
        report.append("-" * 40)
        report.append(f"   Total Observations:    {metrics.total_samples}")
        report.append(f"   Observation Period:    {metrics.observation_time:.1f} hours")
        report.append(f"   Fault Detections:      {metrics.fault_count}")
        report.append(f"   Failure Rate (λ):      {metrics.failure_rate:.6f} /hour")
        report.append(f"   MTTF:                  {metrics.mttf:.1f} hours")
        report.append(f"   Current Reliability:   {metrics.reliability:.4f}")
        report.append(f"   System Availability:   {metrics.availability:.4f}")
        
        report.append("\n2. FAULT DISTRIBUTION")
        report.append("-" * 40)
        for stat in fault_stats:
            if stat.count > 0:
                report.append(f"   {stat.fault_name:20s}: {stat.count:6d} ({stat.proportion*100:5.1f}%)")
        
        report.append("\n3. MAINTENANCE PRIORITIES")
        report.append("-" * 40)
        for i, p in enumerate(priorities[:5], 1):
            report.append(f"   {i}. {p['fault_name']}")
            report.append(f"      Severity: {p['severity_level']}")
            report.append(f"      MTTF: {p['mttf_hours']:.1f} hours")
            report.append(f"      Action: {p['action']}")
            report.append("")
        
        report.append("4. MAINTENANCE SCHEDULE")
        report.append("-" * 40)
        report.append(f"   Recommended Interval:  {schedule['maintenance_interval_hours']:.1f} hours")
        report.append(f"                          ({schedule['maintenance_interval_days']:.1f} days)")
        report.append(f"   Target Reliability:    {schedule['target_reliability']:.2f}")
        report.append(f"   Inspections/Month:     {schedule['recommended_inspections_per_month']:.1f}")
        
        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        return "\n".join(report)


# =============================================================================
# Visualization
# =============================================================================

def plot_reliability_curves(
    calculator: ReliabilityCalculator,
    save_path: Optional[str] = None
):
    """Plot reliability decay curves."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall reliability curve
        t, r = calculator.get_reliability_curve()
        axes[0].plot(t, r, 'b-', linewidth=2)
        axes[0].axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
        axes[0].axhline(y=0.8, color='orange', linestyle='--', label='80% threshold')
        axes[0].set_xlabel('Time (hours)', fontsize=12)
        axes[0].set_ylabel('Reliability R(t)', fontsize=12)
        axes[0].set_title('System Reliability Decay', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.05)
        
        # Per-fault curves
        curves = calculator.get_per_fault_reliability_curves()
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        
        for (name, (t, r)), color in zip(curves.items(), colors):
            if name != 'Steady State':
                axes[1].plot(t, r, label=name, linewidth=2, color=color)
        
        axes[1].set_xlabel('Time (hours)', fontsize=12)
        axes[1].set_ylabel('Reliability R(t)', fontsize=12)
        axes[1].set_title('Per-Fault Reliability Decay', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available")


def plot_fault_distribution(
    predictions: np.ndarray,
    save_path: Optional[str] = None
):
    """Plot fault distribution pie chart."""
    try:
        import matplotlib.pyplot as plt
        
        unique, counts = np.unique(predictions, return_counts=True)
        labels = [FAULT_CLASSES[i]['name'] for i in unique]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
        axes[0].pie(counts, labels=labels, autopct='%1.1f%%', colors=[colors[i] for i in unique])
        axes[0].set_title('Fault Distribution', fontsize=14)
        
        # Bar chart
        axes[1].bar(labels, counts, color=[colors[i] for i in unique])
        axes[1].set_xlabel('Fault Type', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Fault Counts', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available")


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_reliability_report(
    predictions: np.ndarray,
    time_step_hours: float = 1.0,
    print_report: bool = True
) -> Tuple[ReliabilityCalculator, str]:
    """
    Convenience function to compute reliability and generate report.
    
    Args:
        predictions: Model predictions [n_samples]
        time_step_hours: Time step duration in hours
        print_report: Whether to print the report
    
    Returns:
        Tuple of (calculator, report_string)
    """
    calculator = ReliabilityCalculator(time_step_hours=time_step_hours)
    calculator.fit(predictions)
    
    planner = MaintenancePlanner(calculator)
    report = planner.generate_report()
    
    if print_report:
        print(report)
    
    return calculator, report


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Reliability Engineering Module - Demo")
    print("=" * 60)
    
    # Simulate predictions (mix of normal and faults)
    np.random.seed(42)
    predictions = np.random.choice(
        [0, 0, 0, 0, 1, 2, 3, 4, 5],  # Weighted toward normal
        size=500
    )
    
    print(f"Simulated predictions: {len(predictions)} samples")
    
    # Compute reliability
    calculator, report = compute_reliability_report(predictions, time_step_hours=0.5)
    
    # Plot
    plot_reliability_curves(calculator)
    plot_fault_distribution(predictions)
