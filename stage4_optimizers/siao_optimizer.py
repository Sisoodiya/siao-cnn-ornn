"""
Self-Improved Aquila Optimizer (SIAO)

An enhanced version of the Aquila Optimizer with chaotic maps
for improved exploration-exploitation balance.

Improvements over standard AO:
1. Chaotic Quality Function using:
   - Logistic chaos map
   - Exponential decay
   - Gaussian-chaotic map
2. Self-improved search mechanism
3. RMSE-based objective function
4. Convergence tracking and stability analysis

Author: Optimization Algorithm Expert
"""

import logging
from typing import Tuple, Optional, List, Callable, Dict
import numpy as np
from scipy.special import gamma as scipy_gamma
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# =============================================================================
# Chaotic Maps
# =============================================================================

class ChaoticMaps:
    """
    Collection of chaotic maps for Quality Function enhancement.
    """
    
    @staticmethod
    def logistic_map(x: float, r: float = 4.0) -> float:
        """
        Logistic chaos map.
        
        x_{n+1} = r * x_n * (1 - x_n)
        
        Args:
            x: Current value in [0, 1]
            r: Control parameter (typically 4 for full chaos)
        
        Returns:
            Next chaotic value
        """
        return r * x * (1 - x)
    
    @staticmethod
    def logistic_sequence(length: int, x0: float = 0.7, r: float = 4.0) -> np.ndarray:
        """Generate sequence of logistic map values."""
        seq = np.zeros(length)
        seq[0] = x0
        for i in range(1, length):
            seq[i] = ChaoticMaps.logistic_map(seq[i-1], r)
        return seq
    
    @staticmethod
    def exponential_decay(t: int, t_max: int, alpha: float = 2.0) -> float:
        """
        Exponential decay function.
        
        D(t) = exp(-alpha * t / t_max)
        
        Args:
            t: Current iteration
            t_max: Maximum iterations
            alpha: Decay rate
        
        Returns:
            Decay factor
        """
        return np.exp(-alpha * t / t_max)
    
    @staticmethod
    def gaussian_chaotic(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        Gaussian-chaotic map (Eq. 17-19).
        
        Combines Gaussian distribution with chaotic perturbation.
        
        Args:
            x: Input value
            mu: Mean
            sigma: Standard deviation
        
        Returns:
            Gaussian-chaotic value
        """
        # Gaussian component
        gaussian = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        
        # Chaotic perturbation
        chaos = ChaoticMaps.logistic_map(np.abs(x) % 1)
        
        # Combined (Eq. 19)
        return gaussian * chaos + (1 - gaussian) * x
    
    @staticmethod
    def chaotic_quality_function(
        t: int,
        t_max: int,
        chaos_value: float,
        method: str = 'combined'
    ) -> float:
        """
        Enhanced Quality Function with chaotic maps.
        
        Args:
            t: Current iteration
            t_max: Maximum iterations
            chaos_value: Current chaotic sequence value
            method: 'logistic', 'exponential', 'gaussian', or 'combined'
        
        Returns:
            Quality function value
        """
        if method == 'logistic':
            # Logistic chaos-based QF
            return t ** (2 * chaos_value - 1) / (1 - t_max) ** 2
        
        elif method == 'exponential':
            # Exponential decay QF
            decay = ChaoticMaps.exponential_decay(t, t_max)
            return decay * (2 * np.random.random() - 1)
        
        elif method == 'gaussian':
            # Gaussian-chaotic QF
            gc = ChaoticMaps.gaussian_chaotic(chaos_value)
            return gc * t ** (2 * gc - 1)
        
        else:  # 'combined'
            # Combined strategy (Eq. 17-19)
            logistic = ChaoticMaps.logistic_map(chaos_value)
            decay = ChaoticMaps.exponential_decay(t, t_max)
            gaussian = ChaoticMaps.gaussian_chaotic(logistic)
            
            # Weighted combination
            w1, w2, w3 = 0.4, 0.3, 0.3
            combined = w1 * logistic + w2 * decay + w3 * gaussian
            
            return t ** (2 * combined - 1) / max((1 - t_max) ** 2, 1e-10)


# =============================================================================
# Self-Improved Aquila Optimizer (SIAO)
# =============================================================================

class SelfImprovedAquilaOptimizer:
    """
    Self-Improved Aquila Optimizer (SIAO).
    
    Enhanced AO with chaotic quality functions for:
    - Better exploration-exploitation balance
    - Faster convergence
    - Improved stability
    
    Hunting Strategies:
    1. Expanded Exploration (High soar with vertical stoop)
    2. Narrowed Exploration (Contour flight with Levy)
    3. Expanded Exploitation (Low flight with slow descent)
    4. Narrowed Exploitation (Walk and grab prey)
    """
    
    def __init__(
        self,
        objective_func: Callable,
        dim: int,
        lb: np.ndarray,
        ub: np.ndarray,
        pop_size: int = 30,
        max_iter: int = 100,
        chaos_method: str = 'combined',
        minimize: bool = True
    ):
        """
        Initialize SIAO.
        
        Args:
            objective_func: Function to optimize
            dim: Number of dimensions
            lb: Lower bounds array
            ub: Upper bounds array
            pop_size: Population size
            max_iter: Maximum iterations
            chaos_method: Chaotic map method
            minimize: True for minimization, False for maximization
        """
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.array(lb) if not isinstance(lb, np.ndarray) else lb
        self.ub = np.array(ub) if not isinstance(ub, np.ndarray) else ub
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.chaos_method = chaos_method
        self.minimize = minimize
        
        # Population
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = np.inf if minimize else -np.inf
        
        # Chaotic sequence
        self.chaos_seq = ChaoticMaps.logistic_sequence(max_iter + 1)
        
        # Convergence tracking
        self.history = []
        self.stability_history = []
        
        logger.info(f"SIAO: dim={dim}, pop={pop_size}, iter={max_iter}, chaos={chaos_method}")
    
    def _initialize_population(self) -> None:
        """Initialize population with chaos-enhanced distribution."""
        self.population = np.zeros((self.pop_size, self.dim))
        
        for i in range(self.pop_size):
            # Use chaotic initialization
            chaos_val = ChaoticMaps.logistic_sequence(self.dim, x0=0.1 + 0.8 * i / self.pop_size)
            self.population[i] = self.lb + chaos_val * (self.ub - self.lb)
        
        self.fitness = np.zeros(self.pop_size)
        
        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = self._evaluate(self.population[i])
            
            if self._is_better(self.fitness[i], self.best_fitness):
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()
    
    def _evaluate(self, x: np.ndarray) -> float:
        """Evaluate objective function."""
        return self.objective_func(x)
    
    def _is_better(self, new_fitness: float, old_fitness: float) -> bool:
        """Check if new fitness is better."""
        if self.minimize:
            return new_fitness < old_fitness
        return new_fitness > old_fitness
    
    def _levy_flight(self) -> np.ndarray:
        """Generate Levy flight step with chaotic enhancement."""
        beta = 1.5
        sigma_u = (
            scipy_gamma(1 + beta) * np.sin(np.pi * beta / 2) /
            (scipy_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        
        step = u / (np.abs(v) ** (1 / beta))
        return step
    
    def _get_quality_function(self, t: int) -> float:
        """Get chaotic quality function value."""
        chaos_val = self.chaos_seq[t % len(self.chaos_seq)]
        return ChaoticMaps.chaotic_quality_function(
            t, self.max_iter, chaos_val, self.chaos_method
        )
    
    def _expanded_exploration(
        self,
        X: np.ndarray,
        X_best: np.ndarray,
        X_mean: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Strategy 1: Expanded Exploration - High soar with vertical stoop.
        
        X_new = X_best * (1 - t/T) + (X_mean - X_best) * rand * chaos
        """
        chaos_val = self.chaos_seq[t % len(self.chaos_seq)]
        X_new = X_best * (1 - t / self.max_iter) + \
                (X_mean - X_best) * np.random.random() * chaos_val
        return self._bound_check(X_new)
    
    def _narrowed_exploration(
        self,
        X: np.ndarray,
        X_best: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Strategy 2: Narrowed Exploration - Contour flight with Levy.
        
        Uses chaotic Levy flight for fine exploration.
        """
        levy = self._levy_flight()
        chaos_val = self.chaos_seq[t % len(self.chaos_seq)]
        
        y = X_best - X + levy * chaos_val
        X_new = X_best + np.random.random() * y
        return self._bound_check(X_new)
    
    def _expanded_exploitation(
        self,
        X: np.ndarray,
        X_best: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Strategy 3: Expanded Exploitation - Low flight with slow descent.
        
        Uses chaotic spiral for exploitation.
        """
        QF = self._get_quality_function(t)
        
        # Spiral parameters with chaos
        theta = np.random.uniform(-np.pi, np.pi)
        chaos_val = self.chaos_seq[t % len(self.chaos_seq)]
        r = chaos_val * (self.ub - self.lb) * (1 - t / self.max_iter)
        
        X_new = X_best - (X - X_best) * 0.1 * QF - r * np.cos(theta)
        return self._bound_check(X_new)
    
    def _narrowed_exploitation(
        self,
        X: np.ndarray,
        X_best: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Strategy 4: Narrowed Exploitation - Walk and grab prey.
        
        Fine-tuned search around best solution.
        """
        QF = self._get_quality_function(t)
        chaos_val = self.chaos_seq[t % len(self.chaos_seq)]
        
        G1 = 2 * chaos_val - 1
        G2 = 2 * (1 - t / self.max_iter)
        
        X_new = QF * X_best - (G1 * X * np.random.random()) - \
                G2 * np.random.random() * (X - X_best)
        return self._bound_check(X_new)
    
    def _self_improvement(
        self,
        X: np.ndarray,
        X_new: np.ndarray,
        X_best: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Self-improvement mechanism.
        
        Combines current and new solutions with adaptive weights.
        """
        # Adaptive weight based on progress
        w = 1 - t / self.max_iter
        chaos_val = self.chaos_seq[t % len(self.chaos_seq)]
        
        # Self-improved solution
        X_improved = w * X + (1 - w) * X_new + \
                     0.1 * chaos_val * (X_best - X_new)
        
        return self._bound_check(X_improved)
    
    def _bound_check(self, X: np.ndarray) -> np.ndarray:
        """Ensure solution is within bounds."""
        return np.clip(X, self.lb, self.ub)
    
    def _compute_stability(self) -> float:
        """Compute population stability metric."""
        if len(self.history) < 5:
            return 1.0
        
        recent = self.history[-5:]
        std = np.std(recent)
        mean = np.abs(np.mean(recent))
        
        return std / (mean + 1e-10)
    
    def optimize(self) -> Tuple[np.ndarray, float, Dict]:
        """
        Run SIAO optimization.
        
        Returns:
            Tuple of (best_solution, best_fitness, info_dict)
        """
        logger.info("=" * 60)
        logger.info("Starting Self-Improved Aquila Optimizer (SIAO)")
        logger.info("=" * 60)
        
        # Initialize
        self._initialize_population()
        self.history.append(self.best_fitness)
        
        for t in range(1, self.max_iter + 1):
            X_mean = np.mean(self.population, axis=0)
            
            for i in range(self.pop_size):
                X = self.population[i]
                
                # Strategy selection based on progress and randomness
                r = np.random.random()
                t_ratio = t / self.max_iter
                
                if t_ratio <= 2/3:
                    # Exploration phase
                    if r < 0.5:
                        X_new = self._expanded_exploration(X, self.best_solution, X_mean, t)
                    else:
                        X_new = self._narrowed_exploration(X, self.best_solution, t)
                else:
                    # Exploitation phase
                    if r < 0.5:
                        X_new = self._expanded_exploitation(X, self.best_solution, t)
                    else:
                        X_new = self._narrowed_exploitation(X, self.best_solution, t)
                
                # Apply self-improvement
                X_improved = self._self_improvement(X, X_new, self.best_solution, t)
                
                # Evaluate
                fitness_improved = self._evaluate(X_improved)
                fitness_new = self._evaluate(X_new)
                
                # Select best between improved and new
                if self._is_better(fitness_improved, fitness_new):
                    final_X = X_improved
                    final_fitness = fitness_improved
                else:
                    final_X = X_new
                    final_fitness = fitness_new
                
                # Update if better than current
                if self._is_better(final_fitness, self.fitness[i]):
                    self.population[i] = final_X
                    self.fitness[i] = final_fitness
                    
                    if self._is_better(final_fitness, self.best_fitness):
                        self.best_fitness = final_fitness
                        self.best_solution = final_X.copy()
            
            # Track convergence
            self.history.append(self.best_fitness)
            self.stability_history.append(self._compute_stability())
            
            if t % 10 == 0 or t == 1:
                logger.info(f"Iter {t}/{self.max_iter}: Best = {self.best_fitness:.6f}, "
                           f"Stability = {self.stability_history[-1]:.4f}")
        
        # Compile results
        info = {
            'history': np.array(self.history),
            'stability': np.array(self.stability_history),
            'final_population': self.population.copy(),
            'final_fitness': self.fitness.copy()
        }
        
        logger.info("=" * 60)
        logger.info("SIAO Optimization Complete!")
        logger.info(f"  Best Solution: {self.best_solution}")
        logger.info(f"  Best Fitness: {self.best_fitness:.6f}")
        logger.info("=" * 60)
        
        return self.best_solution, self.best_fitness, info


# =============================================================================
# RMSE Objective Function
# =============================================================================

def create_rmse_objective(y_true: np.ndarray, model_func: Callable) -> Callable:
    """
    Create RMSE objective function for optimization.
    
    Args:
        y_true: Ground truth values
        model_func: Function that takes parameters and returns predictions
    
    Returns:
        Objective function that computes RMSE
    """
    def rmse_objective(params: np.ndarray) -> float:
        y_pred = model_func(params)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse
    
    return rmse_objective


# =============================================================================
# Visualization
# =============================================================================

def plot_siao_convergence(
    history: np.ndarray,
    stability: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """Plot SIAO convergence and stability."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2 if stability is not None else 1, figsize=(14, 5))
        
        if stability is None:
            axes = [axes]
        
        # Convergence plot
        axes[0].plot(history, 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('RMSE / Fitness', fontsize=12)
        axes[0].set_title('SIAO Convergence', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Stability plot
        if stability is not None:
            axes[1].plot(stability, 'r-', linewidth=2)
            axes[1].set_xlabel('Iteration', fontsize=12)
            axes[1].set_ylabel('Stability Index', fontsize=12)
            axes[1].set_title('Population Stability', fontsize=14)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")


# =============================================================================
# Benchmark Functions
# =============================================================================

class BenchmarkFunctions:
    """
    Standard benchmark functions for optimization testing.
    """
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Sphere function (unimodal): f(x) = sum(x^2)"""
        return np.sum(x ** 2)
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin function (multimodal)"""
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock function (valley-shaped)"""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Ackley function (multimodal)"""
        d = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e
    
    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """Griewank function (multimodal)"""
        sum_sq = np.sum(x ** 2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_sq - prod_cos + 1


def run_siao_demo(
    func_name: str = 'sphere',
    dim: int = 10,
    pop_size: int = 30,
    max_iter: int = 100,
    show_plot: bool = True
) -> Tuple[np.ndarray, float, Dict]:
    """
    Run SIAO optimization demo with a benchmark function.
    
    Args:
        func_name: 'sphere', 'rastrigin', 'rosenbrock', 'ackley', 'griewank'
        dim: Number of dimensions
        pop_size: Population size
        max_iter: Maximum iterations
        show_plot: Whether to show convergence plot
    
    Returns:
        Tuple of (best_solution, best_fitness, info_dict)
    """
    # Get benchmark function
    functions = {
        'sphere': (BenchmarkFunctions.sphere, -10, 10),
        'rastrigin': (BenchmarkFunctions.rastrigin, -5.12, 5.12),
        'rosenbrock': (BenchmarkFunctions.rosenbrock, -5, 10),
        'ackley': (BenchmarkFunctions.ackley, -32.768, 32.768),
        'griewank': (BenchmarkFunctions.griewank, -600, 600),
    }
    
    if func_name not in functions:
        raise ValueError(f"Unknown function: {func_name}. Choose from {list(functions.keys())}")
    
    func, lb_val, ub_val = functions[func_name]
    lb = lb_val * np.ones(dim)
    ub = ub_val * np.ones(dim)
    
    print(f"Running SIAO on {func_name} function (dim={dim})")
    print("=" * 60)
    
    # Create and run SIAO
    siao = SelfImprovedAquilaOptimizer(
        objective_func=func,
        dim=dim,
        lb=lb,
        ub=ub,
        pop_size=pop_size,
        max_iter=max_iter,
        chaos_method='combined',
        minimize=True
    )
    
    best_solution, best_fitness, info = siao.optimize()
    
    print(f"\nResults:")
    print(f"  Best Fitness: {best_fitness:.6e}")
    print(f"  Best Solution (first 5): {best_solution[:min(5, dim)]}")
    
    if show_plot:
        plot_siao_convergence(info['history'], info['stability'])
    
    return best_solution, best_fitness, info


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Self-Improved Aquila Optimizer (SIAO) - Demo")
    print("=" * 60)
    
    # Run demo with Sphere function
    best_solution, best_fitness, info = run_siao_demo(
        func_name='sphere',
        dim=10,
        pop_size=30,
        max_iter=100,
        show_plot=True
    )
