import numpy as np
import logging
from collections import deque
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ForkliftRouteOptimizer:
    def __init__(self, route_min: int = 1, route_max: int = 10, tau: float = 2.0, memory: int = 10):
        """
        Contextual Thompson Sampling for Forklift Route Optimization in Warehouses
        
        Args:
            route_min: Minimum route ID (e.g., shortest)
            route_max: Maximum route ID (e.g., longest or most complex)
            tau: Softmax temperature (higher = more exploration)
            memory: Number of past interactions to remember per context
        """
        self.validate_init_params(route_min, route_max, tau, memory)
        self.route_min = route_min
        self.route_max = route_max
        self.tau = tau
        self.memory = memory

        # Contextual memory of performance
        self.route_history: Dict[Tuple[Any, Any], deque] = {}
        self.beta_params: Dict[Tuple[Any, Any], Dict[int, Tuple[int, int]]] = {}

    @staticmethod
    def validate_init_params(route_min: int, route_max: int, tau: float, memory: int) -> None:
        if route_min >= route_max:
            raise ValueError("route_min must be less than route_max")
        if tau <= 0:
            raise ValueError("tau must be positive")
        if memory < 1:
            raise ValueError("memory must be at least 1")

    def softmax(self, values: np.ndarray) -> np.ndarray:
        scaled = values / self.tau
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        return probs / np.sum(probs)

    def select_route(self, zone_id: Any, task_type: Any) -> int:
        """
        Select a route using contextual Thompson Sampling and softmax exploration.
        """
        context = (zone_id, task_type)

        if context not in self.route_history:
            self.route_history[context] = deque(maxlen=self.memory)
            self.beta_params[context] = {route: [1, 1] for route in range(self.route_min, self.route_max + 1)}

        samples = {route: np.random.beta(a, b) for route, (a, b) in self.beta_params[context].items()}
        values = np.array(list(samples.values()))
        keys = np.array(list(samples.keys()))
        probs = self.softmax(values)

        return int(np.random.choice(keys, p=probs))

    def update(self, zone_id: Any, task_type: Any, route: int, success: int) -> None:
        """
        Update route feedback based on success (1) or failure/delay (0).
        """
        context = (zone_id, task_type)

        if context not in self.route_history:
            self.route_history[context] = deque(maxlen=self.memory)
            self.beta_params[context] = {r: [1, 1] for r in range(self.route_min, self.route_max + 1)}

        self.route_history[context].append((route, success))
        self._recompute_betas(context)

    def _recompute_betas(self, context: Tuple[Any, Any]) -> None:
        counts = {r: [1, 1] for r in range(self.route_min, self.route_max + 1)}

        for route, success in self.route_history[context]:
            counts[route][0] += success      # alpha (wins)
            counts[route][1] += 1 - success  # beta (failures)

        self.beta_params[context] = counts

    def get_route_stats(self, zone_id: Any, task_type: Any) -> Dict[int, list]:
        context = (zone_id, task_type)
        return self.beta_params.get(context, {})
