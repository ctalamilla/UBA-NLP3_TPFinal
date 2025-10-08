# fastapi_app/performance.py
"""
Helper para métricas de performance operacionales
"""
import time
from typing import Dict, Any, Optional
from contextlib import contextmanager


# Precios OpenAI (actualizar según modelo)
# https://openai.com/pricing
PRICING = {
    "gpt-4o": {
        "input": 2.50 / 1_000_000,   # $2.50 por 1M tokens
        "output": 10.00 / 1_000_000,  # $10.00 por 1M tokens
    },
    "gpt-4o-mini": {
        "input": 0.150 / 1_000_000,   # $0.15 por 1M tokens
        "output": 0.600 / 1_000_000,  # $0.60 por 1M tokens
    },
    "gpt-3.5-turbo": {
        "input": 0.50 / 1_000_000,
        "output": 1.50 / 1_000_000,
    },
}


class PerformanceTracker:
    """Rastreador de métricas de performance"""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.tokens: Dict[str, Dict[str, int]] = {}
        self.costs: Dict[str, float] = {}
        self.start_time: Optional[float] = None
    
    def start(self):
        """Inicia el tracking"""
        self.start_time = time.time()
    
    @contextmanager
    def track(self, operation: str):
        """Context manager para trackear una operación"""
        start = time.time()
        try:
            yield
        finally:
            self.timings[operation] = time.time() - start
    
    def add_llm_metrics(
        self,
        operation: str,
        usage: Any,
        model: str = "gpt-4o-mini"
    ):
        """
        Agrega métricas de una llamada LLM
        
        Args:
            operation: Nombre de la operación (ej: "summary", "answer")
            usage: Objeto usage de OpenAI
            model: Modelo usado
        """
        # Extraer tokens
        tokens = {
            "input": usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0,
            "output": usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
            "total": usage.total_tokens if hasattr(usage, 'total_tokens') else 0,
        }
        
        self.tokens[operation] = tokens
        
        # Calcular costo
        pricing = PRICING.get(model, PRICING["gpt-4o-mini"])
        cost = (
            tokens["input"] * pricing["input"] +
            tokens["output"] * pricing["output"]
        )
        self.costs[operation] = cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Genera resumen de métricas"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Agregar tokens totales
        total_tokens = {
            "input": sum(t.get("input", 0) for t in self.tokens.values()),
            "output": sum(t.get("output", 0) for t in self.tokens.values()),
            "total": sum(t.get("total", 0) for t in self.tokens.values()),
        }
        
        # Costo total
        total_cost = sum(self.costs.values())
        
        return {
            "total_time": round(total_time, 3),
            "breakdown": {
                k: round(v, 3) for k, v in self.timings.items()
            },
            "tokens": {
                **{f"{k}_{subk}": v for k, tok in self.tokens.items() 
                   for subk, v in tok.items()},
                "total_input": total_tokens["input"],
                "total_output": total_tokens["output"],
                "total": total_tokens["total"],
            },
            "cost": {
                **{k: round(v, 6) for k, v in self.costs.items()},
                "total_usd": round(total_cost, 6),
            }
        }