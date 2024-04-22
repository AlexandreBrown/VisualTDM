import torch
import math


class SimpleLogger:
    def __init__(self, stage_prefix: str):
        self.stage_prefix = stage_prefix
        self.step_log_prefix = f"{self.stage_prefix}step_"
        self.step_metrics = {}
        self.episode_metrics = {}
    
    def accumulate_step_metrics(self, metrics: dict):
        self._accumulate_metrics(self.step_metrics, metrics)
    
    def _accumulate_metrics(self, agg_metrics: dict, new_metrics: dict):
        for (k,v) in new_metrics.items():
            self._accumulate_metric(agg_metrics, k, v)
    
    def _accumulate_metric(self, agg_metrics: dict, key: str, value: float):
        if key not in agg_metrics.keys():
            agg_metrics[key] = [value]
        else:
            agg_metrics[key] += [value]
    
    def accumulate_step_metric(self, key: str, value: float):
        self._accumulate_metric(self.step_metrics, key, value)
        
    def compute_step_metrics(self) -> dict:
        output = self._compute_metrics(self.step_metrics, prefix=self.step_log_prefix)
        self.step_metrics = {}
        return output
    
    def _compute_metrics(self, agg_metrics: dict, prefix: str) -> dict:
        output_metrics = {}
        for (k,v) in agg_metrics.items():
            values = torch.tensor(v)
            k = k.replace(prefix, "")
            k = k.replace("_mean", "")
            k = k.replace("_median", "")
            k = k.replace("_std", "")
            k = k.replace("_min", "")
            k = k.replace("_max", "")
            output_metrics[f"{prefix}{k}_mean"] = torch.mean(values).item()
            output_metrics[f"{prefix}{k}_median"] = torch.median(values).item()
            std = torch.std(values).item()
            if not math.isnan(std):
                output_metrics[f"{prefix}{k}_std"] = std
            output_metrics[f"{prefix}{k}_min"] = torch.min(values).item()
            output_metrics[f"{prefix}{k}_max"] = torch.max(values).item()
        
        return output_metrics
