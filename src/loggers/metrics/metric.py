from tensordict import TensorDict


class DefaultMetric:
    def __init__(self, name: str):
        self.name = name
    
    def compute(self, data: TensorDict) -> float:
        return data['next'][self.name].item()
