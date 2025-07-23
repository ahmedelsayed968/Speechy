from abc import ABC,abstractmethod
from typing import Union

import torch

class VoiceGenderDetectorStrategy(ABC):
    @abstractmethod
    def predict(audio : Union[str,torch.Tensor],device: torch.device):
        pass