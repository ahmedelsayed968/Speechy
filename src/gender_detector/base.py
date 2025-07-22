from abc import ABC,abstractmethod

import torch

class VoiceGenderDetectorStrategy(ABC):
    @abstractmethod
    def predict(audio_file:str,device: torch.device):
        pass