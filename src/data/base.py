from abc import ABC,abstractmethod

import torch
class EncoderBase(ABC):
    @abstractmethod
    def to(self,device:torch.device):
      pass
    @abstractmethod
    def encode_batch(self,batch:torch.Tensor,**kwargs):
        pass

class AudioNormalizer(ABC):
    @abstractmethod
    def normalize(self,audio):
        pass
