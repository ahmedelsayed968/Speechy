from collections import deque
from pathlib import Path
import torch
from vad.base import VADServiceBase
from typing import Union,Optional
from config.paths import CONFIG_DIR
class SileroVADModel:
    def __init__(self,
                 sample_rate: int = 16000,
                 chunk_size: int = 512):
        # Load the model and utils from the Silero VAD repository
        self.model_vad, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        # Unpack the tuple returned by torch.hub.load
        (self.get_speech_timestamps, self.save_audio, self.read_audio,
         self.VADIterator, self.collect_chunks) = self.utils
        
        self.model_stream = self.VADIterator(self.model_vad)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
    def process(self, chunks: torch.Tensor, **kwargs):
        """Process audio chunks for streaming VAD"""
        return self.model_stream(chunks, **kwargs)

    def process_file(self,
                    audio: Union[str, torch.Tensor],
                    threshold: float = 0.5,
                    **kwargs) -> Optional[torch.Tensor]:
        """Process entire audio file for VAD"""
        if isinstance(audio, (str,Path)): # FIX 
            wav = self.read_audio(audio)
            # wav = wav.type(torch.float32)
        else:
            wav = audio
        # Process speech timestamps
        timestamps = self.get_speech_timestamps(wav, self.model_vad, 
                                              threshold=threshold, **kwargs)
        if timestamps:
            # Collect chunks
            filtered_audio = self.collect_chunks(timestamps, wav)
            return filtered_audio
        else:
            return None

class SileroVADService(VADServiceBase):
    def __init__(self, processor: SileroVADModel):
        super().__init__()
        
        self.processor = processor

    def process_file(self, file_path:str,**kwargs):
        """Process an entire audio file"""
        return self.processor.process_file(audio=file_path,**kwargs)
    
    def process_stream(self, audio):
        return super().process_stream(audio)

