from pathlib import Path
from typing import Literal, Optional, Tuple, Union
import librosa
import numpy as np
from pydantic import BaseModel, Field
import torch
from gender_detector.base import VoiceGenderDetectorStrategy
from config.paths import GENDER_MODEL_PATH,GENDER_MODEL_SCALER_PATH
from joblib import load
from data.featurize import ECAPASpeechBrainEncoder
from datasets import ClassLabel
from data.prepare import DataProcessor,LoudNessNormalizer,PeakNormalizer, trim_and_pad_audio
from vad.silero import SileroVADModel, SileroVADService


class SpeecyModelResponse(BaseModel):
    label: Optional[Literal['male','female']] = None
    probability: Optional[float] = Field(default=None,le=1.0,ge=0.0)
    speech: bool
    message: Optional[str] = None

class SpeechyGenderClassifier(VoiceGenderDetectorStrategy):
    def __init__(self,device:torch.device):
        super().__init__()
        self.head =load(GENDER_MODEL_PATH)
        self.scaler =  load(GENDER_MODEL_SCALER_PATH)
        self.encoder = ECAPASpeechBrainEncoder().to(device)
        self.labels = ClassLabel(num_classes=2,names=['male','female'])
        self.device  = device
    def predict(self,audio)->Tuple[str,float]:
        with torch.no_grad():
            # get embeddings
            audio = audio.to(self.device)
            embeds = self.encoder.encode_batch(audio) #[B,1,D]
            embeds = embeds.squeeze(0) # shape: [B, D]
        # convert to numpy 
        features = embeds.cpu().numpy()
        # scaler the feaures
        scaled_features = self.scaler.transform(features)
        # pass through head
        prediction = self.head.predict_proba(scaled_features)
        # get the argmax
        label = int(np.argmax(prediction))
        return self.labels.int2str(int(label)), float(prediction[:,label].item())
    
class SpeechyVoiceGenderDetectionService:
    def __init__(self,device:torch.device):
        self.loud_normalizer = LoudNessNormalizer()
        self.peak_normalizer = PeakNormalizer() 
        self.silero_model = SileroVADModel()
        self.vad_service  = SileroVADService(self.silero_model) 
        self.data_processor = DataProcessor(vad_service=self.vad_service,
                                    loudness_normalizer=self.loud_normalizer,
                                    peak_normalizer=self.peak_normalizer)  
        self.sample_rate = 16000
        self.classifier = SpeechyGenderClassifier(device)
        self.threshold = 11.264 # based on training 95 percentile to cut the audio files 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def predict(self,audio_path:Union[str,Path])->SpeecyModelResponse:
        if not isinstance(audio_path,(str,Path)):
            raise TypeError(f"Expected audio_path as str or Path, got {type(audio_path)}")
        
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            return SpeecyModelResponse(
                speech=False, 
                message=f"Failed to load audio: {str(e)}"
            )
        audio = self.data_processor.process_audio(signal=audio,sr=self.sample_rate) # numpy 
        if audio is None:
            # no speech detected on the file
            return SpeecyModelResponse(speech=False,message="No Speech detected!")
        audio = torch.tensor(audio,dtype=torch.float32).unsqueeze(0)
        audio = trim_and_pad_audio(audio,threshold=self.threshold,sample_rate=self.sample_rate)
        class_label, probability = self.classifier.predict(audio)
        return SpeecyModelResponse(label=class_label,
                                   probability=probability,
                                   speech=True,
                                    message="Speech detected and gender classified successfully.")

