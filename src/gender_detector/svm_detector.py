import librosa
import numpy as np
import torch
from gender_detector.base import VoiceGenderDetectorStrategy
from config.paths import GENDER_MODEL_PATH,GENDER_MODEL_SCALER_PATH
from joblib import load
from data.featurize import ECAPASpeechBrainEncoder
from datasets import ClassLabel
from data.prepare import DataProcessor,LoudNessNormalizer,PeakNormalizer
from vad.silero import SileroVADModel, SileroVADService
from torch.nn.functional import pad

class SVMDetector(VoiceGenderDetectorStrategy):
    def __init__(self):
        super().__init__()
        self.head =load(GENDER_MODEL_PATH)
        self.scaler =  load(GENDER_MODEL_SCALER_PATH)
        self.encoder = ECAPASpeechBrainEncoder()
        self.labels = ClassLabel(num_classes=2,names=['male','female'])
    def predict(self,audio, device):
        with torch.no_grad():
            # get embeddings
            embeds = self.encoder.encode_batch(audio) #[B,1,D]
            embeds = embeds.squeeze(0) # shape: [B, D]
        # convert to numpy 
        features = embeds.cpu().numpy()
        # scaler the feaures
        scaled_features = self.scaler.transform(features)
        # pass through head
        prediction = self.head.predict_proba(scaled_features)
        # get the argmax
        label = np.argmax(prediction)
        return self.labels.int2str(int(label)), float(prediction[:,label])
    
def trim_and_pad_audio(
                    audio:torch.Tensor,
                    threshold:float,
                    sample_rate:int
                    )->torch.Tensor:

    target_num_sample = int(threshold * sample_rate)
    audio_num_samples = audio.size(1)
    if audio_num_samples == target_num_sample:
        return audio

    elif audio_num_samples > target_num_sample:
        # do trim
        return audio[:,:target_num_sample]
    else:
        # do padding
        padding_amount = target_num_sample - audio_num_samples
        audio_padded = pad(audio,(0, padding_amount),mode='constant', value=0)
        return audio_padded
if __name__ == "__main__":
    audio, sr = librosa.load("/kaggle/working/Speechy/.data/Teenager Can 'Delay' Her Voice [m5TaxX6RHZ4].wav",sr=16000)
    loud_normalizer = LoudNessNormalizer()
    peak_noramlizer = PeakNormalizer() 
    silero_model = SileroVADModel()
    vad_service  = SileroVADService(silero_model) # torch input
    data_processor = DataProcessor(vad_service=vad_service,
                                   loudness_normalizer=loud_normalizer,
                                   peak_normalizer=peak_noramlizer)

    detector = SVMDetector()

    processed_audio = data_processor.process_audio(signal=audio,sr=sr) # numpy 
    processed_audio_tr = torch.tensor(processed_audio).unsqueeze(0)
    # trim or pad then
    threshold = 11.264
    processed_audio_tr = trim_and_pad_audio(processed_audio_tr,threshold=threshold,sample_rate=sr)

    class_label, probe = detector.predict(processed_audio_tr,None)
    print(class_label,probe)    