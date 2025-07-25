import io
from typing import Optional
import librosa
import numpy as np
import torch
import torchaudio
from data.base import EncoderBase
from speechbrain.inference.speaker import EncoderClassifier
from datasets import load_dataset,Audio
from torch.nn.functional import pad

class ECAPASpeechBrainEncoder(EncoderBase):
    def __init__(self) -> None:
      super().__init__()
      self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    def to(self, device: torch.device)->EncoderClassifier:
       _ = self.classifier.mods.to(device)
       return self

    def encode_batch(self, batch: torch.Tensor, **kwargs):
       return self.classifier.encode_batch(
           batch,
           **kwargs
       )

class FeatureExtractor:
    def __init__(self,
               repo_id:str,
               revision:Optional[str],
               encoder:EncoderBase,
               device:torch.device,
               target_col:str,
               sample_rate:int=16000) -> None:

        self.ds = load_dataset(repo_id,revision=revision)
        self.target_col = target_col
        self.ds  = self.ds.cast_column(self.target_col,Audio(sampling_rate=sample_rate,decode=False))
        self.encoder = encoder.to(device=device)
        self.sample_rate=sample_rate
        self.device = device
        self.embeds = None
        self.threshold = None

    def get_audio_duration(self,batch):
        durations = []
        for entry in batch[self.target_col]:
            if entry is None:
                durations.append(None)
            else:
                bytes_data = io.BytesIO(entry['bytes'])
                audio_data,_ = librosa.load(bytes_data,sr=self.sample_rate)
                durations.append(librosa.get_duration(y=audio_data,sr=self.sample_rate))
        return {
            "duration_sec": durations
        }

    def _trim_and_pad_audio(
                        self,
                        audio:torch.Tensor,
                        )->torch.Tensor:

        target_num_sample = int(self.threshold * self.sample_rate)
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
    def _get_embeddings(self,batch):
        wavs = []
        for audio in batch[self.target_col]:

            wav,_ = torchaudio.load(audio['bytes']) # [channel,Time]
            processed_wav = self._trim_and_pad_audio(wav) # [channel,Time]
            wav = processed_wav.squeeze(0) # [Time]
            wavs.append(wav)

        wavs_tensor = torch.stack(wavs) # [batch,Time]
        wavs_tensor = wavs_tensor.to(self.device) 

        # Pass through model
        with torch.no_grad():
            embeddings = self.encoder.encode_batch(wavs_tensor)  # shape: [B, 1, D]
            embeddings = embeddings.squeeze(1) # [B,Dim]
        return {
            "inputs": embeddings.cpu().numpy()
        }

    def create_embeddings(self):
        # get the duration of each audio file
        ds_with_duration = self.ds.map(self.get_audio_duration,batched=True,batch_size=16)
        # remove nones from the all splits
        ds_with_duration = ds_with_duration.filter(lambda x: x.get(self.target_col) is not None,batched=False)
        train_durations = ds_with_duration['train']['duration_sec']
        # get the threshold to keep 95% of the training data
        self.threshold = np.percentile(train_durations,95)
        # filter all entries by the the threshold
        ds_with_duration = ds_with_duration.filter(lambda x: x.get("duration_sec") <= self.threshold,batched=False)
        # create embeddings
        ds_with_embeddings = ds_with_duration.map(self._get_embeddings,batched=True,batch_size=32)
        self.embeds = ds_with_embeddings
        return self.embeds
