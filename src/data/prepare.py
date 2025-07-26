import io
from pathlib import Path
from typing import Union
from datasets import Dataset,DatasetDict,Audio,ClassLabel
import librosa
import numpy as np
import pandas as pd
import os

import torch
from dotenv import load_dotenv
import argparse
import pyloudnorm as pyln
import torchaudio
from torchaudio.transforms import Resample
from data.base import AudioNormalizer
from vad.base import VADServiceBase
import noisereduce as nr
from librosa.util import normalize
from torch.nn.functional import pad
load_dotenv()

class VoxCelebDataset:
    def __init__(self,
                 path:str,
                 revision:str):
        self.path = Path(path) if path is not None else None
        self.revision = revision
    def process(self):
        male_dir = self.path / "male"
        female_dir = self.path / "female"

        # get all files in each directory
        male_all_files = list(map(str,male_dir.iterdir()))
        female_all_files = list(map(str,female_dir.iterdir()))
        # clear class attribute with the same length of corresponding files
        male_class = ['male'] * len(male_all_files)
        female_class = ['female'] * len(female_all_files)
        # create dataframe 
        male_df = pd.DataFrame({
            "audio":male_all_files,
            "class":male_class
        })
        female_df = pd.DataFrame({
            "audio":female_all_files,
            "class":female_class
        })
        df = pd.concat([male_df,female_df],ignore_index=True)
        # downsample the df
        df = self.down_sampling_dataframe(df)
        # rename the class to label
        df = df.rename(columns={"class": "label"})
        # create ClassLabel 
        label_feature = ClassLabel(names=["male", "female"])

        # convert to hf dataset
        ds = Dataset.from_pandas(df)
        # cast label to label feature
        ds = ds.cast_column("label",label_feature)
        # split the dataset
        splitted_ds = self.split_dataset(ds)

        # Cast the audio column
        splitted_ds = splitted_ds.cast_column("audio",Audio())
        splitted_ds.push_to_hub(repo_id=os.environ.get("DATASET_REPO_ID"),
                       token=os.environ.get("HF_TOKEN"),
                       revision=self.revision) 

    def split_dataset(self,ds:Dataset)->DatasetDict:
        # split the set by 60 20 20
        div1 = ds.train_test_split(seed=211120,test_size=0.2,shuffle=True,)
        div2 = div1['train'].train_test_split(seed=211120,test_size=0.2,shuffle=True)
        # get the splits 
        train_ds = div2['train']
        val_ds = div2['test']
        test_ds = div1['test'] 
        # create new datasetDict 
        new_version_ds = DatasetDict({
            "train":train_ds,
            "val":val_ds,
            "test":test_ds
        })
        return new_version_ds
    def down_sampling_dataframe(self,df:pd.DataFrame)->pd.DataFrame:
        # get class distribution
        class_dist  = df['class'].value_counts().to_dict()
        # get the minority class num_samples
        min_num_samples = min(list(class_dist.values()))
        subsets = []
        for label in class_dist.keys():
            subset = df[df['class'] == label].sample(n=min_num_samples,replace=False,random_state=211120)
            subsets.append(subset)
        return pd.concat(subsets,ignore_index=True)


class LoudNessNormalizer(AudioNormalizer):
    def __init__(self,loudness_level=-12) -> None:
        super().__init__()
        self.loudness_level = loudness_level
    def normalize(self, audio:np.array,sr:int)->np.array:
      # measure the loudness first
      meter = pyln.Meter(sr) # create BS.1770 meter
      loudness = meter.integrated_loudness(audio)

      # loudness normalize audio
      loudness_normalized_audio = pyln.normalize.loudness(audio, loudness, self.loudness_level)
      return loudness_normalized_audio

class PeakNormalizer(AudioNormalizer):
    def __init__(self):
        super().__init__()
    def normalize(self, audio):
        return normalize(S=audio)

class DataProcessor:
    def __init__(self,
                 vad_service:VADServiceBase,
                 loudness_normalizer: AudioNormalizer,
                 peak_normalizer: AudioNormalizer   
                 ):
        
        self.vad_service = vad_service
        self.loudness_normalizer = loudness_normalizer
        self.peak_normalizer = peak_normalizer

    def process_audio(self,signal:Union[bytes,np.ndarray],sr:int)->np.ndarray:
        if isinstance(signal,bytes):
            bytes_data = io.BytesIO(signal)
            signal,sr= librosa.load(bytes_data,sr=sr)

        # remove all noise from the audio file
        audio = nr.reduce_noise(signal,sr=sr)
        # trim silence
        results = self.vad_service.process_file(torch.tensor(audio,dtype=torch.float32))
        if results is None:
            return None

        # loud normalization
        loud_normalizer = LoudNessNormalizer(loudness_level=-14)
        audio = loud_normalizer.normalize(audio,sr=sr)
        # Peak Normalization
        audio = self.peak_normalizer.normalize(audio)
        audio = np.clip(audio, -1.0, 1.0)

        return audio

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
    
def load_audio(path: str) -> torch.Tensor:
    audio, sr = torchaudio.load(path)
    if sr != 16000:
        resampler = Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
    return audio.mean(dim=0, keepdim=True)  # Convert to mono if stereo
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input")
    parser.add_argument("-r","--revision")
    args = parser.parse_args()
    ds = VoxCelebDataset(args.input,revision=args.revision)
    ds.process()