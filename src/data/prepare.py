from pathlib import Path
from datasets import Dataset,DatasetDict,Audio
import pandas as pd
import os
from dotenv import load_dotenv
import argparse
load_dotenv()

class VoxCelebDataset:
    def __init__(self,path:str):
        self.path = Path(path)
    
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
        # convert to hf dataset
        ds = Dataset.from_pandas(df)
        # Cast the audio column
        ds = ds.cast_column("audio",Audio())
        ds.push_to_hub(repo_id=os.environ.get("DATASET_REPO_ID"),
                       token=os.environ.get("HF_TOKEN")) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input")
    args = parser.parse_args()
    ds = VoxCelebDataset(args.input)
    ds.process()