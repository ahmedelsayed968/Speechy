from pathlib import Path
from datasets import Dataset,DatasetDict,Audio,load_dataset,concatenate_datasets,ClassLabel
import pandas as pd
import os
from dotenv import load_dotenv
import argparse
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input")
    parser.add_argument("-r","--revision")
    args = parser.parse_args()
    ds = VoxCelebDataset(args.input,revision=args.revision)
    ds.process()