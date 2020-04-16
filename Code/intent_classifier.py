import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file, delimiter = ',', header=None names):
        # AG_NEWS : names = ['category', 'head', 'content']
        # IMDB_Dataset : names = ['catetory', 'sentence']
        file_type = file[file.find('.'):]
        if file_type == '.csv':
            self.df = pd.read_csv(file, delimiter=delimiter, header=header, names=names)
        elif file_type == '.json':
            pass

        self.sentence = self.df[names[0]]
        self.category = self.df[names[1]]

    def __len__(self):
        return len(self.df.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()




def run():
    # Check if there is a GPU available.
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print(torch.cuda.get_device_name(0), 'will be used.')
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')


    # Load data from custom dataset class
    train_dataset = CustomDataset(file="./Dataset/IMDB_Dataset.csv", names=['catetory', 'sentence'])
