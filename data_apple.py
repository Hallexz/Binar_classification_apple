import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch


''' Reading the dataset. The column 'Quality' has only two values str, we transfer it to the bin system. 
The column 'Acidity' has type object a translatable type float64. Delete the ID column 'A_id' '''
apple = pd.read_csv('apple_quality.csv', engine='python')
apple['Quality'] = apple['Quality'].map({'good' : 1, 'bad' : 0})
apple['Acidity'] = apple['Acidity'].astype('float64')
apple = apple.drop(columns=['A_id'], axis=1)

''' Replaces those values ​​that cannot be passed to '''
apple = apple.apply(lambda x: pd.to_numeric(x, errors='coerce')) 
apple = apple.dropna(axis=1) #Delete all NaN

''' I transform the data in a tensor with a data type float64, 
returning the number of elements of the tensor and the tensor of the element by index.'''
class CSVDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X.iloc[idx].values).float(), torch.tensor(self.y.iloc[idx]).float()


X = apple.drop('Quality', axis=1)
y = apple['Quality']


''' Divides data into test and training '''
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


''' A data loader is created, which divides it into pieces of 32 elements. 
Data is dragged to each training epoch '''
train_dataset = CSVDataset(X_train, y_train)
test_dataset = CSVDataset(X_test, y_test)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
















