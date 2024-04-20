from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, x, y, max_len, tokenizer):
      self.tokenizer=tokenizer
      self.max_len=max_len
      
      self.x_raw, self.y_raw = x, y
      self.x = self.prepare_features(x,y)

    def __len__(self):
      return len(self.x)

    def __getitem__(self, idx):
      return {'x_raw': self.x_raw[idx], 'x': self.x[idx]}
  
    def prepare_features(self, x, y):
      total = len(x)

      sentences = x + y

      sent_features = self.tokenizer(
          sentences,
          max_length=self.max_len,
          truncation=True,
          padding="max_length",
      )

      features = {}
      for key in sent_features:
            features[key]=[[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
      return features
    
class ViznetDataset(Dataset):
    def __init__(self, x, y, max_len, tokenizer):
      self.tokenizer=tokenizer
      self.max_len=max_len
      
      self.x_raw, self.y_raw = x, y
      self.x = self.prepare_features(x)
      self.y = y

    def __len__(self):
      return len(self.x)

    def __getitem__(self, idx):
      return {'x': self.x[idx], 'y': self.y[idx]}
  
    def prepare_features(self, x):
      sent_features = self.tokenizer(
          x,
          max_length=self.max_len,
          truncation=True,
          padding="max_length",
      )
      return sent_features