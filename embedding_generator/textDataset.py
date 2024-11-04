
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, ids, tokenizer, max_length=128):
        self.texts = texts
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        misconception_id = self.ids.iloc[idx]

        # Tokenize the text with padding and truncation
        encoding = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        encoding['misconception_id'] = misconception_id
        return encoding
