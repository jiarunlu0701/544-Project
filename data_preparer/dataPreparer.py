import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreparer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
    def load_data(self):
        """Loads the dataset from the specified file path."""
        self.df = pd.read_csv(self.file_path)

    def prepare_features(self):
        """Prepares and combines features into a single text input."""
        X = self.df[['answer', 'ConstructName', 'QuestionText']].astype(str)
        X = " Instruction: Why is the given answer wrong under such circumstances \n " + "answer: " + X['answer'] + " " + "ConstructName: " + X['ConstructName'] + " " + "QuestionText: " + X[
            'QuestionText']
        return X


    def split_data(self, test_size=0.1, random_state=42):
        """Splits the data into training and test sets."""
        X = self.prepare_features()
        Y = self.df['MisconceptionName']
        return train_test_split(X, Y, test_size=test_size, random_state=random_state)

    def prepare_data(self):
        """Full preparation pipeline to load data, prepare features, encode labels, and split."""
        self.load_data()
        return self.split_data()


