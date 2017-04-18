

import pandas as pd


def dataloader(train_path="data/train.csv", test_path="data/test.csv"):
    # Training set
    train = pd.read_csv(train_path)
    train = train.fillna('empty') # There are two nulls in question2
    # Test set
    test = pd.read_csv(test_path)
    test = test.fillna('empty') # There are two nulls in question1 and four question2
    return train, test

