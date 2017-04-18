'''Main function




'''

from basic_preproc import basic_preproc
from dataloader import dataloader

from sklearn.model_selection import KFold
import numpy as np

if __name__ == "__main__":
    print("Loading dataset...")
    train, test = dataloader()
    print("Data loaded")
    print("Cleaning training text...")
    train.question1, train.question2 = pre_train = basic_preproc(train.question1, train.question2,
                                                                 filepath="data/data_pre/train.csv", use_cached=False)
    a = 0
    for i in range(a, a + 10):
        print(train.question1[i])
        print(train.question2[i])
        print()

    print("Text cleaned")
    print("Cleaning test text...")
    test.question1, test.question2 = pre_test = basic_preproc(test.question1, test.question2,
                                                              filepath="data/data_pre/train.csv", use_cached=False)
    print("Text cleaned")

    training_set = np.concatenate((train.question1,train.question2),axis=1)
    training_labels = train.is_duplicate
    test_set = np.concatenate((test.question1,test.question2),axis=1)

    print("Getting cross-validation")
    kf = KFold(n_splits=2)
    kf.get_n_splits(training_set)
    # Training
#    for train_index, test_index in kf.split(training_set):

    # Test
