import os
import glob
import json
import random
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def load_files(images_path, indexes):
    files = glob.glob(f'{images_path}/*/*.json')
    random.shuffle(files)
    labels = []
    features = []
    print('Loading files...')
    for f in tqdm(files):
        #print(f)
        f = f.replace("\\",'/')
        label = f.split('/')[-2]
        labels.append(label)

        with open(f, 'r') as features_file:
            features.append(np.array(json.load(features_file)))
    labels = np.array(labels)
    features = np.array(features)
    return features, labels

def main():

    #X, y = make_classification(n_samples=100, random_state=1)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
    #                                                random_state=1)
    #
    #print(X)
    #print(y)
    #print(type(X))
    #print(type(y))

    dataset = 'butterflies'
    dataset_path = f'data/{dataset}'
    train_directory = os.path.join(dataset_path, 'train')
    valid_directory = os.path.join(dataset_path, 'valid')
    test_directory = os.path.join(dataset_path, 'test')

    classes = os.listdir(valid_directory)
    indexes = {v: k for k, v in enumerate(classes)}

    x_train, y_train = load_files(train_directory, indexes)
    x_valid, y_valid = load_files(valid_directory, indexes)

    print('Training classifier...')
    clf = MLPClassifier(random_state=1, 
                        max_iter=300, 
                        verbose=True, 
                        shuffle=True, 
                        learning_rate_init=0.00001).fit(x_train, y_train)
    print(clf.predict(x_valid))
    print(clf.score(x_valid, y_valid))

if __name__ == '__main__':
    main()