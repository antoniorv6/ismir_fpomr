import argparse
import numpy as np
from os import path
import os

def parse_arguments_ds():
    parser = argparse.ArgumentParser(description="Program arguments to work")
    parser.add_argument('--corpus_name', type=str, default=None, help="Image folder to load")
    parser.add_argument('--image_folder', type=str, default=None, help="Image folder to load")
    parser.add_argument('--agnostic_folder', type=str, default=None, help="Agnostic folder to load")

    args = parser.parse_args()
    return args

def parse_arguments():
    parser = argparse.ArgumentParser(description="Program arguments to work")
    parser.add_argument('--corpus_name', type=str, default=None, help="Corpus name")
    parser.add_argument('--model_name', type=str, default=None, help="Model name")
    parser.add_argument('--data_path', type=str, default=None, help="Image folder to load")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size")


    args = parser.parse_args()
    return args

def check_and_retrieveVocabulary(YSequences, pathOfSequences, nameOfVoc):
    w2ipath = pathOfSequences + "/" + nameOfVoc + "w2i.npy"
    i2wpath = pathOfSequences + "/" + nameOfVoc + "i2w.npy"

    w2i = []
    i2w = []

    if not path.isdir(pathOfSequences):
        os.mkdir(pathOfSequences)

    if path.isfile(w2ipath):
        w2i = np.load(w2ipath, allow_pickle=True).item()
        i2w = np.load(i2wpath, allow_pickle=True).item()
    else:
        w2i, i2w = make_vocabulary(YSequences, pathOfSequences, nameOfVoc)

    return w2i, i2w

def make_vocabulary(YSequences, pathToSave, nameOfVoc):
    vocabulary = set()
    for samples in YSequences:
        for element in samples:
                #print(token)
                vocabulary.update(element)

    #Vocabulary created
    w2i = {symbol:idx+2 for idx,symbol in enumerate(vocabulary)}
    i2w = {idx+2:symbol for idx,symbol in enumerate(vocabulary)}
    
    w2i['<pad>'] = 0
    i2w[0] = '<pad>'

    #Save the vocabulary
    np.save(pathToSave + "/" + nameOfVoc + "w2i.npy", w2i)
    np.save(pathToSave + "/" + nameOfVoc + "i2w.npy", i2w)

    return w2i, i2w

# Dados vectores de X (imagenes) e Y (secuencia de etiquetas numéricas -no one hot- devuelve los 4 vectores necesarios para CTC)
def data_preparation_CTC(X, Y, lines):
    # X_train, L_train
    max_image_width = max([img.shape[1] for img in X])
    max_image_height = max([img.shape[0] for img in X])

    X_train = np.zeros(shape=[len(X), max_image_height, max_image_width, 1], dtype=np.float32)
    L_train = np.zeros(shape=[len(X),1])

    for i, img in enumerate(X):
        X_train[i, 0:img.shape[0], 0:img.shape[1],0] = img
        if lines:
            L_train[i] = (img.shape[1] // 8)
        else:
            L_train[i] = (img.shape[1] // 8) * (img.shape[0] // 32)
            
    # Y_train, T_train
    max_length_seq = max([len(w) for w in Y])

    Y_train = np.zeros(shape=[len(X),max_length_seq])
    T_train = np.zeros(shape=[len(X),1])
    for i, seq in enumerate(Y):
        Y_train[i, 0:len(seq)] = seq
        T_train[i] = len(seq)

    inputs = {'the_input': X_train,
                 'the_labels': Y_train,
                 'input_length': L_train,
                 'label_length': T_train,
                 }
    
    outputs = {'ctc': np.zeros([len(X_train)])}

    return inputs, outputs

def ctc_batch_generator(BATCH_SIZE, X, Y, lines):
    index = 0
    while True:
        BatchX = X[index:index + BATCH_SIZE]
        BatchY = Y[index:index + BATCH_SIZE]

        yield data_preparation_CTC(BatchX, BatchY, lines)

        index = (index + BATCH_SIZE) % len(X)


def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
