from utils import parse_arguments, check_and_retrieveVocabulary
from SPAN_TF.SPAN_MODEL import get_span_model
from data_load import load_data_testcase
import numpy as np
from utils import levenshtein
from itertools import groupby
import random
from tqdm import tqdm

def validateModel(model, X, Y, i2w):
    acc_ed_ser = 0
    acc_len_ser = 0

    randomindex = random.randint(0, len(X)-1)

    for i in range(len(X)):
        pred = model.predict(np.expand_dims(np.expand_dims(X[i],axis=0),axis=-1))[0]

        out_best = np.argmax(pred,axis=1)

        # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
        out_best = [k for k, g in groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c < len(i2w):  # CTC Blank must be ignored
                decoded.append(i2w[c])

        groundtruth = [i2w[label] for label in Y[i]]

        if(i == randomindex):
            print(f"Prediction - {decoded}")
            print(f"True - {groundtruth}")

        acc_len_ser += len(Y[i])
        acc_ed_ser += levenshtein(decoded, groundtruth)


    ser = 100. * acc_ed_ser / acc_len_ser
    return ser

def data_preparation_CTC(X, Y):
    max_image_width = max([img.shape[1] for img in X])
    max_image_height = max([img.shape[0] for img in X])

    X_train = np.zeros(shape=[len(X), max_image_height, max_image_width, 1], dtype=np.float32)
    L_train = np.zeros(shape=[len(X)])

    for i, img in enumerate(X):
        X_train[i, 0:img.shape[0], 0:img.shape[1], 0] = img
        L_train[i] = (img.shape[1] // 8) * (img.shape[0] // 32)

    max_length_seq = max([len(w) for w in Y])

    Y_train = np.ones(shape=[len(Y),max_length_seq])
    T_train = np.zeros(shape=[len(Y)])
    for i, seq in enumerate(Y):
        Y_train[i, 0:len(seq)] = seq
        T_train[i] = len(seq)
    
    return X_train, Y_train, L_train, T_train


def batch_generator(X,Y, BATCH_SIZE, synth_prop=0.3):
    idx = 0
    while True:
        BatchX = X[idx:idx+BATCH_SIZE]
        BatchY = Y[idx:idx+BATCH_SIZE]
        yield data_preparation_CTC(BatchX, BatchY)

        idx = (idx + BATCH_SIZE) % len(X)

def main():
    args = parse_arguments()
    print("Loading train set:")
    XTrain, YTrain = load_data_testcase(PATH=f"{args.data_path}/train/")
    print("Loading val set:")
    XVal, YVal = load_data_testcase(PATH=f"{args.data_path}/val/")
    print("Loading test set:")
    XTest, YTest = load_data_testcase(PATH=f"{args.data_path}/test/")

    w2i, i2w = check_and_retrieveVocabulary([YTrain, YVal, YTest], f"./vocab", f"{args.corpus_name}")
    
    for i in range(len(XTrain)):
        XTrain[i] = (255. - XTrain[i]) / 255.
        seq = []
        for symbol in YTrain[i]:
            seq.append(w2i[symbol])
        
        YTrain[i] = seq
    
    for i in range(len(XVal)):
        XVal[i] = (255. - XVal[i]) / 255.
        seq = []
        for symbol in YVal[i]:
            seq.append(w2i[symbol])
        
        YVal[i] = seq
    
    for i in range(len(XTest)):
        XTest[i] = (255. - XTest[i]) / 255.
        seq = []
        for symbol in YTest[i]:
            seq.append(w2i[symbol])
        
        YTest[i] = seq
    
    #maxwidth = max([img.shape[1] for img in XTrain])
    #maxheight = max([img.shape[0] for img in XTrain])
    model_train, model_pred, _ = get_span_model(input_shape=(None, None, 1), out_tokens=len(w2i))

    batch_gen = batch_generator(XTrain, YTrain, args.batch_size, synth_prop=0.5)

    print(f"Training with {len(XTrain)} samples")
    print(f"Validating with {len(XVal)} samples")
    print(f"Testing with {len(XTest)} samples")

    best_ser = 10000
    patience = 5

    for epoch in range(5000):
        model_train.fit(batch_gen, steps_per_epoch=len(XTrain)//args.batch_size, epochs=1, verbose=1)
        SER_VAL = validateModel(model_pred, XVal, YVal, i2w)
        SER_TEST = validateModel(model_pred, XTest, YTest, i2w)
        print(f"EPOCH {epoch + 1} ---  VAL SER {SER_VAL} | TEST SER {SER_TEST}")
        if SER_VAL < best_ser:
           print("SER improved - Saving epoch")
           model_train.save_weights(f"models/{args.model_name}_{args.corpus_name}.h5")
           best_ser = SER_VAL
           patience = 5
        else:
            patience -= 1
            if patience == 0:
                break
        
if __name__=="__main__":
    main()
