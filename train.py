from utils import parse_arguments, check_and_retrieveVocabulary
from SPAN.SPAN import get_span_model
import torch
from data_load import load_data_testcase
import numpy as np
import torch.optim as optim
from utils import levenshtein
from itertools import groupby
import random
from tqdm import tqdm

def test_model(model, X, Y, i2w, device):
    acc_ed_dist = 0

    randomindex = random.randint(0, len(X)-1)

    with torch.no_grad():
      for i in range(len(X)):
          pred = model(torch.Tensor(np.expand_dims(np.expand_dims(X[i],axis=0),axis=0)).to(device))
          
          b, c, h, w = pred.size()
          pred = pred.reshape(b, c, h*w)
          pred = pred.permute(0,2,1)
          
          pred = pred[0]
          out_best = torch.argmax(pred,dim=1)

          # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
          out_best = [k for k, g in groupby(list(out_best))]
          decoded = []
          for c in out_best:
              if c < len(i2w):  # CTC Blank must be ignored
                  decoded.append(i2w[c.item()])

          groundtruth = [i2w[label] for label in Y[i]]

          if(i == randomindex):
              print(f"Prediction - {decoded}")
              print(f"True - {groundtruth}")

          
          edit_dist = levenshtein(decoded, groundtruth)
          
          acc_ed_dist += edit_dist / len(groundtruth)


      ser = 100.*acc_ed_dist / len(X)
    
    return ser

def data_preparation_CTC(X, Y):
    max_image_width = max([img.shape[1] for img in X])
    max_image_height = max([img.shape[0] for img in X])

    X_train = np.zeros(shape=[len(X), 1, max_image_height, max_image_width], dtype=np.float32)
    L_train = np.zeros(shape=[len(X)])

    for i, img in enumerate(X):
        X_train[i, 0, 0:img.shape[0], 0:img.shape[1]] = img
        L_train[i] = (img.shape[1] // 8) * (img.shape[0] // 32)

    max_length_seq = max([len(w) for w in Y])

    Y_train = np.ones(shape=[len(Y),max_length_seq])
    T_train = np.zeros(shape=[len(Y)])
    for i, seq in enumerate(Y):
        Y_train[i, 0:len(seq)] = seq
        T_train[i] = len(seq)
    
    return torch.tensor(X_train), torch.tensor(Y_train, dtype=torch.long), torch.tensor(L_train, dtype=torch.long), torch.tensor(T_train, dtype=torch.long)


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
    
    maxwidth = max([img.shape[1] for img in XTrain])
    maxheight = max([img.shape[0] for img in XTrain])
    model, device = get_span_model(maxwidth=maxwidth, maxheight=maxheight, in_channels=1, out_size=len(w2i))
    batch_gen = batch_generator(XTrain, YTrain, args.batch_size, synth_prop=0.5)
    criterion = torch.nn.CTCLoss(blank=len(w2i)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Training with {len(XTrain)} samples")
    print(f"Validating with {len(XVal)} samples")
    print(f"Testing with {len(XTest)} samples")

    for epoch in range(5000):
        model.train()
        for mini_epoch in range(10):
            accum_loss = 0
            for _ in tqdm(range(len(XTrain)//args.batch_size)):
                net_input, net_tar, input_len, tar_len = next(batch_gen)
                predictions = model(net_input.to(device))
                
                # From Conv2D output to sequential interpretation by making row concat
                b, c, h, w = predictions.size()
                predictions = predictions.reshape(b, c, h*w)
                predictions = predictions.permute(2,0,1)

                loss = criterion(predictions, net_tar.to(device), input_len.to(device), tar_len.to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accum_loss += loss.item()
            
            epoch_loss = accum_loss / (len(XTrain)//args.batch_size)
            print(f"Step {mini_epoch + 1} / 10 - Loss: {epoch_loss}")

        model.eval()
        SER_VAL = test_model(model, XVal, YVal, i2w, device)
        SER_TEST = test_model(model, XVal, YVal, i2w, device)
        print(f"EPOCH {epoch + 1} ---  VAL SER {SER_VAL} | TEST SER {SER_TEST}")

        
if __name__=="__main__":
    main()
