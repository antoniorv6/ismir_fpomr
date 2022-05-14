from base64 import decode
from augmentations import Brightness, Contrast, DPIAdjusting, Dilation, Erosion, Perspective
from utils import parse_arguments, check_and_retrieveVocabulary
from SPAN_Torch.SPAN import get_span_recurrent_model, get_span_transformer_model
import torch
from data_load import load_data_jsonMuret, load_data_testcase
import numpy as np
import torch.optim as optim
from utils import levenshtein
from itertools import groupby
import random
import cv2
from utils import writeResults
import os

# Utility function for obtaining the best split-seq path
def forced_split_decoder(y_pred, i2w):
    """Decodes (CTC forced split-sequence decoder + i2w conversion) the output of the CRNN-CTC model.
    Arguments:
    - y_pred    tensor (samples, time_steps, num_categories) containing the prediction
    - i2w       dictionary for the int-to-word conversion."""
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = [y_pred]

    dec_pred = []
    # Iterate over each batch item in y_pred
    for id, sample in enumerate(y_pred):
        dec_sample = []
        prev_blank = False
        prev_sym = None
        prev_type = 1  # To force the first decoded symbol (that is not a blank label) to be of shape type
        # Iterate over each time step in each batch item
        for ts in sample:
            ts_asc = np.argsort(ts)
            for cat in reversed(ts_asc):
                # Get decoded symbol for current time step
                csym = i2w[int(cat)] if int(cat) != len(i2w) else 'blank'
                # Get symbol type (shape or pitch); shape = 0, pitch = 1
                if csym[0] == 'S' or csym[0] == 'L':
                    ctype = 1
                else:
                    ctype = 0
                # Append symbol
                if csym == 'blank':
                    dec_sample.append(csym)
                    prev_sym, prev_type = prev_sym, prev_type
                    prev_blank = True
                    break
                elif csym == prev_sym and ctype == prev_type and not prev_blank:
                    dec_sample.append(csym)
                    prev_sym, prev_type = csym, ctype
                    prev_blank = False
                    break
                elif csym != prev_sym and ctype != prev_type:
                    dec_sample.append(csym)
                    prev_sym, prev_type = csym, ctype
                    prev_blank = False
                    break
        # Merge repeated symbols
        dec_sample = [k for k, g in groupby(dec_sample)]
        # Delete blank labels
        dec_sample = [sym for sym in dec_sample if sym != 'blank']
        # Append sequence
        dec_pred = dec_sample

    return dec_pred


def test_sseq(model, X, Y, i2w, device):
    acc_ed_dist = 0
    acc_len = 0

    randomindex = random.randint(0, len(X)-1)

    preds = []
    gts = []

    with torch.no_grad():
      for i in range(len(X)):
          pred = model(torch.Tensor(np.expand_dims(np.expand_dims(X[i],axis=0),axis=0)).to(device))  
          pred = pred[0]
          out_best = torch.argmax(pred,dim=1)

          # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
          #out_best = [k for k, g in groupby(list(out_best))]
          #decoded = []
          #for c in out_best:
          #    if c < len(i2w):  # CTC Blank must be ignored
          #        decoded.append(i2w[c.item()])

          decoded = forced_split_decoder(out_best, i2w)

          groundtruth = [i2w[label] for label in Y[i]]

          preds.append(decoded)
          gts.append(groundtruth)

          if(i == randomindex):
              print(f"Prediction - {decoded}")
              print(f"True - {groundtruth}")

          acc_ed_dist += levenshtein(decoded, groundtruth)
          acc_len += len(groundtruth)

    ser = 100.*acc_ed_dist / acc_len
    
    return ser, preds, gts


def test_model(model, X, Y, i2w, device):
    acc_ed_dist = 0
    acc_len = 0

    randomindex = random.randint(0, len(X)-1)

    preds = []
    gts = []

    with torch.no_grad():
      for i in range(len(X)):
          pred = model(torch.Tensor(np.expand_dims(np.expand_dims(X[i],axis=0),axis=0)).to(device))  
          pred = pred[0]
          out_best = torch.argmax(pred,dim=1)

          # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
          out_best = [k for k, g in groupby(list(out_best))]
          decoded = []
          for c in out_best:
              if c < len(i2w):  # CTC Blank must be ignored
                  decoded.append(i2w[c.item()])

          groundtruth = [i2w[label] for label in Y[i]]

          preds.append(decoded)
          gts.append(groundtruth)

          if(i == randomindex):
              print(f"Prediction - {decoded}")
              print(f"True - {groundtruth}")

          acc_ed_dist += levenshtein(decoded, groundtruth)
          acc_len += len(groundtruth)

    ser = 100.*acc_ed_dist / acc_len
    
    return ser, preds, gts

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


def batch_generator(X,Y, BATCH_SIZE):
    idx = 0
    while True:
        BatchX = X[idx:idx+BATCH_SIZE]
        BatchY = Y[idx:idx+BATCH_SIZE]

        yield data_preparation_CTC(BatchX, BatchY)

        idx = (idx + BATCH_SIZE) % len(X)

def augmentation_process(X):

    X = np.array(X)

    if np.random.rand() < 0.2:
        #print("DPI")
        scale = np.random.uniform(0.75, 1)
        X = DPIAdjusting(X, scale)
    
    if np.random.rand() < 0.2:
        kernel_size = np.random.randint(1, 3)
        iterations = 1
        #print(f"Dilation - {kernel_size}")
        X = Dilation(X, kernel_size, iterations)
    
    if np.random.rand() < 0.2:
        kernel_size = np.random.randint(1, 3)
        iterations = 1
        #print(f"Erosion - {kernel_size}")
        X = Erosion(X, kernel_size, iterations)
    
    if np.random.rand() < 0.2:
        brightness_factor = np.random.uniform(0.01, 1)
        #print(f"Brightness - {brightness_factor}")
        X = Brightness(X, brightness_factor)
    
    if np.random.rand() < 0.2:
        contrast_factor = np.random.uniform(0.01, 1)
        #print(f"Contrast - {contrast_factor}")
        X = Contrast(X, contrast_factor)
    
    if np.random.rand() < 0.2:
        scale_factor = np.random.uniform(0, 0.3)
        #print(f"Random perspective - {scale_factor}")
        X = Perspective(X, scale_factor)

    X = (255. - X) / 255.
    return X

def batch_generator_aug(X,Y, BATCH_SIZE):
    idx = 0
    while True:
        BatchX = X[idx:idx+BATCH_SIZE]
        BatchY = Y[idx:idx+BATCH_SIZE]

        BatchX = augmentation_process(BatchX[0])

        yield data_preparation_CTC([BatchX], BatchY)

        idx = (idx + BATCH_SIZE) % len(X)

def main():
    args = parse_arguments()
    
    XTrain, YTrain, XVal, YVal, XTest, YTest = [], [], [], [], [], []

    if args.corpus_name == "ToyPrimus" or args.corpus_name == "FP-Primus" or args.corpus_name == "CAPITAN":
        print("Loading train set:")
        XTrain, YTrain = load_data_testcase(PATH=f"{args.data_path}/train/")
        print("Loading val set:")
        XVal, YVal = load_data_testcase(PATH=f"{args.data_path}/val/")
        print("Loading test set:")
        XTest, YTest = load_data_testcase(PATH=f"{args.data_path}/test/")
    else:
        print("Loading MuRet train set:")
        if args.model_name == "SPAN_SYNTH":
            XTrain, YTrain = load_data_jsonMuret(PATH=f"{args.data_path}/train_daug", encoding=args.encoding)
        else:
            XTrain, YTrain = load_data_jsonMuret(PATH=f"{args.data_path}/train", encoding=args.encoding)

        print("Loading MuRet val set:")
        XVal, YVal = load_data_jsonMuret(PATH=f"{args.data_path}/val", encoding=args.encoding)
        print("Loading MuRet test set:")
        XTest, YTest = load_data_jsonMuret(PATH=f"{args.data_path}/test", encoding=args.encoding)

    w2i, i2w = check_and_retrieveVocabulary([YTrain, YVal, YTest], f"./vocab", f"{args.corpus_name}_{args.encoding}")
    
    ratio = 1

    for i in range(len(XTrain)):
        img = (255. - XTrain[i]) / 255.
        #img = XTrain[i]
        width = int(np.ceil(img.shape[1] * ratio))
        height = int(np.ceil(img.shape[0] * ratio))
        XTrain[i] = cv2.resize(img, (width, height))
        seq = []
        for symbol in YTrain[i]:
            seq.append(w2i[symbol])
        
        YTrain[i] = seq
    
    for i in range(len(XVal)):
        img = (255. - XVal[i]) / 255.
        width = int(np.ceil(img.shape[1] * ratio))
        height = int(np.ceil(img.shape[0] * ratio))
        XVal[i] = cv2.resize(img, (width, height))
        seq = []
        for symbol in YVal[i]:
            seq.append(w2i[symbol])
        
        YVal[i] = seq
    
    for i in range(len(XTest)):
        img = (255. - XTest[i]) / 255.
        width = int(np.ceil(img.shape[1] * ratio))
        height = int(np.ceil(img.shape[0] * ratio))
        XTest[i] = cv2.resize(img, (width, height))
        seq = []
        for symbol in YTest[i]:
            seq.append(w2i[symbol])
        
        YTest[i] = seq
    
    maxwidth = max([img.shape[1] for img in XTrain])
    maxheight = max([img.shape[0] for img in XTrain])

    print(maxwidth)
    print(maxheight)
    
    if args.model_name == "SPAN_TRANSFORMER" or args.model_name == "SPAN_TRANSFORMER_AUG":
        model, device = get_span_transformer_model(maxwidth=maxwidth, 
                                        maxheight=maxheight, 
                                        in_channels=1, 
                                        out_size=len(w2i), 
                                        encoder_weights=args.encoder_weights)
    else:
        model, device = get_span_recurrent_model(maxwidth=maxwidth, 
                                        maxheight=maxheight, 
                                        in_channels=1, 
                                        out_size=len(w2i), 
                                        encoder_weights=args.encoder_weights)

    print(f"Using {device} device")
    
    batch_gen = None
    if args.model_name == "SPAN_AUG" or args.model_name == "SPAN_TRANSFORMER_AUG":
        print("Using basic data augmentation generation")
        batch_gen = batch_generator_aug(XTrain, YTrain, args.batch_size)
    else:
        print("Using simple generator")
        batch_gen = batch_generator(XTrain, YTrain, args.batch_size)
    
    criterion = torch.nn.CTCLoss(blank=len(w2i)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Training with {len(XTrain)} samples")
    print(f"Validating with {len(XVal)} samples")
    print(f"Testing with {len(XTest)} samples")

    numsamples = len(XTrain)//args.batch_size

    bestSer = np.Inf
    bestTest = np.Inf
    patience = args.patience

    if not os.path.isdir(f"test_predictions/{args.corpus_name}/"):
        os.makedirs(f"test_predictions/{args.corpus_name}/")

    for epoch in range(5000):
        model.train()
        for mini_epoch in range(5):
            accum_loss = 0
            for _ in range(numsamples):
                
                optimizer.zero_grad()
                
                net_input, net_tar, input_len, tar_len = next(batch_gen)
                predictions = model(net_input.to(device))
                
                # From Conv2D output to sequential interpretation by making row concat
                predictions = predictions.permute(1,0,2)

                loss = criterion(predictions, net_tar.to(device), input_len.to(device), tar_len.to(device))

                loss.backward()
                optimizer.step()

                accum_loss += loss.item() * args.batch_size
            
            avg = accum_loss / numsamples
            print(f"Step {mini_epoch + 1} - Loss: {avg}")
            
        model.eval()
        if args.encoding == "sseq":
            SER_TRAIN, _, _ = test_sseq(model, XTrain, YTrain, i2w, device)
            SER_VAL, _, _ = test_sseq(model, XVal, YVal, i2w, device)
            SER_TEST, preds, gts = test_sseq(model, XTest, YTest, i2w, device)
        else:
            SER_TRAIN, _, _ = test_model(model, XTrain, YTrain, i2w, device)
            SER_VAL, _, _ = test_model(model, XVal, YVal, i2w, device)
            SER_TEST, preds, gts = test_model(model, XTest, YTest, i2w, device)


        if SER_VAL < bestSer:
            print("Validation SER improved - Saving weights")
            torch.save(model.state_dict(), f"models/weights/{args.model_name}_{args.corpus_name}.pt")
            torch.save(optimizer.state_dict(), f"models/optimizers/{args.model_name}_{args.corpus_name}.pt")
            bestSer = SER_VAL
            bestTest = SER_TEST
            patience = args.patience
            for i, pred in enumerate(preds):
                writeResults(args.corpus_name, i, pred, gts[i])
        elif epoch > args.warmup:
            patience -= 1
            if patience == 0:
                print(f"Training finished with best SER in test of {bestTest}")
                break

        print(f"EPOCH {epoch + 1} --- TRAIN SER {SER_TRAIN} | VAL SER {SER_VAL} | TEST SER {SER_TEST}")
        print(f"BEST TEST - {bestTest}")

if __name__=="__main__":
    main()

