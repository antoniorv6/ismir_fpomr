from FPHR.FPHR import get_fphr_model
from data_load import load_data_jsonMuret_im2s, load_data_testcase_im2s
from utils import levenshtein, parse_arguments, check_and_retrieveVocabulary
import numpy as np
import cv2
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random

def batch_confection_fphr(BatchX, BatchY):
    
    max_img_width = max([img.shape[1] for img in BatchX])
    max_img_height = max([img.shape[0] for img in BatchX])
    max_len_target = max([len(seq) for seq in BatchY])

    encoder_input = np.zeros(shape=[len(BatchX), 3, max_img_height, max_img_width], dtype=np.float32)
    decoder_input = np.zeros((len(BatchY), max_len_target), dtype=float)
    decoder_output = np.zeros((len(BatchY), max_len_target), dtype=float)

    for i, img in enumerate(BatchX):
        encoder_input[i, 0, 0:img.shape[0], 0:img.shape[1]] = img
        encoder_input[i, 1, 0:img.shape[0], 0:img.shape[1]] = img
        encoder_input[i, 2, 0:img.shape[0], 0:img.shape[1]] = img

    for i, seq in enumerate(BatchY):
       for j, char in enumerate(seq):
           decoder_input[i][j] = char
           if j > 0:
               decoder_output[i][j-1]= char
            
    encoder_input = torch.tensor(encoder_input, dtype=torch.float)
    decoder_input = torch.tensor(decoder_input, dtype=torch.float)
    decoder_output = torch.tensor(decoder_output, dtype=torch.long)

    return encoder_input, decoder_input, decoder_output

def batch_generator(X, Y, BATCH_SIZE, device):
    index = 0
    while True:
        
        BatchX = X[index:index+BATCH_SIZE]
        BatchY = Y[index:index+BATCH_SIZE]

        #BatchX, BatchY = set_synth_mix_batch(BatchX, BatchY, data_generator, w2i)

        encoder_input = None 
        decoder_input = None 
        gt = None

        encoder_input, decoder_input, gt = batch_confection_fphr(BatchX, BatchY)

        yield encoder_input.to(device), decoder_input.to(device), gt.to(device)

        index = (index + BATCH_SIZE) % len(X)


def test_model(model, X, Y, w2i, i2w, device, maxlen):
    with torch.no_grad():
            acc_ed_dist = 0
            acc_len = 0
            randsample = random.randint(0, len(X))
            print(f"Showing validation sample {randsample}")
            for i, sample in enumerate(X):
                decoded = [w2i['<sos>']]
                sequence = ['<sos>']
                memory = None

                for _ in range(maxlen):
                    encoder_input = None
                    decoder_input = None
                    logits = None
                    encoder_input, decoder_input, _ = batch_confection_fphr([sample], [decoded])

                    if memory is None:
                        memory = model.forward_encoder(encoder_input.to(device))
                    
                    logits = model.forward_decoder(decoder_input.to(device), memory)
                    
                    prediction = torch.argmax(logits[-1:])
                    pred_char= i2w[prediction.item()]
                    decoded.append(prediction)
                    sequence.append(pred_char)
                    if pred_char == '<eos>':
                        break

                ground_truth = [i2w[char] for char in Y[i]]

                if i == randsample:
                    print(f"Prediction: {sequence}")
                    print(f"True: {ground_truth}")
            
                acc_ed_dist += levenshtein(decoded, ground_truth)
                acc_len += len(ground_truth)
    
    ser = 100.*acc_ed_dist / acc_len
    
    return ser



def main():
    args = parse_arguments()

    #img, json = DataAugmentationGenerator.generateNewImageFromListByBoundingBoxesRandomSelectionAuto(f"/workspace/experiments/Data/SEILS/train/", 1, False, False, 0.2)
    
    XTrain, YTrain, XVal, YVal, XTest, YTest = [], [], [], [], [], []

    if args.corpus_name == "ToyPrimus" or args.corpus_name == "FP-Primus" or args.corpus_name == "CAPITAN":
        print("Loading train set:")
        XTrain, YTrain = load_data_testcase_im2s(PATH=f"{args.data_path}/train/")
        print("Loading val set:")
        XVal, YVal = load_data_testcase_im2s(PATH=f"{args.data_path}/val/")
        print("Loading test set:")
        XTest, YTest = load_data_testcase_im2s(PATH=f"{args.data_path}/test/")
    else:
        print("Loading MuRet train set:")
        if args.model_name == "SPAN_SYNTH":
            XTrain, YTrain = load_data_jsonMuret_im2s(PATH=f"{args.data_path}/train_daug")
        else:
            XTrain, YTrain = load_data_jsonMuret_im2s(PATH=f"{args.data_path}/train")

        print("Loading MuRet val set:")
        XVal, YVal = load_data_jsonMuret_im2s(PATH=f"{args.data_path}/val")
        print("Loading MuRet test set:")
        XTest, YTest = load_data_jsonMuret_im2s(PATH=f"{args.data_path}/test")

    w2i, i2w = check_and_retrieveVocabulary([YTrain, YVal, YTest], f"./vocab", f"{args.corpus_name}_im2s")
    
    ratio = 1

    for i in range(len(XTrain)):
        img = (255. - XTrain[i]) / 255.
        img = XTrain[i]
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
    maxlength = max([len(seq) for seq in YTrain])

    model= None
    device = None
    optimizer = None

    model, device = get_fphr_model(maxwidth, maxheight, maxlength, len(w2i))
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    print(f"Using Torch with {device}")
    print(f"Training with {len(XTrain)} samples")
    print(f"Validating with {len(XVal)} samples")
    print(f"Testing with {len(XTest)} samples")


    train_generator = batch_generator(X=XTrain, Y=YTrain, BATCH_SIZE=args.batch_size, device=device)

    numsamples = len(XTrain)//args.batch_size
    bestSer = 10000

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    for epoch in range(5000):
        model.train()
        running_avg = 0
        for mini_epoch in range(5):
            accum_loss = []
            for _ in range(numsamples):
                
                optimizer.zero_grad()
                enc_in, dec_in, gt = next(train_generator)
                optimizer.zero_grad()

                output = None
                
                output = model(enc_in, dec_in)

                loss = 0

                output = output.permute(1,2,0) # (B, num_classes, Sy)
                loss = criterion(output, gt)
                accum_loss.append(loss.item())
                loss.backward()
                optimizer.step()
        
            running_avg = np.convolve(accum_loss, np.ones(len(accum_loss))/len(accum_loss), mode='valid')[0]
            print(f"Step {mini_epoch + 1} - Loss: {running_avg}")

        model.eval()
        SER_TRAIN = test_model(model, XTrain, YTrain, w2i, i2w, device, maxlength)
        SER_VAL = test_model(model, XVal, YVal, w2i, i2w, device, maxlength)
        SER_TEST = test_model(model, XTest, YTest, w2i, i2w, device, maxlength)

        if SER_VAL < bestSer:
            print("Validation SER improved - Saving weights")
            torch.save(model.state_dict(), f"models/weights/{args.model_name}_{args.corpus_name}.pt")
            torch.save(optimizer.state_dict(), f"models/optimizers/{args.model_name}_{args.corpus_name}.pt")
            bestSer = SER_VAL

        print(f'SUPER EPOCH {epoch + 1} | Validation CER {SER_VAL} | Test CER {SER_TEST}')


if __name__=="__main__":
    main()