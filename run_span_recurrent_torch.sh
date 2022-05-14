#!/bin/bash

python train_recurrent_span.py --data_path Data/SEILS --model_name SPAN_AUG --corpus_name SEILS --encoding sseq --batch_size 1 --patience 10
