# Text-Imputation
## Overview of the project
A machine learning based computational model for automatic sentence completion implemented in PyTorch. It is based on the concept of language models, and uses both past and future context to predict the word missing from the sentence at a fixed position.
(Original code forked from [PyTorch word level language modeling example](https://github.com/pytorch/examples/tree/master/word_language_model).

## Model Architecture
There are two types of models - a bidirectional LSTM based model and an attention augumneted bidirectional LSTM based model. Both utilize an RNN with LSTM cell to encode past and future words and then use a fully connected neural network to predict the missing word. In case of the model with attention, the fully-connected network also utilizes a weighted vector of all inputs words as input.

## Training details
The training scripts are in [train](https://github.com/kartik144/Text_Imputation/tree/master/train) directory. The `train_bidirectional.py` file can be used to train the bi-LSTM based model, and the `train_bi_attn.py` can be used to train the attention based model. The folder also contains scripts to train a language model and a reverse language model (that predicts previous words instead of future words), which are `train_LM.py` and `train_revLM.py` respectively.

The training scripts accept the following optional arguments: 

```
general arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --bptt BPTT        sequence length
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --decay DECAY      learning rate decay per epoch
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
  --threshold N      specify threshold such that words with frequency less 
                     than the threshold will be discarded

arguments pecific to train_bidirectional.py  
  --dict DICT        path to file where the dictionary would be saved

arguments pecific to train_bidirectional.py  
  --dict DICT        path to file where the dictionary would be saved
  --sen_length N     threshold to eliminate unneccary long sentences.

  ```

During training, if a keyboard interrupt (Ctrl-C) is received, training is stopped and the current model is evaluated against the test dataset.

With these training parameters, a variety of models can be trained. Some exaples are:
```
python3 train_LM.py --model LSTM --emsize 600 --nhid 600 --epochs 10 --cuda --data /data/Google-1B --batch_size 64 --save ../models/LM_g1b_77_batch64_em_600.pt --tied --log-interval 10000 --threshold 77
python3 train_revLM.py --model LSTM --emsize 600 --nhid 600 --epochs 10 --cuda --data /data/Google-1B --batch_size 80 --save ../models/revLM_g1b_77_batch80_em_600.pt --tied --log-interval 1000 --threshold 77
python3 train_bidirectional.py --model LSTM --emsize 600 --nhid 600 --epochs 10 --cuda --data /data/Google-1B --batch_size 64 --save ../models/model_g1b_77_batch64_em_600.pt --bptt 35 --log-interval 10000 --tied --threshold 77 --dict dict_g1b_77.pt
python3 train_bi_attn.py --model LSTM --emsize 600 --nhid 600 --epochs 20 --cuda --data /data/Google-1B --batch_size 64 --save ../models/model_attn_g1b-mini_1000_50.pt --sen_length 50 --log-interval 1000 --tied --threshold 3 --dict dict_attn_g1b-mini_3.pt
```
## Evaluation 
The models be used to predict the missing words using the `main.py` file. It uses all the models - both language models, bi-LSTM based model and attention based model to suggest words and then displays them for comparison. The attention based model also outputs attention weights which are visualized.

The `main.py`  accepts the following optional arguments:
```
  --data DATA             location of the data corpus
  --model_left MODEL      path to saved model of language model
  --model_right MODEL     path to saved model of reverse language model
  --model_bi              path to saved model of bi-LSTM based model
  --model_attn            path to saved model of attention based model
  --dict                  path to dictionary used by language model, rev language model and bi-LSTM based model
  --dict_attn             path to dictionary used by attention based model
  --sen_length N          threshold to pad sentences for attention based model.
  --file                  path to file if the inputs are stored in a file
  --N n                   number of words to be suggested by each model
  --seed SEED             random seed
  --cuda                  use CUDA
```
The evailuation file takes input from both - standard input and file - to output the suggestions for words which could be filled in the blank.
