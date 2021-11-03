from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound, JsonResponse
from rest_framework.views import APIView

from app.transformerModel import TransformerModel

import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torchtext.legacy.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from torchtext.legacy.datasets import TranslationDataset

# Create your views here.

def home(request):
    return render(request, 'app/translate.html')

def view_results(request):
    # Submit prediction and show all
    # data = {"dataset": PredResults.objects.all()}
    return render(request, "app/results.html", {})

def machine_translation(request):
    if request.POST.get('action') == 'post':
        english_input = request.POST.get('raw_english')
        answer = process_translate(english_input.split(' '))
        return JsonResponse({'answer': answer})

def process_translate(input):
    print(input)
    def filter_len(example):
        return len(example.src) <= MAX_LENGTH and len(example.trg) <= MAX_LENGTH

    def translate(sent):
        NMTmodel.eval()
        with torch.no_grad():
            src = [SOS_token] + [SRC.vocab.stoi[w] for w in sent] + [EOS_token]
            src = torch.tensor(src, dtype=torch.long)

            wordidx = NMTmodel.inference(src, MAX_LENGTH)
            words = []
            for idx in wordidx:
                words.append(TRG.vocab.itos[idx])

        return words

    MAX_LENGTH = 64
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYER = 6

    SRC = Field(init_token='<sos>', eos_token='<eos>', lower=True)
    TRG = Field(init_token='<sos>', eos_token='<eos>', lower=True)

    train_data = TranslationDataset(r'app/models/document/train', exts=('.src', '.tgt'),
                                    filter_pred=filter_len, fields=(SRC, TRG))

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    INPUT_SIZE = len(SRC.vocab)
    OUTPUT_SIZE = len(TRG.vocab)
    SOS_token = TRG.vocab.stoi['<sos>']
    EOS_token = TRG.vocab.stoi['<eos>']
    PAD_token = TRG.vocab.stoi['<pad>']

    NMTmodel = TransformerModel(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYER,
                                MAX_LENGTH, PAD_token, SOS_token, EOS_token)

    checkpoint = torch.load("app/models/NMT.pt", map_location=torch.device('cpu'))
    NMTmodel.load_state_dict(checkpoint['model_state'])

    answer = translate(input)
    print(answer)

    return answer
