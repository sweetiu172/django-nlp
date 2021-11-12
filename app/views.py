from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound, JsonResponse
# from rest_framework.views import APIView

# from app.transformerModel import TransformerModel

# import torch
# import torch.nn as nn
# from torch import optim
# from torch.nn.utils import clip_grad_norm_
# from torchtext.legacy.data import Field, BucketIterator
# from torchtext.data.metrics import bleu_score
# from torchtext.legacy.datasets import TranslationDataset

# Create your views here.


# from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import MarianMTModel, MarianTokenizer
import torch

if torch.cuda.is_available():       
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-en-vi-small")
# tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-en-vi-small")
# model.to(device)
# model.eval()

from transformers import MarianMTModel, MarianTokenizer
model_path = ""
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path).to(device)
model.eval()




def home(request):
    return render(request, 'app/translate.html')



def view_results(request):
    # Submit prediction and show all
    # data = {"dataset": PredResults.objects.all()}
    return render(request, "app/results.html", {})

def machine_translation(request):
    if request.POST.get('action') == 'post':
        english_input = request.POST.get('raw_english')
        print(english_input)
        # answer = process_translate(english_input.split(' '))
        tokenized_text = tokenizer.encode(english_input, return_tensors="pt").to(device)
        summary_ids = model.generate(
                    tokenized_text,
                    max_length=256, 
                    num_beams=5,
                    repetition_penalty=2.5, 
                    length_penalty=1.0, 
                    early_stopping=True
                )
        answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print("Answer:" + answer)
        return JsonResponse({'answer': answer})

# def process_translate(input):
#     print(input)
#     def filter_len(example):
#         return len(example.src) <= MAX_LENGTH and len(example.trg) <= MAX_LENGTH

#     def translate(sent):
#         NMTmodel.eval()
#         with torch.no_grad():
#             src = [SOS_token] + [SRC.vocab.stoi[w] for w in sent] + [EOS_token]
#             src = torch.tensor(src, dtype=torch.long)

#             wordidx = NMTmodel.inference(src, MAX_LENGTH)
#             words = []
#             for idx in wordidx:
#                 words.append(TRG.vocab.itos[idx])

#         return words

#     MAX_LENGTH = 64
#     EMBED_SIZE = 256
#     HIDDEN_SIZE = 512
#     NUM_LAYER = 6

#     SRC = Field(init_token='<sos>', eos_token='<eos>', lower=True)
#     TRG = Field(init_token='<sos>', eos_token='<eos>', lower=True)

#     train_data = TranslationDataset(r'app/models/document/train', exts=('.src', '.tgt'),
#                                     filter_pred=filter_len, fields=(SRC, TRG))

#     SRC.build_vocab(train_data, min_freq=2)
#     TRG.build_vocab(train_data, min_freq=2)

#     INPUT_SIZE = len(SRC.vocab)
#     OUTPUT_SIZE = len(TRG.vocab)
#     SOS_token = TRG.vocab.stoi['<sos>']
#     EOS_token = TRG.vocab.stoi['<eos>']
#     PAD_token = TRG.vocab.stoi['<pad>']

#     NMTmodel = TransformerModel(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYER,
#                                 MAX_LENGTH, PAD_token, SOS_token, EOS_token)

#     checkpoint = torch.load("app/models/NMT.pt", map_location=torch.device('cpu'))
#     NMTmodel.load_state_dict(checkpoint['model_state'])

#     answer = translate(input)
#     print(answer)

#     return answer
