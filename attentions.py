import os, sys
import numpy as np
import torch
from path import Path
import pickle as pkl
import hashlib

from visual2d import set_data, show_img, show_attns, to_points
from visual3d import show_points

###################
#--- for Models ---
from transformers import BertTokenizer, BertModel

#--- for Input
MODELS='''
bert-base-chinese
bert-base-multilingual-cased
'''.strip().split('\n')

name=MODELS[1]

TEXTS='''
hello, how are you ?
猎人的狗被狼咬死了
'''.strip().split('\n')

text=TEXTS[1]

#--- for TSNE ---
# from sklearn import manifold
# tsne=manifold.TSNE(n_components=3, init='pca', random_state=8888)
tsne=None 

#--- others ---
fp_cache=Path(f'cache/{hashlib.md5((name+":"+text).encode()).hexdigest()}.pkl')
###################

def load_model(name):
    tokenizer = BertTokenizer.from_pretrained(name)
    model = BertModel.from_pretrained(name)
    return tokenizer, model

def tsne_to3d(feats):
    # flattern
    embed_size=feats.shape[-1]
    feats_flat=feats.reshape(-1, embed_size)
    
    #reduce dim
    feats_tsne=tsne.fit_transform(feats_flat)
    
    #normalize
    feats_min, feats_max=feats_tsne.min(0), feats_tsne.max(0)
    feats_norm = (feats_tsne - feats_min)/(feats_max - feats_min)
    
    return feats_norm

def mean_to3d(feats):
    # flattern
    embed_size=feats.shape[-1]
    feats_flat=feats.reshape(-1, embed_size)
    
    #slice
    block = embed_size // 3
    extra = embed_size % 3
    if extra==0:
        feats_remove=feats_flat
    else:
        feats_remove=feats_flat[:,:-extra]
        
    feats_reshape = feats_remove.reshape(-1, 3, block)
    feats_norm = feats_reshape.mean(axis=-1)
    
    return feats_norm

def do_tsne(): # not used
    # feats=output.attentions[3].numpy()
    # feats=output.hidden_states[3].numpy()
    feats=output.last_hidden_state.numpy()

    feats_norm1=tsne_to3d(feats)
    feats_norm2=mean_to3d(feats)

    return feats_norm1.shape, feats_norm2.shape

def cache_exists():
    return fp_cache.exists()

def load_cache():
    with open(fp_cache, 'rb') as fin:
        obj=pkl.load(fin)
    return obj

def make_cache(obj):
    if not fp_cache.parent.isdir():
        os.mkdir(fp_cache.parent)
        
    with open(fp_cache, 'wb') as fout:
        pkl.dump(obj, fout)

def load_data(force=0):
    if cache_exists() and not force: return load_cache()

    tokenizer, model = load_model(name)

    encoded_input = tokenizer(text.lower(), return_tensors='pt')

    vocabs = np.array([i for i in tokenizer.vocab])

    input_ids = encoded_input.input_ids[0].numpy()
    text_tok = ' '.join(vocabs[input_ids])

    with torch.no_grad():
        output = model(**encoded_input
            , output_hidden_states=True
            , output_attentions=True
            # , add_cross_attention=True
            # , use_cache=True
        )

    data = output, vocabs, input_ids, text_tok
    make_cache(data)
    return data

if __name__=='__main__':
    output, vocabs, input_ids, text_tok = load_data()

    labels=[]
    for i, token_id in enumerate(input_ids):
        print(i, ':', vocabs[token_id])
        labels.append(vocabs[token_id])

    # setting data
    set_data(output, vocabs, input_ids)

    show_img(0, 0)

    points = to_points(0)
    print(points.shape)
    show_points(points, 1)


