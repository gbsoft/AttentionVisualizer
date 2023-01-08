import os, sys
import numpy as np
import torch
from path import Path
import pickle as pkl
import hashlib
from visual3d import show_point_cloud

###################
#--- for PLOT ---
import matplotlib
import matplotlib.pyplot as plt

plt.figure_size=(10, 10)
plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 6

#--- for TSNE ---
# from sklearn import manifold
# tsne=manifold.TSNE(n_components=3, init='pca', random_state=8888)
tsne=None 

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

def save(feats):
    print('saving: ', feats.shape)
    np.savetxt('/tmp/point_cloud.txt', feats)

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

def do_tsne(): # not used
    # feats=output.attentions[3].numpy()
    # feats=output.hidden_states[3].numpy()
    feats=output.last_hidden_state.numpy()

    feats.shape

    feats_norm1=tsne_to3d(feats)
    feats_norm2=mean_to3d(feats)

    feats_norm1.shape, feats_norm2.shape

    save(tsne_to3d(output.attentions[0].numpy()))

output=None

def show_img_all_heads_of_one_layer(l=0):
    # head 0-11 of layer l
    layer_idx=l
    n_rows, n_cols=12 // 4, 4
    fig, ax = plt.subplots(n_rows, n_cols)
    plt.tight_layout()
    for i in range(n_rows):
        for j in range(n_cols):
            head_idx=i*n_cols + j
            img=output.attentions[layer_idx].numpy()[0,head_idx,:,:]
            ax[i][j].imshow(img)
            ax[i][j].set_title(f'L{layer_idx}/H{head_idx}')

def show_img_one_head_of_all_layer(h=0):
    # head 2 of layer 0-11
    head_idx=h
    n_rows, n_cols=12 // 4, 4
    fig, ax = plt.subplots(n_rows, n_cols)
    plt.tight_layout()
    for i in range(n_rows):
        for j in range(n_cols):
            layer_idx=i*n_cols + j
            img=output.attentions[layer_idx].numpy()[0,head_idx,:,:]
            ax[i][j].imshow(img)
            ax[i][j].set_title(f'L{layer_idx}/H{head_idx}')

def show_img_mean_head_of_all_layer():
    # head mean of layer 0-11
    n_rows, n_cols=12 // 4, 4
    fig, ax = plt.subplots(n_rows, n_cols)
    plt.tight_layout()
    for i in range(n_rows):
        for j in range(n_cols):
            layer_idx=i*n_cols + j
            img=output.attentions[layer_idx].numpy().squeeze(axis=0).mean(axis=0)
            ax[i][j].imshow(img)
            ax[i][j].set_title(f'L{layer_idx}/Hm')

def show_2d_attns_of_4heads_for_all_layers(head_idx_list=[0,1,2,3]):
    if not np.all(np.array(head_idx_list)>=0): return
    if not np.all(np.array(head_idx_list)<12): return

    n_rows, n_cols=2, 2 #12 // 4, 4
    fig, ax = plt.subplots(n_rows, n_cols)
    plt.tight_layout()
    for i in range(n_rows):
        for j in range(n_cols):
            _plt=ax[i][j]
            head_idx=head_idx_list[i*n_cols + j]
            # _plt.set_xlabel('Tokens')
            # _plt.set_ylabel('Layers')
            # _plt.set_title(f'H{head_idx}')
            _plt.set_ylabel(f'H{head_idx}')
            _plt.set_xticks(range(len(labels)), labels, rotation=90)
            # _plt.set_yticks(range(12+1), rotation=0)

            for layer_idx in range(12):
                attn_matrix=output.attentions[layer_idx].numpy().squeeze(axis=0)[head_idx,:,:]
                for j1 in range(seq_len):
                    for j2 in range(seq_len):
                        _plt.plot([j1, j2], [layer_idx+1, layer_idx], linewidth=attn_matrix[j1][j2])

def show_2d_attns_of_one_head_for_all_layers(head=0):
    # head: head idx or min, max, mean method
    n_rows, n_cols=12 // 4, 4
    plt.xlabel('Tokens')
    plt.ylabel('Layers')
    # plt.title('Attentions')
    ticks, txts=plt.xticks(range(len(labels)), labels, rotation=0)
    ticks, txts=plt.yticks(range(12+1), rotation=0)

    for layer_idx in range(12):
        attn_matrix=output.attentions[layer_idx].numpy().squeeze(axis=0)
        if type(head) is int:
            attn_matrix=attn_matrix[head,:,:]
        elif head=='mean':
            attn_matrix=attn_matrix.mean(axis=0)
        elif head=='max':
            attn_matrix=attn_matrix.max(axis=0)
        elif head=='min':
            attn_matrix=attn_matrix.min(axis=0)
        for j1 in range(seq_len):
            for j2 in range(seq_len):
                plt.plot([j1, j2], [layer_idx+1, layer_idx], linewidth=attn_matrix[j1][j2])

def gen_4d_attns_arr_of_one_head_for_all_layers(head=0):
    arr=[]
    for layer_idx in range(12):
        attn_matrix=output.attentions[layer_idx].numpy().squeeze(axis=0)
        if type(head) is int:
            attn_matrix=attn_matrix[head,:,:]
        elif head=='mean':
            attn_matrix=attn_matrix.mean(axis=0)
        elif head=='max':
            attn_matrix=attn_matrix.max(axis=0)
        elif head=='min':
            attn_matrix=attn_matrix.min(axis=0)
        for j1 in range(seq_len):
            for j2 in range(seq_len):
                arr.append([j1, j2, layer_idx+1, attn_matrix[j1][j2]]) # x(j1), y(j2), z(layer), v
    return np.asarray(arr, dtype='float32')

if __name__=='__main__':
    output, vocabs, input_ids, text_tok = load_data()
    seq_len = len(input_ids)

    labels=[]
    for i, token_id in enumerate(input_ids):
        print(i, ':', vocabs[token_id])
        labels.append(vocabs[token_id])

    arr = gen_4d_attns_arr_of_one_head_for_all_layers()
    print(arr.shape)
    show_point_cloud(arr, 0)

    for i in range(0):
        # show_img_all_heads_of_one_layer(i)

        # show_img_one_head_of_all_layer(i)
        # show_img_one_head_of_all_layer(i+1)
        
        # show_img_mean_head_of_all_layer()

        show_2d_attns_of_4heads_for_all_layers()

        # show_2d_attns_of_one_head_for_all_layers()
        plt.show()

