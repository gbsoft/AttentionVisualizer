import os, sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

plt.figure_size=(10, 10)
plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 6

### --- Global Vars Begin
#model output
output=None 

#tokenizer vocab
vocabs=None 

#tokened input ids
input_ids=None 

#other auto calc vars
labels=None
seq_len=None
### --- Global Vars End

def set_data(_output, _vocabs, _input_ids):
    global output, vocabs, input_ids, labels, seq_len
    output = _output
    vocabs = _vocabs
    input_ids = _input_ids

    labels = [vocabs[i] for i in input_ids]
    seq_len = len(input_ids)

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

def show_img_one_heads_of_one_layer(l=0, h=0):
    # head h of layer l
    layer_idx=l
    head_idx=h
    img=output.attentions[layer_idx].numpy()[0,head_idx,:,:]
    plt.imshow(img)
    plt.title(f'L{layer_idx}/H{head_idx}')
    plt.xticks(range(len(labels)), labels, rotation=0)
    plt.yticks(range(len(labels)), labels, rotation=0)
    plt.colorbar()

def show_img_one_head_of_all_layer(h=0):
    # head h 2 of layer 0-11
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
            _plt.set_ylabel('Layers')
            _plt.set_title(f'H{head_idx}')
            _plt.set_xticks(range(len(labels)), labels, rotation=0)
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
    plt.title(f'Attentions of head {head}')
    
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

def show_img(l=0, h=0):
    if l in range(12) and h in range(12):
        show_img_one_heads_of_one_layer(l, h)
    elif l in range(12) and h=='*':
        show_img_all_heads_of_one_layer(l)
    elif l=='*' and h in range(12):
        show_img_one_head_of_all_layer(h)
    elif l=='*' and h=='mean':
        show_img_mean_head_of_all_layer()
    else:
        assert 0
    plt.show()

def show_attns(h):
    if type(h) is list and len(h)==4:
        show_2d_attns_of_4heads_for_all_layers(h)
    elif h in range(12) or h in 'min max mean'.split():
        show_2d_attns_of_one_head_for_all_layers(h)
    else:
        assert 0
    plt.show()

def to_points(h):
    '''h in range(12) or min max mean'''
    return gen_4d_attns_arr_of_one_head_for_all_layers(h)

if __name__=='__main__':
    import pickle as pkl
    with open('cache/3d5cc864787ed4d8c0adff28baf659c8.pkl', 'rb') as fin:
        data=pkl.load(fin)

    set_data(data[0], data[1], data[2])

    # show_img(0, 0)
    # show_img(0, '*')
    # show_img('*', 0)
    # show_img('*', 'mean')

    show_attns([0,2,4,6])
    # show_attns(1)
    # show_attns('mean')