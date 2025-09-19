# %%
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence


# %%
def batch_padding_collate_fn(batch, MAX_LEN_SEQ, global_padd=False, emb_dim=128):

    batch_features, batch_labels, batch_ids = zip(*batch)
    batch_features = np.array(batch_features)
    unpadded_seqs = []
    batch_protA_lenghts = []
    batch_protB_lenghts = []

    for i in range(0, batch_features.shape[0]):
        # append proteins pair
        unpadded_seqs.append(torch.from_numpy(batch_features[i][0]))
        unpadded_seqs.append(torch.from_numpy(batch_features[i][1]))
        batch_protA_lenghts.append(len(batch_features[i][0]))
        batch_protB_lenghts.append(len(batch_features[i][1]))

    if global_padd:
        # pad proteins embedings according to the largest in the entire dataset
        unpadded_seqs.append(torch.zeros(MAX_LEN_SEQ, emb_dim))
        padded_seq = pad_sequence(unpadded_seqs, batch_first=True)
        padded_seq = padded_seq[:-1]
    else:
        # pad proteins embedings according to the largest in all proteins in the batch
        padded_seq = pad_sequence(unpadded_seqs, batch_first=True)

    # create new tensor of shape N * 2(protein pair) * L(longest seq) * Emb dim
    s = padded_seq.shape
    padded_pairs = torch.empty((batch_features.shape[0], 2, s[1], s[2]))
    # redo the pairs by skiping 1 protein
    padded_pairs[:, 0] = padded_seq[0::2]
    padded_pairs[:, 1] = padded_seq[1::2]

    return padded_pairs, torch.FloatTensor(batch_labels), \
           torch.FloatTensor(batch_protA_lenghts), torch.FloatTensor(batch_protB_lenghts), \
           np.array(batch_ids)


# %%
def PROMEOS_collate_fn(batch, max_size_set, emb_size=64, pytorch_pad=False):

    batch_features, batch_labels, batch_ids = zip(*batch)
    # batch_features = np.array((batch_features), dtype=object)
    unpadded_seqs = []
    padd_mask_pytorch = torch.ones((len(batch_features), 2, max_size_set), dtype=torch.bool)
    padd_mask = torch.empty((len(batch_features), 2, max_size_set, max_size_set))
    protA_seq_list = []
    protB_seq_list = []

    for i in range(0, len(batch_features)):
        protA = batch_features[i][0][0]  # 0번: protA 0번: go emb
        protA_seq = batch_features[i][0][1]  # 0번: protA 1번: seq
        protB = batch_features[i][1][0]  # 0번: protB 0번: go emb
        protB_seq = batch_features[i][1][1]  # 0번: protB 1번: seq

        unpadded_seqs.append(torch.FloatTensor(protA))
        unpadded_seqs.append(torch.FloatTensor(protB))
        protA_seq_list.append(torch.FloatTensor(protA_seq))
        protB_seq_list.append(torch.FloatTensor(protB_seq))

        # mask those positions which are not padding
        padd_mask_pytorch[i][0][0:len(protA)] = False
        padd_mask_pytorch[i][1][0:len(protB)] = False

    # pad proteins embedings according to the largest in the entire dataset
    unpadded_seqs.append(torch.zeros(max_size_set, emb_size))
    padded_seq = pad_sequence(unpadded_seqs, batch_first=True)[:-1]

    # create new tensor of shape N * 2(protein pair) * L(longest seq) * Emb dim
    s = padded_seq.shape
    padded_pairs = torch.empty((len(batch_features), 2, s[1], s[2]))
    # redo the pairs by jumping 1 protein
    padded_pairs[:, 0] = padded_seq[0::2]
    padded_pairs[:, 1] = padded_seq[1::2]

    for i in range(0, padded_pairs.shape[0]):
        padd_mask[i][0] = get_padd_mask_transformer(padded_pairs[i][0])
        padd_mask[i][1] = get_padd_mask_transformer(padded_pairs[i][1])

    if pytorch_pad:
        return padded_pairs, torch.FloatTensor(batch_labels), padd_mask_pytorch, list(batch_ids)

    return padded_pairs, torch.FloatTensor(batch_labels), padd_mask, list(batch_ids), protA_seq_list, protB_seq_list


# %%
def get_padd_mask_transformer(prot):
    """Gets an embedded protein and returns its padding mask
    Args:
        proteinA (numpy): numpy of shape (seqLen, emb_dim)

    Returns:
    numpy: matrix of size (seqLen, seqLen)
    """
    mask = (prot.numpy() != 0)
    mask = np.matmul(mask, mask.T)
    return torch.from_numpy(mask)


# %%
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# %%
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# %%
def print_status(epoch, epoch_mins, epoch_secs, train_loss, train_acc, valid_loss, valid_acc, roc_train, roc_val,
                 optimizer):
    print('=========== valid ===========')
    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s',
          f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%',
          f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%',
          f'\t Roc Train: {roc_train:.3f}', f'\t Roc Valid: {roc_val:.3f}',
          ",  ", optimizer.param_groups[0]['lr'], "--LR", end='\r')
    print('=========== valid ===========')


# %%
def write_scalars_tensorboard(writer, train_loss, valid_loss, train_acc, valid_acc, epoch):
    writer.add_scalars('Loss', {'train': train_loss, 'valid': valid_loss}, epoch)
    writer.add_scalars('Acc', {'train': train_acc, 'valid': valid_acc}, epoch)


