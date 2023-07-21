import numpy as np
from scipy import interpolate

def interp(x, y, num, ruls, rul_factor):
    ynew = []
    for i in range(y.shape[1]):
        f = interpolate.interp1d(x, y[:, i], kind='linear')
        x_new = np.linspace(x[0], x[-1], num)
        ytmp = f(x_new)
        ynew.append(ytmp)
    ynew = np.vstack(ynew)
    ynew = ynew.T
    newruls = [i for i in range(1, ynew.shape[0] + 1)]
    newruls.reverse()
    newruls = np.array(newruls).astype(float)
    # remove rul_factor
    # newruls /= rul_factor
    new_right_end_value = ruls[-1] * (num / len(x))
    for i in range(len(newruls)):
        newruls[i] += new_right_end_value
    return ynew, newruls


def data_aug(feas, ruls, scale_ratios, rul_factor):
    augmented_feas, augmented_ruls = [], []
    for scaleratio in scale_ratios:
        if int(scaleratio * feas.shape[0]) <= 100:
            continue
        augmented, rul = interp([i for i in range(feas.shape[0])], feas,
                                int(scaleratio * feas.shape[0]), ruls,
                                rul_factor)
        augmented_feas.append(augmented)
        augmented_ruls.append(rul)
    return augmented_feas, augmented_ruls

def split_seq(fullseq, rul_labels, seqlen, seqnum):
    if isinstance(fullseq, list):
        all_fea, all_lbls = [], []
        for seqidx in range(len(fullseq)):
            tmp_all_fea = np.lib.stride_tricks.sliding_window_view(
                fullseq[seqidx], (seqlen, fullseq[seqidx].shape[1]))

            tmp_all_fea = tmp_all_fea.squeeze()
            tmp_lbls = rul_labels[seqidx][seqlen - 1:]
            tmp_fullseqlen = rul_labels[seqidx][0]
            fullseqlens = np.array(
                [tmp_fullseqlen for _ in range(tmp_all_fea.shape[0])])
            # print(tmp_lbls.shape, fullseqlens.shape, fullseq[seqidx].shape, rul_labels[seqidx].shape)
            lbls = np.vstack((tmp_lbls, fullseqlens)).T
            if seqnum <= tmp_all_fea.shape[0]:
                all_fea.append(tmp_all_fea[:seqnum])
                all_lbls.append(lbls[:seqnum])
            else:
                all_fea.append(tmp_all_fea)
                all_lbls.append(lbls)
        all_fea = np.vstack(all_fea)
        all_lbls = np.vstack(all_lbls)
        all_lbls = all_lbls.astype(int)
        return all_fea, all_lbls
    else:
        all_fea = np.lib.stride_tricks.sliding_window_view(
            fullseq, (seqlen, fullseq.shape[1]))
        all_fea = all_fea.squeeze()
        # ruls = rul_labels[seqlen-1:]
        fullseqlen = rul_labels[0]
        lbls = rul_labels[seqlen - 1:]
        fullseqlens = np.array([fullseqlen for _ in range(all_fea.shape[0])])
        lbls = np.vstack((lbls, fullseqlens)).T
        lbls = lbls.astype(int)
        if seqnum <= all_fea.shape[0]:
            return all_fea[:seqnum], lbls[:seqnum]
        else:
            return all_fea, lbls

def get_train_test_val(series_len=100,
                       rul_factor=3000,
                       dataset_name='train',
                       seqnum=500,
                       data_aug_scale_ratios=None):

    metadata = np.load('ne_data/meta_data.npy', allow_pickle=True)
    if dataset_name == 'train':
        set = metadata[0]
    elif dataset_name == 'valid':
        set = metadata[1]
    elif dataset_name == 'trainvalid':
        set = metadata[0] + metadata[1]
    else:
        set = metadata[2]

    allseqs, allruls, batteryids = [], [], []
    batteryid = 0
    for batteryname in set:
        seqname = 'ne_data/' + batteryname + '.npy'
        lblname = 'ne_data/' + batteryname + '_rul.npy'
        seq = np.load(seqname, allow_pickle=True)
        lbls = np.load(lblname, allow_pickle=True)
        origin_dq = np.array([seq[0][0] for i in range(seq.shape[0])]).reshape(
            (-1, 1))
        seq = np.hstack((seq, origin_dq))

        if data_aug_scale_ratios is not None:
            seqs, ruls = data_aug(seq, lbls, data_aug_scale_ratios, rul_factor)
            feas, ruls = split_seq(seqs, ruls, series_len, seqnum)
        else:
            feas, ruls = split_seq(seq, lbls, series_len, seqnum)

        allseqs.append(feas)
        allruls.append(ruls)
        batteryids += [batteryid for _ in range(feas.shape[0])]
        batteryid += 1
    batteryids = np.array(batteryids).reshape((-1, 1))
    allruls = np.vstack(allruls)
    allruls = np.hstack((allruls, batteryids))
    allseqs = np.vstack(allseqs)
    print("origin data:", allruls.shape, allseqs.shape)
    return allseqs, allruls