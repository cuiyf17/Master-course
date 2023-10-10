import copy
import string
import numpy as np
import torch
import torch.nn as nn
from batch import Batch

MASK = "_"
PAD = "#"
pad_token = 0
mask_token = 1


def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


def pad_words(words, PAD):
    word_lens = [len(word) for word in words]
    max_len = max(word_lens)
    words_padded = [word + [PAD] * (max_len - word_lens[i]) for i, word in enumerate(words)]

    return words_padded


def evaluate_acc(model, vocab, dev_data, device):
    was_training = model.training
    model.eval()

    # no_grad() signals backend to throw away all gradients
    total_correct_guess = 0
    total_n = 0
    with torch.no_grad():
        for batch in create_words_batch(dev_data, vocab, mini_batch=30, shuffle=False, device=device):
            x_mask = batch.src.unsqueeze(1).to(model.device)
            output = model(batch.src, x_mask)
            generator_mask = torch.zeros(batch.src.shape[0], len(vocab.char2id), device=model.device)
            generator_mask = generator_mask.scatter_(1, batch.src, 1)

            p = model.generator(output, generator_mask)
            most_prob_letter = torch.argmax(p, dim=1)
            most_prob_mask = torch.zeros(batch.tgt.shape, device=device)
            most_prob_mask = most_prob_mask.scatter_(1, most_prob_letter.view(-1, 1), 1)
            batch_guess = (batch.tgt * most_prob_mask).sum(dim=1)

            total_correct_guess += (batch_guess != 0).sum().item()
            total_n += batch_guess.shape[0]

        acc = total_correct_guess / total_n

    if was_training:
        model.train()

    return acc

def evaluate_acc_v3(model, vocab, dev_data, device):
    was_training = model.training
    model.eval()

    # no_grad() signals backend to throw away all gradients
    total_correct_guess = 0
    total_n = 0
    with torch.no_grad():
        for batch in create_words_batch_v3(dev_data, vocab, mini_batch=30, shuffle=False, device=device):
            x = batch.src
            x_mask = batch.src_mask
            #width = x.shape[1]
            if(batch.guessed is not None):
                x = torch.cat([batch.src, batch.guessed], dim = 1)
                #x_mask = ((x != mask_token) & (x != pad_token)).unsqueeze(-2)
                x_mask = x.unsqueeze(1).to(model.device)
            out = model.forward(x, x_mask)
            #out = out[:, :width, :]
            generator_mask = torch.zeros(x.shape[0], len(vocab.char2id), device=model.device)
            generator_mask = generator_mask.scatter_(1, x, 1)

            p = model.generator(out, generator_mask)
            most_prob_letter = torch.argmax(p, dim=1)
            most_prob_mask = torch.zeros(batch.tgt.shape, device=device)
            most_prob_mask = most_prob_mask.scatter_(1, most_prob_letter.view(-1, 1), 1)
            batch_guess = (batch.tgt * most_prob_mask).sum(dim=1)

            total_correct_guess += (batch_guess != 0).sum().item()
            total_n += batch_guess.shape[0]

        acc = total_correct_guess / total_n

    if was_training:
        model.train()

    return acc


def convert_target_to_dist(target, vocab, mask, device):
    dist_mask = torch.zeros(target.shape[0], len(vocab.char2id), device=device)
    dist_mask = dist_mask.scatter_(1, target * mask, 1)

    target_numpy = (target * mask).cpu().numpy()
    extra_col = np.ones((target.shape[0], 1), dtype=target_numpy.dtype) * (vocab.char2id['z'] + 1)
    target_numpy = np.hstack((target_numpy, extra_col))
    target_np_dist = np.apply_along_axis(np.bincount, 1, target_numpy)[:, :-1]
    if device.type == 'cuda':
        target_dist = torch.from_numpy(target_np_dist).cuda()
    else:
        target_dist = torch.from_numpy(target_np_dist)
    target_dist[:, 0] = 0

    return target_dist * dist_mask


def create_words_batch(lines, vocab, mini_batch: int, device, shuffle=True):
    if shuffle:
        np.random.shuffle(lines)

    src_buffer = []
    tgt_buffer = []
    ret = []
    for line in lines:
        src_word, tgt_word = line.split(',')
        if len(line) > 1:
            src_buffer.append([vocab.char2id[c] for c in src_word])
            tgt_buffer.append([vocab.char2id[c] for c in tgt_word])

            if len(src_buffer) == mini_batch:
                src = torch.tensor(pad_words(src_buffer, vocab.char2id[PAD]), device=device)
                tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id[PAD]), device=device)
                tgt_dist = convert_target_to_dist(tgt, vocab, src == vocab.char2id[MASK], device=device)
                tgt_dist = torch.div(tgt_dist, tgt_dist.sum(dim=1)[:, None] + 1e-12)

                batch = Batch(src, tgt_dist, mask_token=vocab.char2id[MASK], pad_token=vocab.char2id[PAD])
                #yield batch
                ret.append(batch)
                src_buffer = []
                tgt_buffer = []
    

    if len(src_buffer) != 0:
        src = torch.tensor(pad_words(src_buffer, vocab.char2id[PAD]), device=device)
        tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id[PAD]), device=device)
        tgt_dist = convert_target_to_dist(tgt, vocab, src == vocab.char2id[MASK], device=device)
        tgt_dist = torch.div(tgt_dist, tgt_dist.sum(dim=1)[:, None] + 1e-12)

        batch = Batch(src, tgt_dist, mask_token=vocab.char2id[MASK], pad_token=vocab.char2id[PAD])
        #yield batch
        ret.append(batch)
    #ret = torch.tensor(ret)
    return ret


def create_words_batch_v2(lines, vocab, mini_batch: int, device, shuffle=True):
    line_unique = np.unique(np.array([line.split(",")[1] for line in lines]))
    data = []
    for line in line_unique:
        word = np.array(list(line))
        word_unique = np.unique(word)
        word_length = len(word_unique)
        for k in range(1, word_length):
            if(word_length > k):
                mm = word_length/k
                if(np.random.rand() >0.5):
                    mm = word_length/(word_length - k)
                for a in range(int(np.sqrt(mm))):
                    perm = np.random.permutation(word_unique)
                    selected = perm[:k]
                    table = str.maketrans({c: "_" for c in selected})
                    target = line.translate(table)
                    data.append(",".join([target, line]))
    data = np.array(data)
        
    if shuffle:
        np.random.shuffle(data)
    
    src_buffer = []
    tgt_buffer = []
    ret = []
    for line in data:
        src_word, tgt_word = line.split(",")
        src_buffer.append([vocab.char2id[c] for c in src_word])
        tgt_buffer.append([vocab.char2id[c] for c in tgt_word])

        if len(src_buffer) == mini_batch:
            src = torch.tensor(pad_words(src_buffer, vocab.char2id[PAD]), device=device)
            tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id[PAD]), device=device)
            if(torch.any(torch.isnan(src))):
                print(src_buffer)
            tgt_dist = convert_target_to_dist(tgt, vocab, src == vocab.char2id[MASK], device=device)
            tgt_dist = torch.div(tgt_dist, tgt_dist.sum(dim=1)[:, None] + 1e-12)

            batch = Batch(src, tgt_dist, mask_token=vocab.char2id[MASK], pad_token=vocab.char2id[PAD])
            #yield batch
            ret.append(batch)
            src_buffer = []
            tgt_buffer = []
    

    if len(src_buffer) != 0:
        src = torch.tensor(pad_words(src_buffer, vocab.char2id[PAD]), device=device)
        tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id[PAD]), device=device)
        tgt_dist = convert_target_to_dist(tgt, vocab, src == vocab.char2id[MASK], device=device)
        tgt_dist = torch.div(tgt_dist, tgt_dist.sum(dim=1)[:, None] + 1e-12)

        batch = Batch(src, tgt_dist, mask_token=vocab.char2id[MASK], pad_token=vocab.char2id[PAD])
        #yield batch
        ret.append(batch)
    #ret = torch.tensor(ret)
    return ret

def create_words_batch_v3(lines, vocab, mini_batch: int, device, shuffle=True):
    line_unique = np.unique(np.array([line.split(",")[1] for line in lines]))
    alphabet = set(string.ascii_lowercase)
    data = []
    for line in line_unique:
        word = np.array(list(line))
        word_unique = np.unique(word)
        word_length = len(word_unique)
        letter_set = set(line)
        unused = list(alphabet - letter_set)
        for k in range(1, word_length):
            if(word_length > k):
                mm = word_length/k
                if(np.random.rand() >0.5):
                    mm = word_length/(word_length - k)
                for a in range(int(np.sqrt(mm))):
                    perm = np.random.permutation(word_unique)
                    selected = perm[:k]
                    table = str.maketrans({c: "_" for c in selected})
                    target = line.translate(table)
                    perm = np.random.permutation(unused)
                    guessed_letters = PAD + PAD + PAD.join(perm[:min(np.random.randint(0, 6), len(perm))]) + PAD
                    data.append(",".join([target, line, guessed_letters]))
    data = np.array(data)
        
    if shuffle:
        np.random.shuffle(data)
    
    src_buffer = []
    tgt_buffer = []
    gue_buffer = []
    ret = []
    for line in data:
        src_word, tgt_word, guessed_letter = line.split(",")
        src_buffer.append([vocab.char2id[c] for c in src_word])
        tgt_buffer.append([vocab.char2id[c] for c in tgt_word])
        gue_buffer.append([vocab.char2id[c] for c in guessed_letter])
        if len(src_buffer) == mini_batch:
            src = torch.tensor(pad_words(src_buffer, vocab.char2id[PAD]), device=device)
            tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id[PAD]), device=device)
            gue = torch.tensor(pad_words(gue_buffer, vocab.char2id[PAD]), device=device)
            #gue = None
            tgt_dist = convert_target_to_dist(tgt, vocab, src == vocab.char2id[MASK], device=device)
            tgt_dist = torch.div(tgt_dist, tgt_dist.sum(dim=1)[:, None] + 1e-12)

            batch = Batch(src, tgt_dist, mask_token=vocab.char2id[MASK], pad_token=vocab.char2id[PAD], guessed = gue)
            #yield batch
            ret.append(batch)
            src_buffer = []
            tgt_buffer = []
            gue_buffer = []
    

    if len(src_buffer) != 0:
        src = torch.tensor(pad_words(src_buffer, vocab.char2id[PAD]), device=device)
        tgt = torch.tensor(pad_words(tgt_buffer, vocab.char2id[PAD]), device=device)
        gue = torch.tensor(pad_words(gue_buffer, vocab.char2id[PAD]), device=device)
        #gue = None
        tgt_dist = convert_target_to_dist(tgt, vocab, src == vocab.char2id[MASK], device=device)
        tgt_dist = torch.div(tgt_dist, tgt_dist.sum(dim=1)[:, None] + 1e-12)

        batch = Batch(src, tgt_dist, mask_token=vocab.char2id[MASK], pad_token=vocab.char2id[PAD], guessed = gue)
        #yield batch
        ret.append(batch)
    #ret = torch.tensor(ret)
    return ret


def create_data(input_path = "./words_250000_train.txt", output_train_path = "./pairs_train.txt", output_valid_path = "./pairs_valid.txt", shuffle=True):
    lines = []
    with open(input_path, 'r') as f:
        lines = f.read().split("\n")
    if(shuffle):
        np.random.shuffle(lines)
    train_lines = lines
    valid_lines = []
    length = [len(train_lines), 0]
    if(output_valid_path is not None):
        train_lines = lines[:int(len(lines)*0.9)]
        valid_lines = lines[int(len(lines)*0.9):]
        length = [len(train_lines), len(valid_lines)]
    lines = [train_lines, valid_lines]

    mask_rate = [np.random.uniform(0.3, 1.0, size=(length[0], 3)), np.random.uniform(0.3, 1.0, size=(length[1], 3))]
    data = [[], []]
    for j in range(2):
        for i in range(length[j]):
            word = np.array(list(lines[j][i]))
            word_unique = np.unique(word)
            word_length = len(word_unique)
            for k in range(1, word_length):
                if(word_length > k):
                    for a in range(int(np.sqrt(word_length/k))):
                        perm = np.random.permutation(word_unique)
                        selected = perm[:k]
                        table = str.maketrans({c: "_" for c in selected})
                        target = lines[j][i].translate(table)
                        data[j].append(",".join([target, lines[j][i]]))
        data[j] = np.unique(data[j])

    train_data = data[0]
    valid_data = data[1]
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)

    with open(output_train_path, 'w') as f:
        f.write("\n".join(train_data))
    with open(output_valid_path, 'w') as f:
        f.write("\n".join(valid_data))
        
def create_data_small(input_path = "./words_250000_train.txt", output_train_path = "./pairs_train_small.txt", output_valid_path = "./pairs_valid_small.txt", shuffle=True):
    lines = []
    with open(input_path, 'r') as f:
        lines = f.read().split("\n")
    if(shuffle):
        np.random.shuffle(lines)
    train_lines = lines
    valid_lines = []
    length = [len(train_lines), 0]
    if(output_valid_path is not None):
        train_lines = lines[:int(len(lines)*0.9)]
        valid_lines = lines[int(len(lines)*0.9):]
        length = [len(train_lines), len(valid_lines)]
    lines = [train_lines, valid_lines]

    mask_rate = [np.random.uniform(0.3, 1.0, size=(length[0], 3)), np.random.uniform(0.3, 1.0, size=(length[1], 3))]
    data = [[], []]
    for j in range(2):
        for i in range(length[j]):
            if(len(lines[j][i])>=6):
                continue
            word = np.array(list(lines[j][i]))
            word_unique = np.unique(word)
            word_length = len(word_unique)
            for k in range(1, word_length):
                if(word_length > k):
                    for a in range(int(np.sqrt(word_length/k))):
                        perm = np.random.permutation(word_unique)
                        selected = perm[:k]
                        table = str.maketrans({c: "_" for c in selected})
                        target = lines[j][i].translate(table)
                        data[j].append(",".join([target, lines[j][i]]))
        data[j] = np.unique(data[j])

    train_data = data[0]
    valid_data = data[1]
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)

    with open(output_train_path, 'w') as f:
        f.write("\n".join(train_data))
    with open(output_valid_path, 'w') as f:
        f.write("\n".join(valid_data))


def read_train_data(filepath = "./pairs_train.txt", small = False):
    with open(filepath) as f:
        words = f.read().split('\n')
    if small:
        return words[:10]
    else:
        return words


def read_dev_data(filepath = "./pairs_valid.txt", small = False):
    with open(filepath) as f:
        words = f.read().split('\n')
    if small:
        return words[:10]
    else:
        return words