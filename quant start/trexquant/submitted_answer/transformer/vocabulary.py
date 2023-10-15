import string
from utils import PAD, MASK
from utils import pad_token, mask_token

class Vocab:
    def __init__(self):
        self.char2id = dict()
        self.char2id[PAD] = pad_token
        self.char2id[MASK] = mask_token
        self.char_list = string.ascii_lowercase
        for i, c in enumerate(self.char_list):
            self.char2id[c] = i + 2
        self.id2char = {v: k for k, v in self.char2id.items()}