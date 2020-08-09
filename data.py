import pickle
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

test_fold = json.load(open("test_fold_setting1.txt"))
train_folds = json.load(open("train_fold_setting1.txt"))

with open('KIBA_protein.pickle', 'rb') as f:
    store = pickle.load(f)
    protein_seq = store['seq']

with open(r'KIBA_ligand.pickle', 'rb') as f:
    store = pickle.load(f)
    smiles_actives = store['smiles']
    drug_fp = store['fingerprint']
with open(r'KIBA_relation.pickle', 'rb') as f:
    relationship = pickle.load(f)
    label_row_inds, label_col_inds = np.where(np.isnan(relationship)==False)

prot_fp = np.load('train_fp.npy')

class Datas(Dataset):
    def __init__(self, index, is_train=False):
        indexes = []
        if is_train:
            for i in range(0,index):
                indexes.extend(train_folds[i])
            for i in range(index+1,5):
                indexes.extend(train_folds[i])
        else:
            indexes = test_fold
        self.indexes = indexes
    def __getitem__(self, index):
        i = self.indexes[index]
        drug_i = label_row_inds[i]
        protein_i = label_col_inds[i]
        relation = torch.tensor(relationship[drug_i][protein_i]).float().cuda()
        protein = torch.from_numpy(protein_seq[protein_i]).float().cuda()
        protein_fp = torch.from_numpy(prot_fp[protein_i]).float().cuda()
        drug = torch.from_numpy(smiles_actives[drug_i]).float().cuda()
        drug_phy = torch.from_numpy(drug_fp[drug_i]).float().cuda()
        return drug, drug_phy, protein, protein_fp, relation

    def __len__(self):
        return len(self.indexes)
index = np.ones(relationship.shape[0] * relationship.shape[1])
datas = []

datas = []
for i in range(5):
    datas.append({
        'train': DataLoader(Datas(i, True), batch_size=128, shuffle=True),
        'test': DataLoader(Datas(i, False), batch_size=128, shuffle=True)
    })