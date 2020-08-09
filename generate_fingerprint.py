import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from gensim.models import word2vec

from sklearn.cluster import AgglomerativeClustering


############################################
#                   drug
############################################

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, 'p':15}

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind)-1))
	for i, ch in enumerate(line[:MAX_SMI_LEN]):
		X[i, (smi_ch_ind[ch]-1)] = 1
	return X

def SimlesToFp(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_vec = np.array(fp)
    return fp_vec



# drug_fp = SimlesToFp(smi).reshape((1,1024))
# drug = one_hot_smiles(smi, 100, CHARISOSMISET).reshape((1,100,64))

############################################
#                 protein
############################################
with open('KIBA_protein.pickle', 'rb') as f:
    store = pickle.load(f)
    protein_seq = store['seq']


words = []
for i, mat in enumerate(protein_seq):
    word = []
    for j, seq in enumerate(mat):
        is_end = True
        for n, label in enumerate(seq):
            if label == 1:
                word.append(int(n))
                is_end = False
                break
        if is_end:
            break
    words.append(word)

protein_words = words

all_vector = []
proteins = []
slice_dic = {}
out_cls = 1024
ac = None

def convert_to_onehot(window, out_cls):
    """
        window：window length
        out_cls：output class number
    """
    global ac

    word_num = 0
    for code in protein_words:
        protein = []
        for index in range((len(code) - window + 1)):
            c = 0
            for j in range(window):
                c = c * 21 + code[index + j]
            protein.append(str(c))
            word_num += 1
        proteins.append(protein)
    print("totally {} slice".format(word_num))

    # 生成片段对应的vec （利用 word2vec 算法）
    slice_window = 9 - window
    model = word2vec.Word2Vec(proteins, sg=0, size=32, window=slice_window, min_count=3, negative=1, sample=0.001, hs=1, workers=4,
                              batch_words=10, iter=1000, alpha=0.0001)

    all_words = model.wv.index2word
    print("totally {} class".format(len(all_words)))
    for word in all_words:
        all_vector.append(model[word])

    # divide into 1024 class
    ac = AgglomerativeClustering(n_clusters=1024)
    cls = ac.fit_predict(all_vector)
    print(cls)

    # generate class for all slice
    for i, word in enumerate(all_words):
        slice_dic[word] = cls[i]

    # generate onehot
    for i, protein in enumerate(proteins):
        for slice in protein:
            if slice in slice_dic:
                protein_onehot[i][slice_dic[slice]] = 1
    return protein_onehot

protein_onehot = np.zeros((442, out_cls))
codes = convert_to_onehot(5, out_cls)