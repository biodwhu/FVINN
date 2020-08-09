import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from torch import nn
from tqdm import tqdm

from data import datas
from model import DeepDTA

Losssss = 9000000

def evaluate(model, epoch, data, name=''):
    global Losssss
    batch = 0
    loss_value = 0
    rm2_index = 0
    P = []
    Y = []
    model.eval()
    for drug, drug_phy, protein, protein_fp, target_value in data:
        batch += 1
        drug = drug.permute(0, 2, 1)
        protein = protein.permute(0, 2, 1)

        judge = model(drug, drug_phy, protein, protein_fp)
        #         P.append(judge.squeeze(1).detach().cpu().numpy())
        #         Y.append(target_value.detach().cpu().numpy())
        loss_value += loss(judge, target_value.unsqueeze(1)).detach().cpu()
    #         rm2_index += get_rm2(judge.detach().cpu().numpy(), target_value.unsqueeze(1).detach().cpu().numpy())
    #     P = np.concatenate((P), axis=0)
    #     Y = np.concatenate((Y), axis=0)
    #     CI_index = CI(P, Y)
    print('epoch',epoch, ": MSE", loss_value / batch)
    # print("CI_index", CI_index, "\n")
    # print("rm2_index", rm2_index / batch, "\n")
    # with open('train.log', 'a') as f:
    #     f.write(str(epoch) + '  MSE ' + str(loss_value / batch) + '  CI_index  ' + str(CI_index) + '  rm2_index' + str(
    #         rm2_index / batch) + '\n')
    if loss_value < Losssss:
        torch.save(model.state_dict(), name +'_FVINN.state')
    else:
        model.load_state_dict(torch.load(name + '_FVINN.state'))


def train(model, loss, optimizer, data, name='', epoch=300):
    global pre_auc
    progress = tqdm(range(epoch))
    batch = 0
    pre_auc = -1

    for epoch in progress:
        model.train()
        for batch, [drug, drug_phy, protein, protein_fp, target_value] in enumerate(data['train']):
            drug = drug.permute(0, 2, 1)
            protein = protein.permute(0, 2, 1)
            judge = model(drug, drug_phy, protein, protein_fp)
            loss_value = loss(judge, target_value.unsqueeze(1))
            progress.set_description('epoch: {}   batch: {}   loss: {}'.format(epoch, batch+1, loss_value))
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        evaluate(model, epoch, data['test'], name)



loss = nn.MSELoss().cuda()
model = DeepDTA().cuda()
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
train(model, loss, optimizer, datas[0], 'cross' + str(0))
train(model, loss, optimizer, datas[1], 'cross' + str(0))
train(model, loss, optimizer, datas[2], 'cross' + str(0))
train(model, loss, optimizer, datas[3], 'cross' + str(0))
train(model, loss, optimizer, datas[4], 'cross' + str(0))
