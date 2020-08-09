import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, length_in, length_out):
        super().__init__()
        length_out = length_out // 4
        self.x1 = nn.Conv1d(length_in, length_out, kernel_size=1)
        self.x2 = nn.Conv1d(length_out + length_in, length_out, kernel_size=3, padding=1)
        self.x3 = nn.Conv1d(length_out * 2 + length_in, length_out, kernel_size=5, padding=2)
        self.x4 = nn.Conv1d(length_out * 3 + length_in, length_out, kernel_size=7, padding=3)

    def forward(self, data_in):
        x1 = self.x1(data_in)
        x2 = self.x2(torch.cat((x1, data_in), dim=1))
        x3 = self.x3(torch.cat((x2, x1, data_in), dim=1))
        x4 = self.x4(torch.cat((x3, x2, x1, data_in), dim=1))
        data_out = torch.cat((x1, x2, x3, x4), dim=1)
        #         data_out = torch.nn.functional.dropout(data_out, p=0.5)
        data_out = nn.functional.relu(data_out, inplace=False)

        return data_out


class CNN(nn.Module):
    def __init__(self, type_num=64):
        super().__init__()
        #        self.x1 = nn.Conv1d(type_num, 128, 1)
        self.x1 = ConvBlock(type_num, 128)
        self.x2 = ConvBlock(128, 256)
        self.x3 = ConvBlock(256, 96)

    def forward(self, data_in):
        data_out = self.x1(data_in)
        data_out = self.x2(data_out)
        data_out = self.x3(data_out)
        #        data_out = self.x4(data_out)
        return data_out


class FC(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=True):
        super().__init__()
        self.x1 = nn.Linear(dim_in, dim_out)
        self.x2 = torch.nn.Dropout()
        self.dropout = dropout

    def forward(self, x):
        x = self.x1(x)
        if self.dropout:
            x = self.x2(x)
        x = nn.functional.leaky_relu(x, inplace=False)
        return x


class fp_FC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.x1 = nn.Linear(dim_in, 512)
        self.x2 = torch.nn.Dropout()
        self.x3 = nn.Linear(512, dim_out)

    def forward(self, x):
        x = self.x1(x)
        x = self.x2(x)
        #        x = nn.functional.leaky_relu(x)
        x = self.x3(x)
        #        x = nn.functional.leaky_relu(x)
        return x


class DeepDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_model = CNN(64)
        # self.drug_model[0].load_state_dict(torch.load('temp4generator.state'))
        self.protein_model = CNN(21)
        self.fp_drug = fp_FC(1024, 96)
        self.fp_protein = fp_FC(1024, 96)
        self.atten_protein = nn.Conv1d(402, 96, 1)
        self.atten = nn.Conv1d(96, 1, 1)
        self.fc1 = FC(384, 1024)
        self.fc2 = FC(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, drug, drug_phy, protein, protein_fp):
        drug = self.drug_model(drug)
        protein = self.protein_model(protein)
        drug_out_fp = self.fp_drug(drug_phy)
        protein_out_fp = self.fp_protein(protein_fp)

        # attention
        drug_out = nn.functional.relu(self.atten(drug))
        protein_out = nn.functional.relu(self.atten(protein))
        atten = nn.functional.tanh(drug_out.transpose(dim0=1, dim1=2).bmm(protein_out))
        atten_for_drug = torch.sum(atten, dim=2)
        atten_for_protein = torch.sum(atten, dim=1)
        drug_out = drug * atten_for_drug.unsqueeze(1)
        protein_out = protein * atten_for_protein.unsqueeze(1)

        drug_out_fp = drug_out_fp * torch.sum(drug, 2)
        protein_out_fp = protein_out_fp * torch.sum(protein, 2)

        drug_out = nn.functional.adaptive_max_pool1d(drug_out, output_size=1).squeeze(2)
        protein_out = nn.functional.adaptive_max_pool1d(protein_out, output_size=1).squeeze(2)
        data_out = torch.cat((drug_out, drug_out_fp, protein_out, protein_out_fp), dim=1)

        # fc
        data_out = self.fc1(data_out)
        data_out = self.fc2(data_out)
        data_out = self.fc3(data_out)
        data_out = self.fc4(data_out)
        return data_out