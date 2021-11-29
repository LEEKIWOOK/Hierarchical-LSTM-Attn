import torch
import torch.nn as nn
import torch.nn.functional as F

class MSC_GRU(nn.Module):
    def __init__(self, len: int):
        super(MSC_GRU, self).__init__()
        dim = 5
        seq_len = len+2
        dropout_rate = 0.3
        self.RNN_hidden = 20
        self.dropout = nn.Dropout(dropout_rate)

        self.ConvLayer1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.MaxPool1d(kernel_size=3, stride=1),
        )

        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.BatchNorm1d(dim * 2),
            nn.MaxPool1d(kernel_size=3, stride=1),
        )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.BatchNorm1d(dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding = "same"),
            nn.BatchNorm1d(dim * 2),
            nn.MaxPool1d(kernel_size=3, stride=1),
        )
        self.ConvLayer4 = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.BatchNorm1d(dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding = "same"),
            nn.BatchNorm1d(dim  * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(dim * 2, dim * 3, kernel_size=3, padding = "same"),
            nn.BatchNorm1d(dim * 3),
            nn.MaxPool1d(kernel_size=3, stride=1),
        )
        self.gru = nn.GRU(
            33, self.RNN_hidden, num_layers=2, bidirectional=True
        )
        self.flattening = nn.Flatten()

        self.predictor = nn.Sequential(
            nn.Linear(in_features=1600, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=256, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, inputs):
        x = F.pad(inputs, (1, 1), "constant", 0)
        x = F.one_hot(x).to(torch.float) #256, 35, 5
        x = x.transpose(1, 2)
        
        x2 = x.clone()
        x3 = x.clone()
        x4 = x.clone()
        
        x = self.ConvLayer1(x)
        x2 = self.ConvLayer2(x2)
        x3 = self.ConvLayer3(x3)
        x4 = self.ConvLayer4(x4)
        
        xout = self.dropout(torch.cat([x, x2, x3, x4], dim=1))
        #xout = self.flattening(xout)
        xout, _ = self.gru(xout)
        F_RNN = xout[:, :, : self.RNN_hidden]
        R_RNN = xout[:, :, self.RNN_hidden :]
        xout = torch.cat((F_RNN, R_RNN), 2)
        xout = self.dropout(xout)
        xout = self.flattening(xout)
        
        xout = self.predictor(xout)
        return xout.squeeze()
