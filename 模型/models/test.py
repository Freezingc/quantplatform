import torch
import LSTM, GRU, ALSTM, GAT, Transformer, TCN, sLSTM, mLSTM, kRNN, Master, SFM
model_dict = {
            'LSTM': LSTM,
            'GRU': GRU,
            'ALSTM': ALSTM,
            'GAT': GAT,
            'Transformer': Transformer,
            'TCN': TCN,
            'mLSTM': mLSTM,
            'sLSTM': sLSTM,
            'kRNN': kRNN,
            'Master': Master,
            'SFM': SFM
        }
device = torch.device('cuda:0')

model = model_dict['SFM'].Model("").float()
x = torch.randn(200, 100, 1)
y = model(x)
print(y.shape)