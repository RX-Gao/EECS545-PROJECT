import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import scipy.io as scio


def sliding_windows(data, seq_length):
    X = []
    Y = []

    for i in range(len(data)-seq_length-pre_steps+1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length:i+seq_length+pre_steps]
        X.append(x)
        Y.append(y)

    return np.array(X),np.array(Y)

def generate_data(training_set, seq_length):
    ### ge
    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)

    x, y = sliding_windows(training_data, seq_length)

    train_size = int(len(y) * 0.8)
    test_size = len(y) - train_size

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))
    dataX = dataX.permute(1,0,2)

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
    trainX = trainX.permute(1,0,2)

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    testX = testX.permute(1,0,2)
    return dataX, dataY, trainX, trainY, testX, testY, sc, train_size



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                          dropout=dropout_prob, batch_first=False)

    def forward(self, x, state):
        h_size = list(x.size())
        hs = torch.zeros((h_size[0],h_size[1],self.hidden_size))
        for i in range(h_size[0]):
            if i == 0:
                (y_out, h_out) = self.gru(x, torch.zeros((self.num_layers,h_size[1],self.hidden_size)))
            else:
                (y_out, h_out) = self.gru(x, h_out)
                h_last = h_out
            hs[i,:,:] = h_out[self.num_layers-1,:,:]
        return (y_out, hs, h_last)
      
    def begin_state(self):
        return None

def attention_model(input_size, attention_size):
    a = nn.Sequential(nn.Linear(input_size, attention_size, bias=False),
                      nn.Tanh(),
                      nn.Linear(attention_size, 1, bias=False)) 
    return a

def attention_forward(a, encoder_state, decoder_state):
    decoder_state = decoder_state.expand(encoder_state.size())
    # print(list(encoder_state.size()))
    # print(list(decoder_state.size()))
    all_states = torch.cat((encoder_state, decoder_state), dim=2) 
    e = a(all_states)
    alpha = F.softmax(e, dim=0)
    out = (alpha*encoder_state).sum(dim=0)
    return out

class Decoder(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, attention_size, dropout_prob):
        super(Decoder, self).__init__()
        self.attention = attention_model(2*hidden_size, attention_size)
        self.gru = nn.GRU(output_size+hidden_size, hidden_size, num_layers, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, current_input, state, encoder_state):
        c = attention_forward(self.attention, encoder_state, state[-1,:,:])
#         print(list(c.size()))
#         print(list(current_input.size()))
        input_c = torch.cat((current_input, c), dim=1)
#         print(list(state.size()))
#         print(list(encoder_state.size()))
        y, h = self.gru(input_c.unsqueeze(0), state)
        y = self.fc(y).squeeze(dim=0)
        return y, h

    def begin_state(self, encoder_state):
        return encoder_state
    
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + 1e-10)
        return loss

num_epochs = 1001
learning_rate = 1e-2
input_size = 1
output_size = 1
pre_steps = 10

# Parameters to be tuned
# =============================================================================
seq_length = 30

hidden_size = 7
attention_size = 7
num_layers = 1
dropout_prob = 0.0
# =============================================================================


RMSE = RMSELoss()

def train(dataset):
    # Train the model
    dataX, dataY, trainX, trainY, testX, testY, sc, train_size = \
    generate_data(dataset, seq_length)
    trainX = trainX
    testX = testX
    dataX = dataX

    encoder = Encoder(input_size, hidden_size, num_layers, dropout_prob)
    decoder = Decoder(output_size, input_size, hidden_size, num_layers, attention_size, dropout_prob)
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    
    encoder.train()
    decoder.train()
    for epoch in range(num_epochs):
        encoder_outputs, encoder_state, encoder_state_last = encoder(trainX, encoder.begin_state())
        decoder_state = decoder.begin_state(encoder_state_last)
        
        current_input = trainX[-1,:,:]
        output = torch.ones(trainY.size())
        
        for i in range(pre_steps):
            output[:,i,:], decoder_state = decoder(current_input, decoder_state, encoder_state)
            current_input = output[:,i,:]
            
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        # obtain the loss function
        loss = RMSE(output.squeeze(), trainY.squeeze())
        loss.backward()
        
        encoder_optimizer.step()
        decoder_optimizer.step()

        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.7f" % (epoch, loss.item()))
    
    return encoder, decoder, train_size


def test(encoder, decoder, dataset):
    encoder.eval()
    decoder.eval()

    dataX, dataY, trainX, trainY, testX, testY, sc, train_size = \
    generate_data(dataset, seq_length)
    total_len = dataX.shape[1]
    train_size = int(total_len * 0.8)

    # Test the model
    encoder_outputs, encoder_state, encoder_state_last = encoder(dataX, encoder.begin_state())
    decoder_state = decoder.begin_state(encoder_state_last)

    current_input = dataX[-1,:,:]
    output_precdict = torch.ones(dataY.size())
        
    for i in range(pre_steps):
        output_precdict[:,i,:], decoder_state = decoder(current_input, decoder_state, encoder_state)
        current_input = output_precdict[:,i,:]


    loss_predict = RMSE(output_precdict[train_size:total_len], testY)
    print('test loss = %1.7f' % (loss_predict).item())
    
    output_precdict = output_precdict.data.numpy().reshape(-1,pre_steps)
    dataY = dataY.data.numpy().reshape(-1,pre_steps)

    output_predict_sc = sc.inverse_transform(output_precdict)
    dataY = sc.inverse_transform(dataY)

    return output_predict_sc



## positive
# =============================================================================
print('train positive\n')
data = pd.read_csv('data.csv')
positive = data.iloc[:,2:3].values[::-1]
enc, dec, train_size = train(positive)
predicted = test(enc, dec, positive)
true_x, true_y = sliding_windows(positive, seq_length)
true_x = true_x.squeeze()
true_y = true_y.squeeze()

# visualize
length = predicted.shape[0]

plt.figure(figsize=(15,15))
plt.suptitle('Infected / non-SIR AGRU Model', fontsize=20)
for i in range(pre_steps):
    plt.subplot(int(pre_steps/2),2,i+1)
    plt.axvline(x=train_size+seq_length-1+i, c='r', linestyle='--')
    plt.plot(np.arange(seq_length+i,seq_length+i+length), \
             true_y[:,i], label='true')
    plt.plot(np.arange(seq_length+i,seq_length+i+length), \
             predicted[:,i], label='predict')
    plt.legend()
    plt.title(i+1)
# =============================================================================


## SIR
# =============================================================================
# train beta and gamma
print('train beta\n')
data = scio.loadmat('beta.mat')
betas = np.array(data['beta']).transpose()
enc_beta, dec_beta, train_size_1 = train(betas)
pred_beta = test(enc_beta, dec_beta, betas)


print('train gamma\n')
data = scio.loadmat('gamma.mat')
gammas = np.array(data['gamma']).transpose()
enc_gamma, dec_gamma, train_size_1 = train(gammas)
pred_gamma = test(enc_gamma, dec_gamma, gammas)


N = 3.28*10**8

R_true = scio.loadmat('ser_R.mat')    
R_true = np.array(R_true['ser_R']).transpose()[0]

I_true = scio.loadmat('ser_I.mat')    
I_true = np.array(I_true['ser_I']).transpose()[0]

S_true = N - R_true - I_true

# ground truth
R_true_X, R_true_Y = sliding_windows(R_true, seq_length)
I_true_X, I_true_Y = sliding_windows(I_true, seq_length)
S_true_X, S_true_Y = sliding_windows(S_true, seq_length)

# prediction
R_pred = np.zeros(R_true_Y.shape)
I_pred = np.zeros(I_true_Y.shape)
S_pred = np.zeros(S_true_Y.shape)

for step in range(pre_steps):
    if step == 0:
        last_R = R_true_X[:,-1]
        last_I = I_true_X[:,-1]
        last_S = S_true_X[:,-1]
    R_pred[:,step] = last_R + pred_gamma[:,step]*last_I
    I_pred[:,step] = last_I - pred_gamma[:,step]*last_I + pred_beta[:,step]*last_I*last_S/N
    S_pred[:,step] = last_S - pred_beta[:,step]*last_I*last_S/N
    
    last_R = R_pred[:,step]
    last_I = I_pred[:,step]
    last_S = S_pred[:,step]

# visualize
length = R_pred.shape[0]

plt.figure(figsize=(15,15))
plt.suptitle('Infected / SIR AGRU Model', fontsize=20)
for i in range(pre_steps):
    plt.subplot(int(pre_steps/2),2,i+1)
    plt.axvline(x=train_size_1+seq_length-1+i, c='r', linestyle='--')
    plt.plot(np.arange(seq_length+i,seq_length+i+length), \
             I_true_Y[:,i], label='true')
    plt.plot(np.arange(seq_length+i,seq_length+i+length), \
             I_pred[:,i], label='predict')
    plt.legend()
    plt.title(i+1)
plt.show()
# =============================================================================


# Evaluation / Criterioin: RMSE + R2 Score
# =============================================================================
RMSE1 = mean_squared_error(true_y[train_size:,:], predicted[train_size:,:], squared=False)
RMSE2 = mean_squared_error(I_true_Y[train_size_1:,:], I_pred[train_size_1:,:], squared=False)
R2_1 = r2_score(true_y[train_size:,:], predicted[train_size:,:])
R2_2 = r2_score(I_true_Y[train_size_1:,:], I_pred[train_size_1:,:])

RMSE1_list, RMSE2_list = list(), list()
for i in range(pre_steps):
    RMSE1_list.append(mean_squared_error(true_y[train_size:,i], predicted[train_size:,i], squared=False))
    RMSE2_list.append(mean_squared_error(I_true_Y[train_size_1:,i], I_pred[train_size_1:,i], squared=False))

print('RMSE of non-SIR AGRU Model = %5f' % RMSE1)
print('R2 Score of non-SIR AGRU Model = %5f' % R2_1)
print('RMSE of single step:')
print(RMSE1_list)
print('\n')
print('RMSE of SIR AGRU Model = %5f' % RMSE2)
print('R2 Score of SIR AGRU Model = %5f' % R2_2)
print('RMSE of single step:')
print(RMSE2_list)
# =============================================================================
    

#
#np.save(HMM_true, true_y)
#np.save(HMM_predict, predicted)
#np.save(HMM_true_SIR, I_true_Y)
#np.save(HMM_predict_SIR, I_pred)