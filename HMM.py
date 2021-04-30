import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm
import scipy.io as scio
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
 

def sliding_windows(data, seq_length):
    X = []
    Y = []

    for i in range(len(data)-seq_length-pred_len+1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length:i+seq_length+pred_len]
        X.append(x)
        Y.append(y)

    return np.array(X),np.array(Y)


class Predictor(object):
    def __init__(self,n_hidden_states=4, n_latency_days=20, pred_len=10,
                 n_steps_frac_change=10,category='positive'):
        self._init_logger()
        
        self.category = category
 
        self.n_latency_days = n_latency_days
        
        self.pred_len = 10
 
        self.hmm = GaussianHMM(n_components=n_hidden_states)
 
        self._split_train_test_data()
 
        self._compute_all_possible_outcomes(
            n_steps_frac_change)
 
    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
 
    def _split_train_test_data(self):
        if self.category == 'death' or self.category == 'positive':
            data = pd.read_csv('data.csv')
            if self.category == 'death':
                data = data.iloc[:,1].values[::-1]
            else:
                data = data.iloc[:,2].values[::-1]
        else:
            if self.category == 'beta':
                data = scio.loadmat('beta.mat')
                data = np.array(data['beta']).transpose().squeeze()
            else:
                data = scio.loadmat('gamma.mat')
                data = np.array(data['gamma']).transpose().squeeze()
#        _train_data, test_data = train_test_split(
#            data, test_size=test_size, shuffle=False)
        self.T = data.shape[0]
        self.test_size = int(self.T*0.2)
        self.data = data
        self._train_data = data[0:self.T-self.test_size]
        self._test_data = data[self.T-self.test_size:]
 
    @staticmethod
    def _extract_features(data):                
        sub = np.zeros(data.shape)
        sub[1:] = data[0:-1]
        frac_change = (data[1:] - sub[1:]) / sub[1:]
        return frac_change.reshape(-1,1)
 
    def fit(self):
#        self._logger.info('>>> Extracting Features')
        feature_vector = Predictor._extract_features(self._train_data)
#        self._logger.info('Features extraction Completed <<<')
 
        self.hmm.fit(feature_vector)
 
    def _compute_all_possible_outcomes(self, n_steps_frac_change):
        if self.category == 'positive':
            frac_change_range = np.linspace(0, 0.1, n_steps_frac_change)
        else:
            frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
        self._possible_outcomes = frac_change_range
 
    def _get_most_probable_outcome(self, day_index):
        previous_data_start_index = day_index - self.n_latency_days + len(self.pred_batch)
        previous_data_end_index = day_index
        
        previous_data = np.zeros(self.n_latency_days)
        previous_data[0:self.n_latency_days-len(self.pred_batch)] = self.data[previous_data_start_index:previous_data_end_index]
        previous_data[self.n_latency_days-len(self.pred_batch):self.n_latency_days] = np.array(self.pred_batch)
        previous_data_features = Predictor._extract_features(previous_data)
        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack((previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(
            outcome_score)]
        return most_probable_outcome
 
    def predict_next(self, day_index):
        if len(self.pred_batch) == 0:
            last = self.data[day_index-1]
        else:
            last = self.pred_batch[-1]
        predicted_frac_change = self._get_most_probable_outcome(day_index)
        self.pred_batch.append(last * (1 + predicted_frac_change))
 
    def predict_next_for_days(self, days, with_plot=False):
        predicted = []
        for day_index in tqdm(range(self.n_latency_days,days+self.n_latency_days)):
            self.pred_batch = []
            for pred_index in range(self.pred_len):
                self.predict_next(day_index)
            predicted.append(self.pred_batch)
        
        predicted = np.array(predicted)
        if with_plot:
            if self.category == 'death' or self.category == 'positive':
                plt.figure(figsize=(15,15))
                plt.suptitle('Infected / non-SIR HMM Model', fontsize=20)
                for i in range(self.pred_len):
                    actual = self.data[self.n_latency_days+i:days+self.n_latency_days+i]
                    plt.subplot(int(self.pred_len/2),2,i+1)
                    plt.plot(np.arange(self.n_latency_days+i,days+self.n_latency_days+i), actual, '-', label="actual")
                    plt.plot(np.arange(self.n_latency_days+i,days+self.n_latency_days+i), predicted[:,i], '-', label="predicted")
                    plt.axvline(x=self.T-self.test_size, c='r', linestyle='--')
                    plt.title(i+1)              
                    plt.legend()
                plt.show()
 
        return predicted
 
pred_len = 10
 
# Parameters to be tuned
# =============================================================================
n_latency_days = 20
n_steps_frac_change = 50
n_hidden_states = 5
# =============================================================================

## positive
# =============================================================================
predictor = Predictor(n_hidden_states, n_latency_days, pred_len, \
                                 n_steps_frac_change,'positive')
predictor.fit()
days_num = predictor.T-n_latency_days-pred_len+1
predicted = predictor.predict_next_for_days(days_num, with_plot=True)  
true_x, true_y = sliding_windows(predictor.data, n_latency_days)
true_x = true_x.squeeze()
true_y = true_y.squeeze()
train_size = predictor.T-predictor.test_size-n_latency_days-pred_len+1
# =============================================================================


## SIR
# =============================================================================
predictor = Predictor(n_hidden_states, n_latency_days, pred_len, \
                                 n_steps_frac_change*2,'beta')
predictor.fit()
days_num = predictor.T-n_latency_days-pred_len+1
pred_beta = predictor.predict_next_for_days(days_num)

predictor = Predictor(n_hidden_states, n_latency_days, pred_len, \
                                 n_steps_frac_change*2,'gamma')
predictor.fit()
days_num = predictor.T-n_latency_days-pred_len+1
pred_gamma = predictor.predict_next_for_days(days_num)

seq_length = n_latency_days
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

for step in range(pred_len):
    if step == 0:
        last_R = R_true_X[:,-1]
        last_I = I_true_X[:,-1]
        last_S = S_true_X[:,-1]
    # inner loop -- forecasting steps
    R_pred[:,step] = last_R + pred_gamma[:,step]*last_I
    I_pred[:,step] = last_I - pred_gamma[:,step]*last_I + pred_beta[:,step]*last_I*last_S/N
    S_pred[:,step] = last_S - pred_beta[:,step]*last_I*last_S/N
    
    last_R = R_pred[:,step]
    last_I = I_pred[:,step]
    last_S = S_pred[:,step]

# visualize
length = R_pred.shape[0]
train_size_1 = predictor.T-predictor.test_size-n_latency_days-pred_len+1

plt.figure(figsize=(15,15))
plt.suptitle('Infected / SIR HMM Model', fontsize=20)
for i in range(10):
    plt.subplot(5,2,i+1)
    plt.axvline(x=predictor.T-predictor.test_size, c='r', linestyle='--')
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

RMSE1_list, RMSE2_list = list(), list()
for i in range(pred_len):
    RMSE1_list.append(mean_squared_error(true_y[train_size:,i], predicted[train_size:,i], squared=False))
    RMSE2_list.append(mean_squared_error(I_true_Y[train_size_1:,i], I_pred[train_size_1:,i], squared=False))

print('RMSE of non-SIR HMM Model = %5f' % RMSE1)
print('RMSE of single step:')
print(RMSE1_list)
print('\n')
print('RMSE of SIR HMM Model = %5f' % RMSE2)
print('RMSE of single step:')
print(RMSE2_list)
# =============================================================================
