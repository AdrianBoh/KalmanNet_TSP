from Simulations.Extended_sysmdl import SystemModel
from KNet.KalmanNet_nn import KalmanNetNN #change step prior
from Pipelines.Pipeline_EKF import Pipeline_EKF

from datetime import datetime
import Simulations.config as config
import torch
import torch.nn as nn
import numpy as np
from helper_nd import *
from Simulations.utils import Short_Traj_Split



#use mxm identity for Q and R (nxn), h is also identity function, f is the minimum jerk prior
#i can remove all if chop blocks i come across

print("Pipeline Start")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ### WILL HAVE TO ADAPT THESE
###################
args = config.general_settings()
### dataset parameters
args.N_E = 1000 #number of chunks
args.N_CV = 100
args.N_T = 200
args.T = 100
args.T_test = 100
### training parameters
args.use_cuda = True # use GPU or not
args.n_steps = 20 #2000
args.n_batch = 30
args.lr = 1e-3
args.wd = 1e-3


if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")


path_results = 'KNet/'
DatafolderName = 'Simulations/Neural_Decoding/data' + '/'
chop = True


m = 2 #state dimension
n = 2 #observation dimension   


Q = torch.eye(m)
R = torch.eye(n)

def h_identity(x):
    return x


print("Loading your neural decoder data...")

x_files = {
    'train': 'Datasets/linear_transformer_x_train_predictions.npy',
    'val':   'Datasets/linear_transformer_x_val_predictions.npy',
    'test':  'Datasets/linear_transformer_x_test_predictions.npy'
}

y_files = {
    'train': 'Datasets/linear_transformer_y_train_predictions.npy',
    'val':   'Datasets/linear_transformer_y_val_predictions.npy',
    'test':  'Datasets/linear_transformer_y_test_predictions.npy'
}

data = load_velocity_data(x_files, y_files)

train_input_long, train_target_long = data['train']
cv_input_long, cv_target_long = data['val']
test_input, test_target = data['test']


if chop:
    print("Chopping sequences for training...")
    train_target, train_input, train_init = Short_Traj_Split(train_target_long, train_input_long, args.T)
    cv_target, cv_input, _ = Short_Traj_Split(cv_target_long, cv_input_long, args.T)
else:
    train_target = train_target_long[:, :, :args.T]
    train_input = train_input_long[:, :, :args.T]
    cv_target = cv_target_long[:, :, :args.T]
    cv_input = cv_input_long[:, :, :args.T]

train_input = train_input.to(device)
train_target = train_target.to(device)
cv_input = cv_input.to(device)
cv_target = cv_target.to(device)
train_init = train_init.to(device)
test_input = test_input.to(device)
test_target = test_target.to(device)

history = 30


sys_model = SystemModel(f_wrapper_torch, Q, h_identity, R, T=train_input.shape[2], T_test=test_input.shape[2], m=m, n=n, history=history)
# Initial state mean and covariance
sys_model.InitSequence(torch.zeros(m, 1), torch.eye(m))


#####################
### Evaluate KNet ###
#####################

## KNet with full info ####################################################################################
################
## KNet full ###
################  
## Build Neural Network
print("KNet with full model info")
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model, args)
# ## Train Neural Network
KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
KNet_Pipeline.setssModel(sys_model)
KNet_Pipeline.setModel(KNet_model)
print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
KNet_Pipeline.setTrainingParams(args) 

# --- train ---
[MSE_cv_linear_epoch,
 MSE_cv_dB_epoch,
 MSE_train_linear_epoch,
 MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit=True, train_init=train_init)

# --- test ---
[MSE_test_linear_arr,
 MSE_test_linear_avg,
 MSE_test_dB_avg,
 Knet_out,
 RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)

predicted_states = Knet_out.detach().cpu().numpy()      # shape: [N_T, m, T_test]
ground_truth = test_target      
observations = test_input       

print("\n=== Results ===")
print("Test MSE (linear):", MSE_test_linear_avg.item())
print("Test MSE (dB):", MSE_test_dB_avg.item())
print("Runtime:", RunTime)


strTime_safe = strTime.replace(":", "-")  # 
filename = "Results/predictions" + strTime_safe + ".npz"
np.savez(filename,
         predicted_states=predicted_states,
         ground_truth=ground_truth,
         observations=observations)



