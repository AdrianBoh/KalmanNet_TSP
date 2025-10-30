import numpy as np
import torch
from scipy.optimize import minimize, LinearConstraint

def load_velocity_data(x_files, y_files):
    """
    x_files, y_files: dict with keys 'train', 'val', 'test'
                       values are file paths to .npy files
                       each file has columns: [timestamps, ground_truth, prediction]
    
    Returns: dict with keys 'train', 'val', 'test'
             each value is a tuple (inputs, targets) with tensors [N, 2, T]
    """
    data_dict = {}
    
    for split in ['train', 'val', 'test']:
        # Load files
        x_data = np.load(x_files[split])
        y_data = np.load(y_files[split])
        
        # Separate ground truth and predictions
        gt_x, pred_x = x_data[:, 1], x_data[:, 2]
        gt_y, pred_y = y_data[:, 1], y_data[:, 2]
        
        # Stack into [2, T]
        gt = np.stack([gt_x, gt_y], axis=0)
        pred = np.stack([pred_x, pred_y], axis=0)
        
        # Add batch dimension [1, 2, T]
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()
        pred_tensor = torch.from_numpy(pred).unsqueeze(0).float()
        
        data_dict[split] = (pred_tensor, gt_tensor)
        
    return data_dict


# def f_prior(x_input): #this is just to check that the wrapper works
#     return torch.mean(x_input, dim=2)

# def f_wrapper(x_history):
#     """
#     x_history: [batch, m, T] or [batch, m, history]
#     Returns next state: [batch, m, 1]
#     """
#     # Take last 10 timesteps
#     x_input = x_history[:, :, -10:]  # shape: [batch, m, 10]
    
#     # Feed to your jerk-minimizing prior function
#     x_next = f_prior(x_input)        # shape: [batch, m]
#     x_next = x_next.unsqueeze(-1)    # restore shape [batch, m, 1]
    

#     return x_next



#----- this will be the actual a priori function:

def canonical_bell(u):
    """Canonical minimal-jerk bell curve, defined for u in [0,1]."""
    out = np.zeros_like(u, dtype=float)
    mask = (u >= 0) & (u <= 1)
    uu = u[mask]
    out[mask] = 30*uu**2 - 60*uu**3 + 30*uu**4
    return out

def reduced_cost(params, x, y, ridge=1e-12):
    """
    Cost function for optimizer: only s and l are free.
    h is computed analytically.
    """
    s, l = params
    if l <= 1e-12:
        return 1e12  # avoid divide by zero

    u = (x - s)/l
    phi = canonical_bell(u)

    # If phi is all zeros, return large cost
    if np.all(phi == 0):
        return 1e12

    # Analytic h
    h = np.sum(phi * y) / (np.sum(phi**2) + ridge)
    resid = y - h*phi
    return float(np.sum(resid**2))

def fit_bell_analytic_h(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0, x9 = x.min(), x.max()

    # Initial guess
    s0 = x0 + 0.25*(x9 - x0)
    l0 = max(0.5, x9 - s0)
    init = np.array([s0, l0])

    # Bounds: s in [0,x0], l>0
    eps = 1e-6
    bounds = [(0, x0), (x9 - x0, None)]

    # Linear constraints: optional, ensure bell covers the segment
    A = np.array([[1, 0], [-1, -1]])
    b = np.array([x0, -x9])
    lin_con = LinearConstraint(A, -np.inf*np.ones_like(b), b)

    # Optimize s and l
    res = minimize(reduced_cost, init, args=(x, y),
                   method='L-BFGS-B', bounds=bounds)

    s_opt, l_opt = res.x
    u = (x - s_opt)/l_opt
    phi = canonical_bell(u)
    # Compute analytic h
    h_opt = np.sum(phi * y) / np.sum(phi**2)
    sse = np.sum((y - h_opt*phi)**2)

    return s_opt, l_opt, h_opt, sse, res

def predict_minimal_jerk(x, y, l_prev=None, h_prev=None, s_prev=None, sse_prev=None, nbr_predictions=1):
    """
    x are timestamps
    y are corresponding values here
    """

    delta_x = x[1] - x[0]
     
    # s, l, h, sse, _ = fit_bell_optimization(x, y)
    s, l, h, sse, _ = fit_bell_analytic_h(x, y)

    next_xs = []
    for i in range(nbr_predictions):
        next_xs.append(x[-1] + (i+1)*delta_x)

    direct_predictions = h*canonical_bell((next_xs - s)/l)

    if l_prev == None:
        return direct_predictions


    predictions_from_previous = h_prev*canonical_bell((next_xs - s_prev)/l_prev)
    
    confidence = 1/sse
    confidence_prev = 1/sse_prev

    alpha = confidence / (confidence + confidence_prev)

    predictions = alpha*direct_predictions + (1-alpha)*predictions_from_previous
    return predictions, s, l, h, sse
    


def f_wrapper(x_history):
    """
    KalmanNet dynamics wrapper using minimal-jerk prior.
    
    Args:
        x_history: [batch, m, history] tensor of past states

    Returns:
        x_next: [batch, m, 1] tensor of predicted next states
    """

    batch_size, m, history = x_history.shape
    device = x_history.device
    x_next = torch.zeros(batch_size, m, device=device)

    for b in range(batch_size):
        for axis in range(m):
            ys = x_history[b, axis, :].detach().cpu().numpy()  # history for this batch and axis

            # predict next value using your minimal-jerk function
            pred = predict_minimal_jerk(x=np.arange(history), y=ys)
            
            # pred is a tuple (predictions, s, l, h, sse), take the first element 
            if isinstance(pred, tuple):
                pred_value = pred[0]
            else:
                pred_value = pred

            x_next[b, axis] = torch.tensor(pred_value, device=device)

    # Unsqueeze last dim for KalmanNet
    x_next = x_next.unsqueeze(-1)  # [batch, m, 1]

    return x_next
