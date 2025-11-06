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

def canonical_bell_torch(u: torch.Tensor) -> torch.Tensor:
    """
    Canonical minimal-jerk bell curve in [0,1], torch version.
    u: tensor of any shape
    """
    out = torch.zeros_like(u)
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


def reduced_cost_torch(params: torch.Tensor, x: torch.Tensor, y: torch.Tensor, ridge=1e-12):
    """
    Cost function for optimizer: s and l are free, h computed analytically.
    params: tensor [2] -> s, l
    """
    s, l = params[0], params[1]
    if l <= 1e-12:
        return torch.tensor(1e12, device=params.device)

    u = (x - s) / l
    phi = canonical_bell_torch(u)

    if torch.all(phi == 0):
        return torch.tensor(1e12, device=params.device)

    h = torch.sum(phi * y) / (torch.sum(phi**2) + ridge)
    resid = y - h * phi
    return torch.sum(resid**2)

def fit_bell_analytic_h(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0, x9 = x.min(), x.max()

    # Initial guess
    s0 = x0 
    l0 = x9 - s0
    init = np.array([s0, l0])

    # Bounds: s in [0,x0], l>0
    eps = 1e-6
    bounds = [(0, x0), (x9 - x0, None)]

    # Linear constraints: optional, ensure bell covers the segment
    A = np.array([[1, 0], [-1, -1]])
    b = np.array([x0, -x9])
    #lin_con = LinearConstraint(A, -np.inf*np.ones_like(b), b)

    # Optimize s and l
    # res = minimize(reduced_cost, init, args=(x, y),
    #                method='L-BFGS-B', bounds=bounds)
    
    res = minimize(reduced_cost, init, args=(x, y),
               method='Powell', bounds=bounds, options={'maxiter': 50, 'xtol': 1e-4})

    s_opt, l_opt = res.x
    u = (x - s_opt)/l_opt
    phi = canonical_bell(u)
    # Compute analytic h
    h_opt = np.sum(phi * y) / np.sum(phi**2)
    sse = np.sum((y - h_opt*phi)**2)

    return s_opt, l_opt, h_opt, sse, res

def fit_bell_analytic_h_torch(x: torch.Tensor, y: torch.Tensor, lr=0.01, n_iter=200, ridge=1e-12):
    """
    Torch version of analytic bell fitting with optimization for s and l.
    l is constrained to [0.1*window, 20*window]
    
    Args:
        x: [T] timestamps
        y: [T] values
        lr: learning rate for optimizer
        n_iter: max iterations for L-BFGS
        ridge: small regularization for analytic h
    Returns:
        s, l, h, sse
    """
    device = x.device
    window_len = (x.max() - x.min()).item()
    
    # Initialize s and l as trainable parameters
    s = torch.tensor(x[0] + 0.25*window_len, dtype=torch.float32, device=device, requires_grad=True)
    l = torch.tensor(max(0.5, 0.25*window_len), dtype=torch.float32, device=device, requires_grad=True)

    print("initialized parameters")

    optimizer = torch.optim.LBFGS([s, l], lr=lr, max_iter=n_iter)

    min_l = 0.1 * window_len
    max_l = 20.0 * window_len

    def closure():
        optimizer.zero_grad()
        # clamp l to avoid divide-by-zero and enforce limits
        l_clamped = torch.clamp(l, min=min_l, max=max_l)
        u = (x - s) / l_clamped
        phi = canonical_bell_torch(u)
        h = torch.sum(phi * y) / (torch.sum(phi**2) + ridge)
        sse = torch.sum((y - h*phi)**2)
        sse.backward()
        return sse

    optimizer.step(closure)

    # final optimized s, l
    with torch.no_grad():
        l_final = torch.clamp(l, min=min_l, max=max_l)
        u = (x - s) / l_final
        phi = canonical_bell_torch(u)
        h_final = torch.sum(phi * y) / (torch.sum(phi**2) + ridge)
        sse_final = torch.sum((y - h_final*phi)**2)

    return s.detach(), l_final.detach(), h_final.detach(), sse_final.detach()

def fit_bell_analytic_h_torch_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    ridge: float = 1e-12,
    lr: float = 0.05,
    n_iter: int = 100,
    min_l_factor: float = 0.1,
    max_l_factor: float = 20.0
):
    """
    Fully batched torch version of analytic bell fitting with optimization for s and l.
    
    Args:
        x: [batch, T] timestamps
        y: [batch, T] values
        ridge: small regularization for h computation
        lr: learning rate for optimizer
        n_iter: number of gradient descent iterations
        min_l_factor: minimum allowed l relative to window size
        max_l_factor: maximum allowed l relative to window size
    
    Returns:
        s: [batch] optimal shifts
        l: [batch] optimal lengths
        h: [batch] optimal amplitudes
        sse: [batch] sum of squared errors
    """
    batch_size, T = x.shape
    device = x.device
    dtype = x.dtype

    x0 = x[:, 0]
    xT = x[:, -1]
    window = xT - x0

    # Initial guesses
    initial_s = x0 + 0.25 * window
    initial_l = torch.clamp(window, min=0.5)

    min_l = min_l_factor * window   # window is [batch_size]
    max_l = max_l_factor * window
    
    s_list, l_list, h_list, sse_list = [], [], [], []

    for b in range(batch_size):
        s = initial_s[b].clone().detach().requires_grad_(True)
        l = initial_l[b].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([s, l], lr=lr)
        
        with torch.no_grad():
            for _ in range(n_iter):
                optimizer.zero_grad()
                l_clamped = torch.clamp(l, min=min_l[b], max=max_l[b])
                u = (x[b] - s) / l_clamped
                phi = canonical_bell_torch(u)
                h = torch.sum(phi * y[b]) / (torch.sum(phi**2) + ridge)
                sse = torch.sum((y[b] - h*phi)**2)
                optimizer.step()
        
        with torch.no_grad():
            l_final = torch.clamp(l, min=min_l[b], max=max_l[b])
            u = (x[b] - s) / l_final
            phi = canonical_bell_torch(u)
            h_final = torch.sum(phi * y[b]) / (torch.sum(phi**2) + ridge)
            sse_final = torch.sum((y[b] - h_final*phi)**2)
        
        s_list.append(s.detach())
        l_list.append(l_final.detach())
        h_list.append(h_final.detach())
        sse_list.append(sse_final.detach())

    s_batch = torch.stack(s_list)
    l_batch = torch.stack(l_list)
    h_batch = torch.stack(h_list)
    sse_batch = torch.stack(sse_list)

    return s_batch, l_batch, h_batch, sse_batch

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

def predict_minimal_jerk_torch(x: torch.Tensor, y: torch.Tensor,
                               l_prev=None, h_prev=None, s_prev=None, sse_prev=None,
                               nbr_predictions=1):
    """
    Torch version of minimal-jerk predictor that mirrors original numpy version.
    x: [T] timestamps
    y: [T] values
    Returns:
        predictions: [nbr_predictions]
        s, l, h, sse
    """
    delta_x = x[1] - x[0]
    s, l, h, sse = fit_bell_analytic_h_torch(x, y)  # returns torch scalars

    next_xs = x[-1] + delta_x * torch.arange(1, nbr_predictions+1, device=x.device, dtype=x.dtype)
    direct_predictions = h * canonical_bell_torch((next_xs - s) / l)

    if l_prev is None or h_prev is None or s_prev is None or sse_prev is None:
        return direct_predictions, s, l, h, sse

    predictions_from_previous = h_prev * canonical_bell_torch((next_xs - s_prev) / l_prev)
    confidence = 1 / sse
    confidence_prev = 1 / sse_prev
    alpha = confidence / (confidence + confidence_prev)
    predictions = alpha * direct_predictions + (1 - alpha) * predictions_from_previous

    return predictions, s, l, h, sse


def predict_minimal_jerk_torch_batch(x: torch.Tensor, y: torch.Tensor,
                               l_prev=None, h_prev=None, s_prev=None, sse_prev=None,
                               nbr_predictions=1):
    """
    Torch version of minimal-jerk predictor for a single sequence.
    Replicates the original numpy version with optional blending of previous prediction.

    Args:
        x: [T] timestamps
        y: [T] values
        l_prev, h_prev, s_prev, sse_prev: previous bell parameters
        nbr_predictions: number of steps to predict

    Returns:
        predictions: [nbr_predictions]
        s, l, h, sse
    """
    delta_x = x[1] - x[0]
    
    # Fit current bell parameters
    s, l, h, sse = fit_bell_analytic_h_torch_batch(x, y)

    # Compute next timestamps
    next_xs = x[-1] + delta_x * torch.arange(1, nbr_predictions + 1, device=x.device, dtype=x.dtype)

    # Direct prediction using current bell
    direct_predictions = h * canonical_bell_torch((next_xs - s) / l)

    # If no previous bell, just return direct prediction
    if l_prev is None or h_prev is None or s_prev is None or sse_prev is None:
        return direct_predictions, s, l, h, sse

    # Prediction from previous bell
    prev_predictions = h_prev * canonical_bell_torch((next_xs - s_prev) / l_prev)

    # Compute blending factor based on confidence
    confidence = 1.0 / sse
    confidence_prev = 1.0 / sse_prev
    alpha = confidence / (confidence + confidence_prev)

    # Blend predictions
    predictions = alpha * direct_predictions + (1 - alpha) * prev_predictions

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

    print("wrapper called")

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

def f_wrapper_torch(x_history: torch.Tensor,
                    l_prev=None, h_prev=None, s_prev=None, sse_prev=None,
                    nbr_predictions=1) -> torch.Tensor:
    """
    Vectorized KalmanNet wrapper using minimal-jerk prior (torch version).

    Args:
        x_history: [batch, m, T] past states
        l_prev, h_prev, s_prev, sse_prev: optional previous bell parameters for blending
        nbr_predictions: number of steps to predict

    Returns:
        x_next: [batch, m, nbr_predictions] predicted next states
    """
    batch_size, m, T = x_history.shape
    device = x_history.device
    dtype = x_history.dtype

    # Output tensor
    x_next = torch.zeros(batch_size, m, nbr_predictions, device=device, dtype=dtype)

    # Time indices
    xs = torch.arange(T, device=device, dtype=dtype).expand(batch_size, T)  # [batch, T]
    delta_x = xs[0, 1] - xs[0, 0]  # assume uniform

    # Flatten batch and axes for vectorized computation
    ys_flat = x_history.reshape(batch_size * m, T)
    xs_flat = xs.repeat(m, 1)  # shape [m*batch, T], matches ys_flat

    # Prepare previous parameters if given
    if l_prev is not None:
        l_prev_flat = l_prev.reshape(batch_size * m)
        h_prev_flat = h_prev.reshape(batch_size * m)
        s_prev_flat = s_prev.reshape(batch_size * m)
        sse_prev_flat = sse_prev.reshape(batch_size * m)
    else:
        l_prev_flat = h_prev_flat = s_prev_flat = sse_prev_flat = None

    # Predict each sequence
    preds = []
    s_all, l_all, h_all, sse_all = [], [], [], []
    for i in range(batch_size * m):
        pred, s, l, h, sse = predict_minimal_jerk_torch(
            xs_flat[i],
            ys_flat[i],
            l_prev=l_prev_flat[i] if l_prev_flat is not None else None,
            h_prev=h_prev_flat[i] if h_prev_flat is not None else None,
            s_prev=s_prev_flat[i] if s_prev_flat is not None else None,
            sse_prev=sse_prev_flat[i] if sse_prev_flat is not None else None,
            nbr_predictions=nbr_predictions
        )
        preds.append(pred)
        s_all.append(s)
        l_all.append(l)
        h_all.append(h)
        sse_all.append(sse)

    # Stack and reshape back to [batch, m, nbr_predictions]
    x_next = torch.stack(preds, dim=0).reshape(batch_size, m, nbr_predictions)
    s_all = torch.stack(s_all).reshape(batch_size, m)
    l_all = torch.stack(l_all).reshape(batch_size, m)
    h_all = torch.stack(h_all).reshape(batch_size, m)
    sse_all = torch.stack(sse_all).reshape(batch_size, m)

    return x_next, s_all, l_all, h_all, sse_all

def f_wrapper_torch_full(x_history: torch.Tensor,
                          l_prev=None, h_prev=None, s_prev=None, sse_prev=None,
                          nbr_predictions=1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fully vectorized minimal-jerk KalmanNet wrapper (torch version).

    Args:
        x_history: [batch, m, T] past states
        l_prev, h_prev, s_prev, sse_prev: optional previous bell parameters for blending
        nbr_predictions: number of steps to predict

    Returns:
        x_next: [batch, m, nbr_predictions]
        s_all, l_all, h_all, sse_all: parameters of the bells for each batch/m
    """
    batch_size, m, T = x_history.shape
    device = x_history.device
    dtype = x_history.dtype

    # Flatten batch and state for vectorized computation
    ys_flat = x_history.reshape(batch_size * m, T)
    xs_flat = torch.arange(T, device=device, dtype=dtype).expand(batch_size * m, T)

    print("about to fit bell")

    # Fit bell for all sequences
    s_all, l_all, h_all, sse_all = fit_bell_analytic_h_torch_batch(xs_flat.detach(), ys_flat.detach())

    print(s_all.shape)

    # Prepare previous parameters if given
    if l_prev is not None:
        l_prev_flat = l_prev.reshape(-1)
        h_prev_flat = h_prev.reshape(-1)
        s_prev_flat = s_prev.reshape(-1)
        sse_prev_flat = sse_prev.reshape(-1)
        use_prev = True
    else:
        use_prev = False

    # Compute next time indices
    delta_x = xs_flat[:, 1] - xs_flat[:, 0]  # shape [batch*m]
    next_xs = xs_flat[:, -1].unsqueeze(1) + delta_x.unsqueeze(1) * torch.arange(1, nbr_predictions+1, device=device, dtype=dtype).unsqueeze(0)  # [batch*m, nbr_predictions]

    # Compute minimal-jerk bell
    u = (next_xs - s_all.unsqueeze(1)) / l_all.unsqueeze(1)  # [batch*m, nbr_predictions]
    direct_preds = h_all.unsqueeze(1) * canonical_bell_torch(u)

    if use_prev:
        u_prev = (next_xs - s_prev_flat.unsqueeze(1)) / l_prev_flat.unsqueeze(1)
        prev_preds = h_prev_flat.unsqueeze(1) * canonical_bell_torch(u_prev)
        confidence = 1 / sse_all.unsqueeze(1)
        confidence_prev = 1 / sse_prev_flat.unsqueeze(1)
        alpha = confidence / (confidence + confidence_prev)
        x_next_flat = alpha * direct_preds + (1 - alpha) * prev_preds
    else:
        x_next_flat = direct_preds

    # Reshape back to [batch, m, nbr_predictions]
    x_next = x_next_flat.reshape(batch_size, m, nbr_predictions)


    return x_next