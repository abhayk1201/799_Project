import os
import pickle
import numpy as np


def pickle_data(data_dict, path="./"):
    if not os.path.isdir(path):
        os.mkdir(path)

    for name, data in data_dict.items():
        with open(path+name+".pkl", "wb") as f:
            pickle.dump(data, f)


def read_pickle(keys, path="./"):
    data_dict = {}
    for key in keys:
        with open(path+key+".pkl", "rb") as f:
            data_dict[key] = pickle.load(f)
    return data_dict
    

def generate_time_grid(T, dt, sig):
    """Generates time grid [0, T] with step dt and noise~N(0, sig^2).

    Args:
        T (float): Terminal time.
        dt (float): Time step.
        sig (float): Std of the Gaussian noise.

    Returns:
        ndarray: 1D array with time points.
    """

    t = np.arange(0, T+dt, dt)
    if t[-1] > T:
        t[-1] = T
    t[1:-1] += np.random.randn(len(t)-2) * sig
    assert all(np.diff(t) > 0)
    return t
    

def undersample_time(t, dt, sig):
    """ Approximates a noisy time grid with step dt and 
    noise~N(0, sig^2) by elements of t.
    
    Args:
        t (ndarray): 1D array with time points.
        dt (float): Time step.
        sig (flaot): Std of the Gaussian noise.
        
    Returns:
        t_new (ndarray): 'Approximation' of the noisy time grid.
        inds (List[int]): Indices of corresponding elements of t_new in t.
    """
    
    assert all(np.diff(t) > 0)

    t_noisy = generate_time_grid(t[-1], dt, sig)
    t_new = np.zeros_like(t_noisy)
    t_new[0] = t_noisy[0]
    t_new[-1] = t_noisy[-1]
    
    ptr = 1
    min_dist = np.inf
    cur_dist = 0.0
    
    inds = np.zeros(len(t_noisy), dtype=np.int64)
    inds[-1] = len(t) - 1

    for i in range(1, len(t_new)-1):
        min_dist = np.inf
        while(1):
            cur_dist = abs(t_noisy[i] - t[ptr])
            if cur_dist <= min_dist:
                min_dist = cur_dist
                ptr += 1
            else:
                t_new[i] = t[ptr-1]
                inds[i] = ptr - 1
                ptr += 1
                break

    return t_new, inds


def subsample_data(dt, time_sig, node_perc, sim_inds, path, save_to):
    """Undersamples time grid and nodes. t is the same for all simulations.
    
    Args:
        dt (float): Required time step in undersampled time grid.
        time_sig (float): Required noise on the undersampled time grid.
        node_perc (float): Value in range (0, 1], defines the perecentage of 
            nodes in the undersampled grid sampled from the full grid.
        sim_inds (List[int]): indices of the simulations to undersample.
        path (str): Path to the full dataset.
        save_to (str): Where to save the udnersampled dataset.
        """

    print("Reading from", path)
    data_dict = read_pickle(['t', 'x', 'u'], path=path)
    print("Data shape")
    for k, v in data_dict.items():
        print(k, v.shape)

    t = data_dict["t"][sim_inds]  # (n_sim, n_times)
    x = data_dict["x"][sim_inds]  # (n_sim, n_nodes, n_space_dim)
    u = data_dict["u"][sim_inds]  # (n_sim, n_times, n_nodes, n_field_dim)

    assert dt >= t[0, 1] - t[0, 0]

    # new_time_inds = subsample_inds(t.shape[1], time_step)
    # t_new = t[:, new_time_inds]

    _, new_time_inds = undersample_time(t[0], dt, time_sig)
    t_new = t[:, new_time_inds]  # use the same time grid for all simulations

    num_nodes = x.shape[1]
    num_new_nodes = int(node_perc*num_nodes)
    x_new = np.empty((x.shape[0], num_new_nodes, x.shape[2]))
    u_new = np.empty((u.shape[0], t_new.shape[1], num_new_nodes, u.shape[3]))
    
    for i in range(x.shape[0]):
        new_node_inds = np.random.choice(num_nodes, int(node_perc*num_nodes), replace=False)
        x_new[i] = x[i, new_node_inds, :]
        u_new_tmp = u[i, new_time_inds, :, :]
        u_new[i] = u_new_tmp[:, new_node_inds, :]

    data_dict_new = {"t": t_new, "x": x_new, "u": u_new}
    pickle_data(data_dict_new, path=save_to)

    print("New data shape")
    for k, v in data_dict_new.items():
        print(k, v.shape)
    print("Saved to", save_to)
    
    # print(data_dict_new['t'])



dt = 0.005  # 
time_sig = 0.00 / 6  # 3*std=dt/2
node_perc = 1.0
sim_inds = list(range(0, 50))
path = "./full/heat_unit_square_test/"
save_to = "./subs/heat_unit_square_test_1_0p/"  # convdiff_2pi_train

subsample_data(dt, time_sig, node_perc, sim_inds, path, save_to)
