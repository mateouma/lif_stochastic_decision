import numpy as np
import matplotlib.pyplot as plt

# def intensity(v, B=1, v_th=1):
#     return np.exp(v-B)

def intensity(v, B=1, v_th=1, p=1):
    x = v - v_th 

    if len(np.shape(x)) > 0:
        x[x < 0] = 0
    elif x < 0:
        x = 0
    else: pass
    
    return B * x**p


# def intensity(v, B=1, v_th=1, p=1):
#     return np.exp(B*(v-v_th))

def intensity_match_linear_reset_mft(v, B=1, v_th=1, p=1):

    x = v - v_th 

    if len(np.shape(x)) > 0:
        x[x < 0] = 0
    elif x < 0:
        x = 0
    else: pass
    
    return B * np.sqrt(v) * x**p

def create_connect_mat(Ne, Ni, f, wstrong, sparse_prob=None, plot=False):
    N = Ne + Ni
    wweak = 1 - ((f*(wstrong - 1)) / (1 - f))
    fNe = int(f * Ne)

    Jmat = np.zeros((N, N))

    # wj = 1
    Jmat[:,:Ne] = 1 # from all excitatory to all else

    # wj = w+
    Jmat[:fNe, :fNe] = wstrong # from selective A to selective A
    Jmat[fNe:(2*fNe), fNe:(2*fNe)] = wstrong # from selective B to selective B

    # wj = w-
    Jmat[fNe:(2*fNe), :fNe] = wweak # from selective A to selective B
    Jmat[:fNe, fNe:(2*fNe)] = wweak # from selective B to selective A
    Jmat[:(2*fNe),(2*fNe):Ne] = wweak # from nonselective to selective

    Jmat[:,Ne:] = 1 # from inhibitory to all else

    # Jmat_max = np.max(np.abs(Jmat))

    if sparse_prob is not None:
        prob_mat = np.random.binomial(n=1, p=sparse_prob, size=(N,N)) 
        Jmat *= prob_mat
        Jmat /= sparse_prob # renormalize

    if plot:
        fig, ax = plt.subplots()
        pmat = ax.imshow(Jmat, cmap='Reds', interpolation='nearest')
        fig.colorbar(pmat, ax=ax)

        plt.show()

    return Jmat


if __name__ == '__main__':
    pass