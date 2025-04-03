import numpy as np

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

def gen_bg_noise(rate, dt, N, sim_len):
    Nt = int(sim_len / dt)
    spks = (np.random.rand(Nt, N) < (rate*dt))*1

    return spks



if __name__ == '__main__':
    pass