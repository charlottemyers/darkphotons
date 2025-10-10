import numpy as np

def gstar_interp(T, path = "/Users/charlottemyers/projects/ctp/heffBW.dat"):
    #if T < T_data.min() or T > T_data.max():
        #return GSTAR
    data = np.loadtxt(path)
    T_data = data[:,0]
    g_eff = data[:,1]

    return np.interp(T, T_data, g_eff)

def m_sv_interp(T, path = "/Users/charlottemyers/projects/ctp/thermal_xsec.txt"):
    m = []
    sv = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                m_str, sv_str = line.strip().split(",")
                m.append(float(m_str))
                sv.append(float(sv_str))
    print(50)
    return np.interp(T, m, sv)


def returnk (T, path = "/Users/charlottemyers/projects/ctp/heffBW.dat"):
    return T
