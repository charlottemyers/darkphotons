import numpy as np

####
GSTAR = 90.0
GSTAR_S = 90.0
ME = 0.000511e-3  # GeV
ALPHA_EM = 1.0/137.0
MPL   = 1.2209e19
#####

def gD_of_alpha(alphaD):           # g_D from alpha_D
    return np.sqrt(4.0*np.pi*alphaD)

def sigma_xx_to_AA(alphaD, mchi, mA):
    gD = gD_of_alpha(alphaD)
    r = mA / mchi
    amplitude = 16*gD**4 * (1 - r**2)/(9 * (r**2 - 2)**2)
    phase_space = 9 * np.sqrt(1 - r**2) / (64* np.pi * mchi**2)
    return amplitude * phase_space

def sigmav_chichi_ee(epsilon, alphaD, mchi, mA):
    # source:1702.07716 (app)
    if mchi <= ME:
        return 0.0 # kinematically forbidden
    r   = mA / mchi
    e2  = 4.0*np.pi*ALPHA_EM
    gD  = gD_of_alpha(alphaD)
    pref = 4.0 * e2 * (epsilon**2) * (gD**2)
    num  = (2.0 + (ME**2)/(mchi**2))
    den  = (r**2 - 4.0)**2

    phase_space  = np.sqrt(1.0 - (ME**2)/(mchi**2)) / (8.0*np.pi*mchi**2)
    return pref * (num/den) * phase_space

def gammaA_ee(epsilon, mA):
    pref = epsilon**2 * ALPHA_EM / 3.0
    return pref * mA * (1 + 2*(ME**2)/(mA**2)) * np.sqrt(1 - 4*(ME**2)/(mA**2))
