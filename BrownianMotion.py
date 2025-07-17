# Brownian Motion Simulation
# 
# B_0 = 0
# B has stationary and independent increments
# B_t - B_s ~ N(0, t-s) with 0 < s < t

import matplotlib.pyplot as plt
import numpy as np

# k = 251 # days

# Uniformly Brownian Motion
def uniform_brownian_motion(k):
    B = [0]
    for i in range(k):
        B.append(B[-1] + np.random.normal(0, 1))
    return B

# Non-uniform Brownian Motion
def non_uniform_brownian_motion(k, times):
    B = [0]
    for i in range(k-1):
        B.append(B[-1] + np.random.normal(0, np.sqrt(times[i+1]-times[i])))
    return B



# TODO:
# See if logvolatility follows fbm with H = 0.5


# Simulate fBm using Davies-Harte algorithm
# fBm is a generalization of Brownian motion where increments are not necessarily independent
# H is the Hurst exponent, 0 < H < 1
# H = 0.5 is Brownian motion
# H < 0.5 possitively correlates increments ("streaky")
# H > 0.5 negatively correlates increments ("jittery")
# H = 1 is deterministic

def DaviesHarte_brownian_motion(T, N, H):
    # Generate a time grid
    t = np.linspace(0, T, N+1)

    # Define the autocovariance
    gamma = lambda k: 0.5 * (abs(k+1)**(2*H) - 2*abs(k)**(2*H) + abs(k-1)**(2*H)) if k <= N else 0

    # Define circulant vector
    c = np.concatenate([np.array([gamma(k) for k in range(N+1)]), np.array([gamma(k) for k in range(N-1, 0, -1)])])

    # Take FFT to find (positive real) eigenvalues
    L = np.fft.fft(c).real
    if not np.allclose(np.fft.fft(c).imag, 0, atol=1e-10):
        raise ValueError("FFT has imaginary components")
    
    if np.any(L < 0):
        raise ValueError("FFT has negative eigenvalues")

    # FFT length
    M = 2 * N

    Z = np.zeros(M, dtype=np.complex128)
    Z[0] = np.sqrt(L[0]) * np.random.normal()
    Z[N] = np.sqrt(L[N]) * np.random.normal()

    X = np.random.normal(0, 1, N-1)
    Y = np.random.normal(0, 1, N-1)

    for k in range(1, N):
        Z[k] = np.sqrt(L[k]) * (X[k-1] + 1j * Y[k-1])
        Z[M-k] = np.conj(Z[k])

    # Inverse FFT to get fGn
    fGn = np.fft.ifft(Z).real[:N] * (T / N) ** H * np.sqrt(M)

    # Cumulative sum to get fBm
    fbm = np.concatenate([[0], np.cumsum(fGn)])

    return fbm

# Cholesky decomposition method
def cholesky_brownian_motion(T, N, H):

    # Define the autocovariance
    gamma = lambda s,t: 0.5 * (s**(2*H) + t**(2*H) - abs(s-t)**(2*H))

    # Generate a time grid
    times = np.linspace(0, T, N+1)

    G = np.zeros((N+1, N+1))

    for i in range(N+1):
        for j in range(N+1):
            if i == j:
                G[i,j] = times[i]**(2*H)
            else:   
                G[i,j] = gamma(times[i], times[j])

    G = 0.5 * (G+G.T)
    epsilon = 1e-10
    G = G + epsilon * np.eye(N+1)
    
    # Generate a random vector
    Z = np.random.normal(0, 1, N+1)

    # Generate a Brownian motion
    B = np.linalg.cholesky(G) @ Z

    return B

# TODO:
# Hosking method
def Hosking_brownian_motion(T, N, H):
    gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
    
    L = np.zeros((N,N))
    X = np.zeros(N)
    V = np.random.standard_normal(size=N)

    L[0,0] = 1.0
    X[0] = V[0]
    
    L[1,0] = gamma(1,H)
    L[1,1] = np.sqrt(1 - (L[1,0]**2))
    X[1] = np.sum(L[1,0:2] @ V[0:2])
    
    for i in range(2,N):
        L[i,0] = gamma(i,H)
        
        for j in range(1, i):         
            L[i,j] = (1/L[j,j])*(gamma(i-j,H) - (L[i,0:j] @ L[j,0:j]))

        L[i,i] = np.sqrt(1 - np.sum((L[i,0:i]**2))) 
        X[i] = L[i,0:i+1] @ V[0:i+1]
    fBm = np.cumsum(X)*(N**(-H))

    fBm = np.concatenate([[0], fBm])

    return (T**H)*(fBm)

fbm_0p9 = DaviesHarte_brownian_motion(1, 1000, 0.9)
fbm_0p75 = DaviesHarte_brownian_motion(1, 1000, 0.75)
fbm_0p5 = DaviesHarte_brownian_motion(1, 1000, 0.5)
fbm_0p25 = DaviesHarte_brownian_motion(1, 1000, 0.25)
fbm_0p1 = DaviesHarte_brownian_motion(1, 1000, 0.1)

plt.plot(fbm_0p9, label='H=0.9')
plt.plot(fbm_0p75, label='H=0.75')
plt.plot(fbm_0p5, label='H=0.5')
plt.plot(fbm_0p25, label='H=0.25')
plt.plot(fbm_0p1, label='H=0.1')
plt.legend()
plt.title('Davies-Harte algorithm')
plt.show()

fbm_0p9_cholesky = cholesky_brownian_motion(1, 1000, 0.9)
fbm_0p75_cholesky = cholesky_brownian_motion(1, 1000, 0.75)
fbm_0p5_cholesky = cholesky_brownian_motion(1, 1000, 0.5)
fbm_0p25_cholesky = cholesky_brownian_motion(1, 1000, 0.25)
fbm_0p1_cholesky = cholesky_brownian_motion(1, 1000, 0.1)

plt.plot(fbm_0p9_cholesky, label='H=0.9')
plt.plot(fbm_0p75_cholesky, label='H=0.75')
plt.plot(fbm_0p5_cholesky, label='H=0.5')
plt.plot(fbm_0p25_cholesky, label='H=0.25')
plt.plot(fbm_0p1_cholesky, label='H=0.1')
plt.legend()
plt.title('Cholesky decomposition')
plt.show()

fbm_0p9_hosking = Hosking_brownian_motion(1, 1000, 0.9)
fbm_0p75_hosking = Hosking_brownian_motion(1, 1000, 0.75)
fbm_0p5_hosking = Hosking_brownian_motion(1, 1000, 0.5)
fbm_0p25_hosking = Hosking_brownian_motion(1, 1000, 0.25)
fbm_0p1_hosking = Hosking_brownian_motion(1, 1000, 0.1)

plt.plot(fbm_0p9_hosking, label='H=0.9')
plt.plot(fbm_0p75_hosking, label='H=0.75')
plt.plot(fbm_0p5_hosking, label='H=0.5')
plt.plot(fbm_0p25_hosking, label='H=0.25')
plt.plot(fbm_0p1_hosking, label='H=0.1')
plt.legend()
plt.title('Hosking method')
plt.show()






