import numpy as np
import math
from numpy.fft import fft, ifft
from ModulationPy import QAMModem
import matplotlib.pyplot as plt
from scipy import special
import time
tic = time.time()

# Configuraçoes iniciais
N = 131072                       # Numero de subportadoras
Mod = 16                         # Ordem da modulacao
M = 1                            # Numero de subsimbolos
Nset = np.arange(20, 65531, 1)   # Alocacao de algumas subportadoras
Non = len(Nset)                  # Numero de portadoras alocadas
Kd = 1000                        # Kd
Np = 65                          # Numero de portadoras pilotos
pilotValue = 1                   # The known value each pilot transmits
SNR = np.arange(10,25,1)         # Valores de SNR em dB
snr = 10**(SNR/10)               # Valores de SNR linear
L = math.sqrt(Mod)
mu = 4*(L-1)/L                   # Número médio de vizinhos
E = 3/(L**2-1)

c = np.random.randint(0, Mod, size=Non)
modem = QAMModem(Mod, bin_input=False, soft_decision=False, bin_output=False)
s = modem.modulate(c) # modulation

P = np.sum(abs(s)**2)/len(s)

# Mapeamento dos símbolos nas portadoras

def mapeamento(s,Nset,N):
   nset = Nset % N
   assert len(nset) <= N
   mset = 1
   nset = nset+1
   res1 = np.zeros( N, dtype=complex)
   res1[nset]= s
   d = res1
   return  d, nset, mset

def demapeamento(rf,nset,mset):
    rf1 = rf[nset]
    return rf1


d, nset,mset = mapeamento(s,Nset,N)


st = np.sqrt(N)*np.fft.ifft(d)
Pt = np.sum(abs(st)**2)/len(st)
pe_teor = np.zeros(len(snr))
pe_sim = np.zeros(len(snr))
erros = np.zeros(len(snr))

for idx in range(0,len(snr)):
    n0 = P/snr[idx]
    noise = np.sqrt(n0/2)*(np.random.randn(len(d)) + 1j*np.random.randn(len(d)))

    y = st+noise

    rf = 1/np.sqrt(N)*np.fft.fft(y)

    # Demapeando os símbolos complexos
    rf1= demapeamento(rf,nset,mset)

    # simbolos estimados
    c_est = modem.demodulate(rf1)
    # Contagem  de erros
    erros[idx] = np.sum(c != c_est)

    # Probabilidade de erro Teórica OFDM em Canal AWGN
    pe_teor[idx] = mu/2*special.erfc(np.sqrt(E*snr[idx])/np.sqrt(2))

    # Probabilidade de erro Simulada OFDM em Canal AWGN
    pe_sim[idx] = erros[idx]/len(c_est)



plt.figure(figsize= (5,5))
plt.plot(rf1.real, rf1.imag, 'bo')
plt.show()

fig = plt.figure(figsize=(7,5))
plt.scatter(SNR, pe_sim, facecolor='None', edgecolor='r', label='OFDM-AWGN Simulation')
plt.plot(SNR, pe_teor, label='OFDM-AWGN Theoretical')
plt.yscale('log')
plt.xlabel('SNR (dB)')
plt.ylabel('Symbol Error Rate')
plt.grid()
plt.legend(fontsize=14)
plt.xlim([10, 30])
plt.ylim([1e-5, 1])
plt.show()

toc = time.time()
tempo = toc-tic
print(f'A simulação demorou {tempo} segundos')
