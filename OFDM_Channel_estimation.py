import numpy as np
import math
from numpy.fft import fft, ifft
from ModulationPy import QAMModem
import matplotlib.pyplot as plt
import scipy
from scipy import special, interpolate, signal
import time
tic = time.time()

# Configuraçoes iniciais
N = 131072                       # Numero de subportadoras
Mod = 16                         # Ordem da modulacao
M = 1                            # Numero de subsimbolos
Nset = np.arange(20, 65531, 1)   # Alocacao de algumas subportadoras
Non = len(Nset)                  # Numero de portadoras alocadas
Np = 33                        # Numero de portadoras pilotos
pilotValue = 1                   # The known value each pilot transmits
SNR = np.arange(5,45,3)          # Valores de SNR em dB
snr = 10**(SNR/10)               # Valores de SNR linear
L = math.sqrt(Mod)
mu = 4*(L-1)/L                   # Número médio de vizinhos
E = 3/(L**2-1)

# Pilotos
allCarriers2 = np.arange(N)      # Todas as portadoras incluindo as que não serã utilizadas
allCarriers = np.arange(Non)

pilotCarriers = allCarriers[::Non//Np] # Pilots is every (K/P)th carrier.
print(len(pilotCarriers))
# For convenience of channel estimation, let's make the last carriers also be a pilot
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
Np = Np+1

# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers)

#print (f'Todas as portadoras: {allCarriers}')
#print (f'Portadoras pilotos: {pilotCarriers}')
#print (f'Portadoras de dados: {dataCarriers}')
plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')

# Modulador QAM
c = np.random.randint(0, Mod, size=Non-len(pilotCarriers))
modem = QAMModem(Mod, bin_input=False, soft_decision=False, bin_output=False)
symbol = modem.modulate(c) # modulation


s = np.zeros(Non, dtype=complex)  # Todas as N portadoras
s[dataCarriers] = symbol       # allocate the pilot subcarriers
P = np.sum(abs(s)**2)/len(dataCarriers)
s[pilotCarriers] = pilotValue  # Alocação das portadoras Pilotos

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

# Inicialização
pe_teor = np.zeros(len(snr))
pe_sim = np.zeros(len(snr))
pe_simp = np.zeros(len(snr))
pe_sim_awgn = np.zeros(len(snr))
pe_teor_ep = np.zeros(len(snr))
MSE = np.zeros(len(snr))
erros = np.zeros(len(snr))
errosp = np.zeros(len(snr))
erros_awgn = np.zeros(len(snr))

# Resposta do canal
channelResponse = np.array([1, 0.7])  # the impulse response of the wireless channel
H_exact = np.fft.fft(channelResponse, N)
#plt.figure(figsize= (5,5))
#plt.plot(allCarriers2, abs(H_exact), label='Real Channel Response')


for idx in range(0,len(snr)):
    n0 = P/snr[idx]
    noise = np.sqrt(n0/2)*(np.random.randn(len(d)) + 1j*np.random.randn(len(d)))
    y2 = st+noise
    # Passando pelo canal
    convolved = np.convolve(st, channelResponse, mode = 'same')
    y = convolved+noise

    rf = 1/(np.sqrt(N))*np.fft.fft(y)
    rf_awgn = 1/np.sqrt(N)*np.fft.fft(y2)
    # Demapeando os símbolos complexos
    rf_rx= demapeamento(rf,nset,mset)
    rf_rx_awgn= demapeamento(rf_awgn,nset,mset)

    # Estimação do canal
    pilots = rf_rx[pilotCarriers]
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)

    # Equalização
    rf_eq = rf_rx / Hest
    rf_eqp = rf_rx / H_exact[allCarriers]
    # Removendo as portadoras pilotos
    rf1 =rf_eq[dataCarriers]
    rf2 = rf_eqp[dataCarriers]
    rf3 = rf_rx_awgn[dataCarriers]
    
    # simbolos estimados
    c_est = modem.demodulate(rf1)
    c_estp = modem.demodulate(rf2)
    c_est_awgn =  modem.demodulate(rf3)
    # Contagem  de erros
    erros[idx] = np.sum(c != c_est)
    errosp[idx] = np.sum(c != c_estp)
    erros_awgn[idx] = np.sum(c != c_est_awgn)
    # Probabilidade de erro Teórica OFDM em Canal AWGN
    pe_teor[idx] = mu/2*special.erfc(np.sqrt(E*snr[idx])/np.sqrt(2))

    # Probabilidade de erro Simulada OFDM em Canal AWGN
    pe_sim[idx] = erros[idx]/len(c_est)
    pe_simp[idx] = errosp[idx]/len(c_estp)
    pe_sim_awgn[idx] = erros_awgn[idx]/len(c_est_awgn)
    # Taxa de erro de bit téorica OFDM Seletivo
    pe_teor_ep[idx]  = (mu/(2*len(H_exact[Nset])))*sum(special.erfc(np.sqrt(abs(H_exact[Nset])**2*E*snr[idx])/np.sqrt(2)))

    MSE[idx] = 1/len(Hest)*sum(abs(H_exact[Nset]-Hest))**2

plt.figure(figsize= (5,5))
plt.plot(rf1.real, rf1.imag, 'bo')
plt.title(f'Estimação Linear com SNR = {SNR[idx]} dB')

plt.figure(figsize= (5,5))
plt.plot(rf2.real, rf2.imag, 'bo')
plt.title('Estimação Perfeita')

plt.figure(figsize= (5,5))
plt.plot(rf3.real, rf3.imag, 'bo')
plt.title('AWGN')

plt.figure(figsize= (8,5))
plt.plot(Nset, abs(H_exact[Nset]), label='Real Channel Response')
plt.stem(pilotCarriers+min(Nset), abs(Hest_at_pilots), use_line_collection  = True, label='Pilot estimates')
plt.plot(Nset, abs(Hest), label='Estimated channel via interpolation')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.ylim(0,2)

fig = plt.figure(figsize=(8,5))
plt.plot(SNR, pe_teor_ep, label='OFDM-Seletive- Theoretical')
plt.scatter(SNR, pe_sim, facecolor='None', edgecolor='r', label='OFDM-Selective Channel- Linear Channel Estimation- Simulation')
plt.scatter(SNR, pe_sim_awgn, facecolor='None', edgecolor='b', label='OFDM-AWGN Channel- Simulation')
plt.scatter(SNR, pe_simp, facecolor='None', edgecolor='g', label='OFDM-Selective Channel- Perfect Channel Estimation- Simulation')
plt.plot(SNR, pe_teor, label='OFDM-AWGN Channel- Theoretical')
plt.yscale('log')
plt.xlabel('SNR (dB)')
plt.ylabel('Symbol Error Rate')
plt.grid()
plt.legend(fontsize=10)
plt.xlim([10, 30])
plt.ylim([1e-5, 1])

toc = time.time()
tempo = toc-tic
print(f'A simulação demorou {tempo} segundos')
plt.show()