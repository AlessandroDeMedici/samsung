import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ellip, lfilter, freqz

# Parametri del filtro
order = 8       # Ordine del filtro
rp = 1          # Ripple in banda passante (dB)
rs = 40         # Attenuazione in banda eliminata (dB)
fs = 100        # Frequenza di campionamento (Hz)
fc = 3          # Frequenza di taglio (Hz)

# Calcolo delle frequenze normalizzate
nyquist = 0.5 * fs
low = fc / nyquist

# Progettazione del filtro ellittico
b, a = ellip(order, rp, rs, low, btype='low', analog=False)

# Creazione della risposta impulsiva
impulse = np.zeros(100)
impulse[0] = 1  # Impulso unitario

# Applicazione del filtro alla risposta impulsiva
response = lfilter(b, a, impulse)
w, h = freqz(b,a,worN=8000)

# Generazione di un segnale casuale con banda di 20 Hz
n_samples = 500
t = np.arange(n_samples) / fs
input_signal = np.random.randn(n_samples)

# Rimozione della componente DC (detrending)
input_signal -= np.mean(input_signal)

# Applicazione del filtro al segnale casuale
output_signal = lfilter(b, a, input_signal)

# Calcolo della trasformata di Fourier dei segnali
input_fft = np.fft.fft(input_signal)
output_fft = np.fft.fft(output_signal)
frequencies = np.fft.fftfreq(n_samples, 1/fs)

# Trova la componente massima nella trasformata di Fourier del segnale di uscita
max_idx = np.argmax(np.abs(output_fft))
max_frequency = np.abs(frequencies[max_idx]) # segnale reale quindi trasformata pari
max_amplitude = np.abs(output_fft[max_idx])

# battito cardiaco stimato
bpm = max_frequency * 60

# Stampa la componente massima
print(f"Massima ampiezza nella trasformata di Fourier del segnale di uscita: {max_amplitude} a {max_frequency} Hz\nBattito Cardiaco Stimato: {bpm} bpm")

# segnale prima del filtraggio
plt.figure()
plt.plot(t, input_signal, label='Segnale di ingresso')
plt.title('Segnale di ingresso nel dominio del tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.grid()
plt.legend()


# spettro ampiezza del segnale in ingresso
plt.figure()
plt.plot(frequencies[:n_samples // 2], np.abs(input_fft)[:n_samples // 2], label='Spettro di ingresso')
plt.title('Trasformata di Fourier del segnale di ingresso')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Ampiezza')
plt.grid()

# spettro ampiezza del segnale in uscita
plt.figure()
plt.plot(frequencies[:n_samples // 2], np.abs(output_fft)[:n_samples // 2], label='Spettro filtrato', color='orange')
plt.plot(max_frequency, max_amplitude, 'ro')  # Evidenzia il punto massimo in rosso
plt.title('Trasformata di Fourier del segnale filtrato')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Ampiezza')
plt.grid()
plt.legend()


# segnale dopo il filtraggio
plt.figure()
plt.plot(t, output_signal, label='Segnale filtrato', color='orange')
plt.title('Segnale filtrato nel dominio del tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.grid()
plt.legend()


# Risposta impulsiva
plt.figure()
plt.stem(np.arange(0, len(response)) / fs, response)
plt.title("Risposta impulsiva del filtro ellittico")
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.grid()

# Risposta in frequenza
plt.figure()
plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
plt.title("Risposta in frequenza del filtro ellittico")
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Ampiezza')
plt.grid()


plt.show()

