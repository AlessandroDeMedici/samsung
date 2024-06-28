import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft
from scipy.signal import ellip, lfilter, freqz, impulse


sampling_rate = 100  # Frequenza di campionamento in Hz
update_interval = 1  # Intervallo di aggiornamento in secondi

# Segnale emulato:
freq_principale = 1.423  # Frequenza principale in Hz

# signal to noise ratio
snr = 1/10

# Creazione di un array per il segnale
time = 10
signal_length = 10 * sampling_rate  # Lunghezza del segnale in campioni (10 secondi)
t = np.linspace(0, signal_length / sampling_rate, signal_length, endpoint=False)
signal = np.sqrt(snr) * np.sin(2 * np.pi * freq_principale * t) +  np.random.randn(signal_length)  # Segnale sinusoidale con rumore

# Filtro ellittico
# frequenza di taglio del filtro (Hz)
cutoff = 3 # frequenza di taglio (Hz)

def ellip_lowpass(cutoff, fs, order=8, rp=1, rs=100):
    nyquist = 0.5 * fs
    b, a = ellip(order, rp, rs, cutoff/nyquist, btype='low', analog=False)
    return b, a

def apply_ellip_filter(data, cutoff, fs, order=8, rp=0.01, rs=40):
    b, a = ellip_lowpass(cutoff, fs, order, rp, rs)
    y = lfilter(b, a, data)
    return y

# Plotting risposta impulsiva e risposta in frequenza del filtro
b, a = ellip_lowpass(cutoff, sampling_rate)

# Calcolo risposta impulsiva
impulse = np.zeros(100);
impulse[0] = 1

response = lfilter(b, a, impulse)
w,h = freqz(b,a,worN=8000)

# Plot della risposta in frequenza
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.title('Risposta in Frequenza')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Ampiezza (dB)')
plt.grid()

# Plot della risposta impulsiva
plt.figure()
plt.stem(response)
plt.title('Risposta impulsiva')
plt.xlabel('Campioni')
plt.ylabel('Ampiezza')
plt.grid()

# Funzione per aggiornare il grafico
def update_plot(frame, time_line, filtered_time_line, fft_line, text, signal, sampling_rate, update_interval):
    global snr, cutoff, t, time

    # aggiorno il tempo
    time += update_interval

    # numero di nuovi campioni
    new_samples_count = update_interval * sampling_rate

    # aggiorno i campioni temporali
    new_t = np.linspace(time, time + 1, new_samples_count, endpoint=False)

    # Calcolare il numero di nuovi campioni
    new_samples = np.sqrt(snr) * np.sin(2 * np.pi * freq_principale * new_t) + np.random.randn(new_samples_count)

    # Aggiornare il segnale con i nuovi campioni
    signal[:-new_samples_count] = signal[new_samples_count:]
    signal[-new_samples_count:] = new_samples

    # Applicare il filtro ellittico al segnale
    filtered_signal = apply_ellip_filter(signal, cutoff, sampling_rate)

    # Calcolare la trasformata di Fourier del segnale filtrato
    N = len(filtered_signal)
    yf = fft(filtered_signal)
    xf = np.fft.fftfreq(N, 1 / sampling_rate)

    yf_abs = np.abs(yf[:N // 2])
    yf_max = np.max(yf_abs)
    yf_db = 20 * np.log10(yf_abs / yf_max)

    # Calcolare il BPM stimato
    idx_peak = np.argmax(yf_abs)
    freq_peak = np.abs(xf[idx_peak])
    bpm_estimated = freq_peak * 60

    # Aggiornare i dati del grafico del segnale nel tempo
    time_line.set_data(t, signal)
    filtered_time_line.set_data(t, filtered_signal)

    # Aggiornare i dati del grafico della trasformata di Fourier
    fft_line.set_data(xf[:N // 2], yf_db)
    text.set_text(f"bpm stimato: {bpm_estimated:.2f}")

    return time_line, filtered_time_line, fft_line, text

# Configurazione del grafico
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
(ax1, ax2), (ax3, _) = axs

# Configurazione del grafico del segnale nel tempo
ax1.set_xlim(0, signal_length / sampling_rate)
ax1.set_ylim(-3, 3)
ax1.set_title("Segnale nel Tempo")
time_line, = ax1.plot([], [], label="Segnale Originale")
ax1.legend()

# Configurazione del grafico del segnale filtrato nel tempo
ax2.set_xlim(0, signal_length / sampling_rate)
ax2.set_ylim(-3, 3)
ax2.set_title("Segnale Filtrato nel Tempo")
filtered_time_line, = ax2.plot([], [], label="Segnale Filtrato", color='orange')
ax2.legend()

# Configurazione del grafico della trasformata di Fourier in decibel
ax3.set_xlim(0, sampling_rate / 2)
ax3.set_ylim(-100, 0)  # Adattare l'asse y per la scala in decibel
ax3.set_title("Trasformata di Fourier (dB)")
fft_line, = ax3.plot([], [], color='green')
text = ax3.text(0.5, 0.9, '', transform=ax3.transAxes, ha='center')

# Nascondere l'asse vuoto
fig.delaxes(axs[1, 1])

# Creazione dell'animazione
ani = animation.FuncAnimation(fig, update_plot, fargs=(time_line, filtered_time_line, fft_line, text, signal, sampling_rate, update_interval),interval=update_interval * 1000, blit=True, cache_frame_data=False)

plt.tight_layout()
plt.show()

