import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal
import librosa
from matplotlib.ticker import EngFormatter

plt.style.use("bmh")

# Load extracted impulse response
h, fs = sf.read("/home/sbn/ir/cab/sbn/boss_katana-50_mk2_ex/qa40x/boss_katana_celstion_a-type_sm57_96000.wav")
t = np.arange(len(h)) / fs
print("Sample rate:", fs)

# Load frequency response (exported from QA403)
f_ref, H_ref = np.loadtxt("/home/sbn/ir/cab/sbn/boss_katana-50_mk2_ex/qa40x/boss_katana_celstion_a-type_sm57_reference_exported_from_qa40x.csv", skiprows=6, delimiter=",", usecols=(0, 1)).T
H_ref += 23

# Direct DFT
_h = h
H1 = np.fft.rfft(_h)
H1 = 20 * np.ma.log10(np.abs(H1))
f1 = np.linspace(0, fs / 2, len(H1))

# Periodogram (DFT of auto-correlation)
_h = np.correlate(h, h, "full")
H2 = np.fft.rfft(_h)
H2 = 10 * np.ma.log10(np.abs(H2))
f2 = np.linspace(0, fs / 2, len(H2))

# Welch's method periodogram
_h = h
f3, H3 = scipy.signal.welch(_h, fs=fs, window="boxcar", scaling="spectrum", nperseg=4096)
H3 = 10 * np.ma.log10(np.abs(H3)) + 74

# Constant-Q transform (CQT)
def spectrum_analysis_cqt(y, sr):
    fmin = 20.0
    bins_per_octave = 3
    num_octaves = 11
    n_bins = int(num_octaves * bins_per_octave)
    C = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, window="rect")
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    C_mean = C.mean(axis=1)
    spectrum_db = librosa.amplitude_to_db(np.abs(C_mean), ref=np.max)
    return freqs, spectrum_db

_h = h
f4, H4 = spectrum_analysis_cqt(_h, fs)
H4 += 30

fig, axs = plt.subplots(2, 1, figsize=(8, 5))

ax = axs[0]
ax.plot(h)

ax = axs[1]
ax.semilogx(f_ref, H_ref, label="ref", alpha=0.2, linewidth=7, color="black")
ax.semilogx(f1, H1, label="H1 (direct FFT)")
ax.semilogx(f2, H2, label="H2 (periodogram)")
ax.semilogx(f3, H3, label="H3 (Welch's method)")
ax.semilogx(f4, H4, label="H4 (CQT)")
ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
ax.set_xlim(20, 20e3)
ax.set_ylim(-50, 50)
ax.xaxis.set_major_formatter(EngFormatter())
ax.legend()

fig.tight_layout()

plt.show()
