# Signal Processing Notebook

This project contains a Jupyter notebook, `Signal_Processing.ipynb`, that demonstrates core audio signal-processing techniques on a speech waveform using `librosa`, `numpy`, and `matplotlib`.

## Notebook Overview

The notebook analyzes an audio file loaded with:

```python
y, sr = librosa.load('./hello.wav')
```

Where:

- `y` is the audio waveform amplitude over time
- `sr` is the sampling rate of the audio

The notebook then extracts and visualizes several important speech/audio features.

## Concepts Covered In The Code

### 1. Waveform / Signal Visualization

The first plot shows the raw audio signal:

```python
plt.plot(y)
```

This helps us inspect how amplitude changes across time. It is the most direct time-domain representation of the sound.

### 2. Fourier Transform and Frequency Spectrum

The notebook computes a short-time Fourier transform using:

```python
ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft+1))
```

This is based on the **Fourier Transform**, which converts a signal from the **time domain** into the **frequency domain**.

Why it matters:

- It reveals which frequencies are present in the audio
- It helps identify pitch-related and timbre-related patterns
- It is the foundation for more advanced features like spectrograms, mel spectrograms, and MFCCs

The plotted spectrum shows frequency content and magnitude for a slice of the signal.

### 3. Spectrogram

The spectrogram is computed from the STFT:

```python
spec = np.abs(librosa.stft(y, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)
```

A **spectrogram** shows:

- time on the x-axis
- frequency on the y-axis
- intensity/amplitude as color

Why it matters:

- It shows how frequency content changes over time
- It is one of the most common visual tools in speech and audio analysis
- It helps detect voiced regions, harmonics, and transient events

### 4. Mel Spectrogram

The notebook also computes a mel spectrogram:

```python
mel_spect = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=2048,
    hop_length=1024
)
```

A **mel spectrogram** maps frequencies onto the **mel scale**, which better matches human auditory perception.

Why it matters:

- Human hearing is more sensitive to lower-frequency changes than higher-frequency changes
- Mel-scaled features are widely used in speech recognition and speaker analysis
- It provides a perceptually meaningful version of the standard spectrogram

### 5. MFCCs (Mel-Frequency Cepstral Coefficients)

The notebook extracts 13 MFCC features:

```python
mfccs = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=13
)
```

**MFCCs** are compact audio features derived from the mel spectrogram and a cepstral transform.

Why they matter:

- They capture the spectral shape of speech
- They are standard features in speech recognition, speaker identification, and emotion analysis
- They reduce dimensionality while preserving useful information

The notebook includes:

- an MFCC heatmap over time
- individual plots for each of the 13 MFCC coefficients

This helps show how speech characteristics vary frame by frame.

### 6. Zero-Crossing Rate (ZCR)

The code computes:

```python
zcr = librosa.feature.zero_crossing_rate(y)
```

**Zero-crossing rate** measures how often the waveform changes sign.

Why it matters:

- Noisy or unvoiced sounds often have a higher ZCR
- Voiced sounds usually have a lower ZCR
- It helps distinguish speech, music, and noise in basic audio classification tasks

### 7. RMS Energy

The notebook computes energy with:

```python
rms = librosa.feature.rms(y=y)
```

**RMS energy** measures the signal strength or loudness over short frames.

Why it matters:

- It helps locate active speech regions
- It can separate silence from non-silence
- It is useful in segmentation, voice activity detection, and emphasis analysis

In the notebook, the energy plot shows how strong the signal is over time.

### 8. Pitch Detection

Pitch is estimated with:

```python
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=50,
    fmax=300
)
```

This estimates the **fundamental frequency (`f0`)** of the signal.

Why pitch detection matters:

- It helps analyze intonation and speaking patterns
- It is useful for speaker analysis
- It can support emotion detection
- It is also useful in singing voice analysis

The plot shows how the detected pitch changes over time in Hz.

### 9. Voice Activity Detection (VAD)

The notebook performs simple voice activity detection using:

```python
intervals = librosa.effects.split(y, top_db=20)
```

This returns intervals where the signal is above a threshold relative to silence.

Why it matters:

- It helps isolate spoken segments
- It removes silent regions before feature extraction
- It improves efficiency in downstream speech-processing tasks

## Step-By-Step Flow Of The Notebook

The notebook follows this progression:

1. Load audio with `librosa.load`
2. Plot the waveform in the time domain
3. Compute a Fourier-based spectrum using STFT
4. Visualize a spectrogram
5. Visualize a mel spectrogram
6. Extract and display MFCC features
7. Compute zero-crossing rate
8. Compute RMS energy
9. Detect pitch using `librosa.pyin`
10. Detect voiced/non-silent intervals

## Libraries Used

- `librosa` for audio loading, transformation, and feature extraction
- `numpy` for numerical operations
- `matplotlib` for plotting

## Key Learning Outcome

This notebook is a compact introduction to practical speech and audio analysis. It demonstrates both:

- **time-domain analysis**: waveform, zero-crossing rate, RMS energy
- **frequency-domain analysis**: Fourier transform, spectrum, spectrogram, mel spectrogram, MFCCs, pitch

Together, these features form the foundation of many applications in:

- speech recognition
- speaker identification
- audio classification
- music information retrieval
- voice activity detection

## File In This Project

- `Signal_Processing.ipynb`: the main notebook containing the analysis and visualizations
- `README.md`: project explanation and concept summary
