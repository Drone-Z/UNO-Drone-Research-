# WiFi CSI Drone Detection — Comprehensive Research Reference
**Topic:** WiFi Channel State Information (CSI) for Drone vs. Human Classification  
**System:** Two ESP32 devices, ~50 ft apart, 10 packets/second  
**Base Code:** [espressif/esp-csi — get-started](https://github.com/espressif/esp-csi/tree/master/examples/get-started)  
**Compiled:** March 2026 | Student Z 

---

## Table of Contents

1. [Introduction & Background](#1-introduction--background)
2. [Experiments to Run: Data Collection Protocol](#2-experiments-to-run-data-collection-protocol)
3. [Optimal Distances and Device Placement](#3-optimal-distances-and-device-placement)
4. [Data Labeling and Annotation Best Practices](#4-data-labeling-and-annotation-best-practices)
5. [Machine Learning Approaches](#5-machine-learning-approaches)
6. [ESP32-Specific Considerations and Limitations](#6-esp32-specific-considerations-and-limitations)
7. [Dataset Considerations: Size, Balance, and Splits](#7-dataset-considerations-size-balance-and-splits)
8. [Signal Preprocessing Pipeline](#8-signal-preprocessing-pipeline)
9. [Key Distinctions: Drone vs. Human CSI Signatures](#9-key-distinctions-drone-vs-human-csi-signatures)
10. [Recommended Research Workflow](#10-recommended-research-workflow)
11. [References](#11-references)

---

## 1. Introduction & Background

WiFi Channel State Information (CSI) describes how a radio signal propagates from a transmitter to a receiver across each OFDM subcarrier. Unlike RSSI (a single scalar value), CSI provides per-subcarrier amplitude and phase measurements, giving a rich multi-dimensional fingerprint of the wireless channel. Any object that moves within or near the signal path modifies the multipath propagation, causing measurable perturbations in the CSI values.

This property has been exploited for a wide variety of device-free sensing tasks: human activity recognition (HAR), fall detection, gesture recognition, breathing and heart-rate monitoring, indoor localization, and — the focus of this project — detection and classification of unmanned aerial vehicles (UAVs) versus humans.

The ESP32 microcontroller (Espressif Systems) exposes raw CSI through its Wi-Fi driver API, making it one of the most accessible and low-cost platforms for CSI-based sensing research.

**This system uses:**
- Two ESP32 nodes separated by ~50 feet (~15.2 meters)
- 10 packets per second collection rate
- Standard 2.4 GHz WiFi channel (HT20 or HT40 bandwidth)
- Goal: binary or multi-class classification of drone activity vs. human activity vs. empty environment

---

## 2. Experiments to Run: Data Collection Protocol

A robust dataset requires deliberate, well-labeled experimental runs covering all classes of interest under diverse conditions.

### 2.1 Empty / Baseline Class
**Purpose:** Establish a noise floor and background channel fingerprint.

- Record 10–15 minutes of "empty field" with no humans or drones within 20–30 meters of the link
- Capture at multiple times of day (morning, afternoon, evening) to account for environmental drift
- Log weather conditions if outdoors (wind speed, temperature, humidity)
- This class is critical as a negative sample and helps calibrate system sensitivity

### 2.2 Human Activities Class
**Purpose:** Generate a representative human motion signature across natural behavioral patterns.

**Walking:**
- Walk perpendicular to the TX-RX link at midpoint (maximum Doppler)
- Walk parallel to the TX-RX link (minimum Doppler, tests sensitivity)
- Walk at different speeds: slow (~0.5 m/s), normal (~1.2 m/s), fast (~2 m/s)
- Walk at different distances from the link: 1 m, 3 m, 5 m, 10 m from midpoint
- Multiple subjects: vary height, weight, gait (at least 5–10 people)

**Standing / Stationary:**
- Person standing still at various positions along the link
- Person standing and breathing (subtle motion — micro-movement)
- Person standing and performing small arm gestures

**Running:**
- Jogging/running through the sensing area
- Start/stop events (acceleration and deceleration)

**Other Activities:**
- Sitting down and standing up
- Arm waving / large gestures
- Falling (important for false-positive rejection)
- Person entering and exiting the sensing area
- Two people walking simultaneously (multi-target scenarios)

> **Recommended duration:** 2–5 minutes per subject per condition (minimum), yielding ~30+ seconds of usable data per segment after cropping.

### 2.3 Drone / UAV Activities Class
**Purpose:** Generate diverse UAV signatures covering typical operational profiles.

**Hovering:**
- Hover stationary at various heights: 1 m, 2 m, 3 m, 5 m, 10 m
- Hover directly over the TX-RX midpoint (maximum reflective effect)
- Hover at offset positions: 1 m, 3 m, 5 m laterally from the link midpoint
- Duration: minimum 3 minutes per height/position combination
- *Rationale: DJI Mavic-class drone at ~5,000 RPM with 2-bladed props → BPF ≈ 166 Hz*

**Flying — Horizontal Passes:**
- Fly across the TX-RX link at slow (~2 m/s), normal (~5–8 m/s), and high speed (~10+ m/s)
- Multiple flight altitudes: 1 m, 2 m, 5 m, 10 m AGL
- Flights along the link axis (parallel), perpendicular (crossing), and at 45-degree angles

**Takeoff and Landing:**
- Capture the takeoff transient (motor spool-up, lift-off)
- Capture landing sequence
- These transition events may have distinctive signatures

**Drone Approaching / Departing:**
- Fly toward the midpoint from 50+ meters away
- Fly away from the sensing area
- Tests detection at the boundary of the sensing zone

**Multiple Drone Models (if available):**
- DJI Mavic Mini / Mini 2 / Mini 3 (small, ~249g)
- DJI Mavic Air 2 or Air 3 (medium consumer)
- DJI Phantom 4 (larger quadrotor)
- Racing FPV drone (different rotor characteristics)
- *Rationale: Different drones have different motor speeds, blade counts, and Doppler signatures — multi-model data prevents overfitting*

**Drone + Human Simultaneously:**
- Drone hovering while human walks nearby (tests separation capability)
- Most challenging and most operationally realistic scenario

### 2.4 Environmental Variation
- Run experiments at different times of day
- Indoor vs. outdoor (if use case spans both)
- Rainy/windy days (affects drone stability and introduces additional vibration)
- Different antenna orientations (rotate ESP32 boards 90 degrees, test impact)
- Presence of other WiFi networks (test robustness)

### 2.5 Minimum Recommended Dataset Size

| Class | Target Duration | Estimated Windows (2s, 50% overlap, 10 Hz) |
|---|---|---|
| Empty environment | 30–60 minutes | ~1,800–3,600 |
| Human activities | 60+ minutes | ~3,600+ |
| Drone activities | 60+ minutes | ~3,600+ |

> At 10 packets/second, 1 minute yields ~600 CSI vectors → ~60 training windows/minute. **10,000+ windows per class recommended for deep learning.**

---

## 3. Optimal Distances and Device Placement

### 3.1 The Fresnel Zone Model

WiFi CSI sensing effectiveness is governed by the Fresnel zone model. The first Fresnel zone radius at the midpoint is:

```
r = sqrt(lambda × d / 4)
where lambda = 0.125 m (at 2.4 GHz)
      d = TX-RX separation
```

| TX-RX Distance | FFZ Radius at Midpoint | Notes |
|---|---|---|
| 3 m (10 ft) | ~0.31 m | Very narrow, highly sensitive |
| 8 m (26 ft) | ~0.50 m | Common lab setup |
| 15.2 m (50 ft) | ~0.69 m | Your outdoor setup |
| 80.5 m (264 ft) | ~1.59 m | Range test |

> **Note:** While the Fresnel zone model explains sensitivity to subtle motion, drones and walking humans create MUCH larger perturbations that are often detectable well beyond the strict first Fresnel zone.

### 3.2 What the Literature Says About Distance

- **Wu et al. (Springer, 2022):** At TX-RX distances of 0.5–4 m, sensing area forms an ellipse with long axis ~4 m and short axis ~3 m. Beyond ~5 m, Fresnel zone widens but sensitivity to fine-grained motion decreases.
- **Di Seglio et al. (IET, 2024):** Passive WiFi radar approach detects gross human and drone motion at practical indoor distances using 2.4 and 5 GHz bands.
- **Niu et al. (ACM IMWUT, 2022):** Proper placement can expand sensing coverage by ~200%. Best placement puts TX and RX at the midpoints of two long room walls.
- **"Wall-Proximity Matters" (arXiv, 2024):** Placing devices within ~0.5 m of walls can expand sensing coverage because walls act as reflectors.

### 3.3 Recommendations for Your 50 ft Setup

**Pros of 50 ft separation:**
- Large sensing zone between the nodes
- Drone flying between the nodes creates strong Doppler perturbations
- More spatial coverage for detecting various flight paths
- Suitable for outdoor deployment

**Cons:**
- Fine-grained motion (breathing, subtle vibration) may be below detection threshold
- Signal amplitude decreases with distance (path loss), reducing SNR
- Phase information becomes less reliable at longer distances

**Practical Recommendations:**
1. Position nodes with clear line-of-sight (LOS) — avoid obstacles in the propagation path
2. Mount both nodes at the same height (1–1.5 m above ground)
3. Orient antennas vertically for omnidirectional horizontal coverage
4. The primary sensing sweet spot is the midpoint and the elliptical zone between nodes
5. For comparison, also test with nodes placed 5–10 m apart
6. Indoors: 5–10 m is more effective. Outdoors: 15 m is reasonable

---

## 4. Data Labeling and Annotation Best Practices

### 4.1 Metadata to Capture Alongside CSI Values

**Mandatory Fields:**
- `label` — e.g., `empty`, `human_walking`, `drone_hovering`
- `timestamp` (UTC, ISO 8601)
- `session_id` — unique identifier for the recording session
- `subject_id` — anonymized identifier for human subjects
- `drone_model` — e.g., `DJI_Mavic_Mini_2`
- `environment` — indoor/outdoor, location name, room type
- `wifi_channel` — e.g., channel 11
- `bandwidth` — HT20 or HT40

**Strongly Recommended:**
- `drone_height_m`, `drone_speed_mps`, `drone_position`
- `human_activity_detail` — e.g., `walking_slow_perpendicular`
- `weather` — temperature (°C), humidity (%), wind speed (m/s)
- `time_of_day`, `interferers_present`, `esp32_rssi`, `esp32_noise_floor`

### 4.2 Labeling Strategies

**Strategy 1 — Pre-scripted annotation (recommended):**
- Write a collection script that logs keypresses with timestamps (e.g., `d` = drone start, `s` = stop, `h` = human)
- After collection, merge label file with CSI data by timestamp
- Add 1–2 second buffer at start/end of each event to exclude transition artifacts

**Strategy 2 — Video-synchronized annotation (highest quality):**
- Record synchronized video during all data collection
- Use a clapper board or LED flash to synchronize video/CSI timestamps
- Used in benchmark datasets like EHUNAM (Diaz et al., 2025)

**Strategy 3 — Google Sheets lightweight annotation:**
- CSI-Bench (arXiv 2505.21866) describes users tapping buttons in Google Sheets to log activities with local timestamps

### 4.3 Label Taxonomy

| Stage | Classes |
|---|---|
| Binary (initial baseline) | `no_target`, `target_present` |
| Primary (3-class) | `empty`, `human`, `drone` |
| Detailed (advanced) | `empty`, `human_walking`, `human_stationary`, `drone_hovering`, `drone_flying`, `drone_and_human` |

### 4.4 Data Quality Screening

Before training, screen each sample for:
- **Packet loss/gaps:** segments with >20% missing packets → discard or interpolate carefully
- **RSSI outliers:** sudden drops may indicate link disruption
- **Phase discontinuities:** large phase jumps (>π) between consecutive packets
- **Transition contamination:** remove first/last N packets of each labeled segment

---

## 5. Machine Learning Approaches

### 5.1 Feature Extraction from CSI Data

**Amplitude:** `|H_k|` for each subcarrier k
- Most robust to phase noise and hardware imperfections
- Most commonly used in the literature

**Phase:** `angle(H_k)` for each subcarrier k
- Requires careful sanitization (see Section 8)
- Phase offset between ESP32 nodes is not calibrated by default

**Derived Time-Domain Features:**
- Mean, variance, standard deviation per subcarrier over a window
- RMS of amplitude, energy, zero-crossing rate, waveform entropy

**Derived Frequency-Domain Features:**
- Power Spectral Density (PSD) — captures periodic drone vibrations
- Spectrogram (STFT) — 2D time-frequency image
- Blade Passing Frequency (BPF) peaks — typically 50–300 Hz for consumer drones

**Dimensionality Reduction:**
- PCA: reduce 52+ subcarriers to 5–20 principal components capturing >95% of variance

### 5.2 Input Representations for Deep Learning

| Representation | Shape | Model |
|---|---|---|
| Raw sequence | (time_steps, n_subcarriers) e.g. (20, 52) | LSTM, 1D-CNN |
| Spectrogram/heatmap | (n_subcarriers, time_steps) — grayscale image | 2D-CNN |
| Dual channel (amp + phase) | (n_subcarriers, time_steps, 2) | 2D-CNN |

### 5.3 Model Comparison

| Model | Accuracy | Notes |
|---|---|---|
| SVM | ~91% | Effective with hand-crafted features; fast inference |
| Random Forest | 85–100% | Robust to noise; 99.93% on UAV traffic detection |
| KNN | ~80–85% | Simple baseline; slow at inference with large datasets |
| 1D-CNN | 90–99% | Processes raw CSI time-series directly |
| LSTM / BiLSTM | High | Captures long-range temporal dependencies |
| 2D-CNN (ResNet, VGG) | ~95–96% | Treats CSI heatmap as image |
| CNN-LSTM | ~94–99% | Hybrid: spatial + temporal features |
| **CNN-LSTM-Attention** | **98.2%** | **Best known result (MDPI Electronics, 2025)** |
| Transformer / ViT | Emerging | SSL competitive with supervised using only 10–20% labels |

### 5.4 Recommended Architecture for Your System

**Phase 1 — Baseline:**
Random Forest or SVM with hand-crafted features (statistical moments, PSD peaks, Doppler energy).
- Input: 52 subcarriers × statistical features per 2-second window
- Expected accuracy: 85–95% for 3-class problem
- Fast to implement and interpret

**Phase 2 — Deep Learning:**
CNN-LSTM or CNN-GRU hybrid on raw CSI amplitude heatmaps.
- Input shape: `(52, 20, 1)` for 2-second windows at 10 Hz
- Expected accuracy: 90–99% depending on data quality

**Phase 3 — Advanced:**
Transformer-based or attention-enhanced BiLSTM for improved generalization across environments and drone models.

### 5.5 Training Pipeline

1. Segment raw CSI into sliding windows (2–5 seconds; 50% overlap)
2. Apply preprocessing pipeline (see Section 8)
3. Extract amplitude matrix per window
4. Apply PCA if using traditional ML, or feed raw heatmap to CNN
5. Train with stratified k-fold cross-validation (k=5 or k=10)
6. Use leave-one-subject-out (LOSO) validation for generalization testing
7. Report: accuracy, precision, recall, F1, confusion matrix
8. Use weighted F1 if class sizes are unequal

---

## 6. ESP32-Specific Considerations and Limitations

### 6.1 Hardware Specifications

**Chip Performance Ranking (Espressif documentation):**
> ESP32-C5 > ESP32-C6 > ESP32-C3 ≈ ESP32-S3 > ESP32 (original)

| Chip | Band | Notes |
|---|---|---|
| ESP32 (WROOM/WROVER) | 2.4 GHz only | Single antenna; HT20/HT40; LLTF: 52 valid subcarriers |
| ESP32-C6 | 2.4 GHz | Improved CSI quality over original |
| ESP32-C5 | 2.4 + 5 GHz | Best option for new deployments |

### 6.2 CSI Data Format (esp-csi get-started)

**Standard ESP32 / C3 / S3 CSV columns:**
```
type, id, mac, rssi, rate, sig_mode, mcs, bandwidth, smoothing,
not_sounding, aggregation, stbc, fec_coding, sgi, noise_floor,
ampdu_cnt, channel, secondary_channel, local_timestamp, ant,
sig_len, rx_state, len, first_word, data
```

**ESP32-C5 / C6 CSV columns:**
```
type, seq, mac, rssi, rate, noise_floor, fft_gain, agc_gain,
channel, local_timestamp, sig_len, rx_state, len, first_word, data
```

The `data` field contains interleaved `[imag, real]` pairs per subcarrier:
```
[sub1_imag, sub1_real, sub2_imag, sub2_real, ...]
```

**Computing amplitude and phase:**
```python
amplitude = sqrt(real_k**2 + imag_k**2)
phase     = atan2(imag_k, real_k)
```

### 6.3 Configuration Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Wi-Fi Channel | 11 (2462 MHz) | Non-overlapping; minimizes interference |
| Bandwidth | HT20 | Safe for all ESP32 variants; 52 valid subcarriers |
| Serial Baudrate | 921,600 bps | Required for CSI throughput at higher packet rates |
| CONFIG_SEND_FREQUENCY | 200 Hz (target) | Default is 20 Hz — increase to 200 for Doppler capability |
| Gain Control | **Forced** | Avoid AGC artifacts that look like motion events |
| MAC Filtering | Transmitter MAC only | Prevents contamination from other WiFi devices |

### 6.4 Known Limitations

1. **No Phase Synchronization** — The two ESP32 nodes do not share a reference oscillator. Raw phase cannot be compared across packets without sanitization.

2. **Single Antenna** — Standard ESP32 boards have one antenna (1×1 MIMO only). Research-grade platforms typically use 3+ antennas.

3. **AGC Artifacts** — Automatic Gain Control introduces amplitude discontinuities that can look like motion events. **Use forced gain control.**

4. **Limited Packet Rate** — At 10 Hz, fast transient events (e.g., fast-flying drone crossing in <0.1s) may be missed. Target 20–25 Hz minimum; 100–200 Hz for Doppler.

5. **Clock Drift** — `local_timestamp` drifts over time. For long sessions (>30 min), synchronize via NTP or external trigger.

6. **Memory Constraints** — Stream raw CSI to host computer via USB serial (921,600 baud). Use TFLite Micro or esp-radar for on-device inference after model development.

7. **2.4 GHz Congestion** — Use channels 1, 6, or 11 (non-overlapping). Consider ESP32-C5 at 5 GHz for cleaner spectrum.

8. **Subcarrier Ordering** — ESP32 CSI subcarrier indices are NOT contiguous in frequency. Exclude null/pilot subcarriers using the Espressif subcarrier index table. See [esp-csi issue #146](https://github.com/espressif/esp-csi/issues/146) and [issue #114](https://github.com/espressif/esp-csi/issues/114).

### 6.5 Optimizations for Your System

- Use channel 11 to minimize interference
- Force HT20 mode for predictable 52-subcarrier output
- Set forced gain control before collection
- Increase baud rate to 921,600 before increasing packet rate above 10 Hz
- Filter incoming CSI by transmitter MAC address at the callback level
- Mount both nodes at identical heights on stable tripods (avoid ESP32 vibration contaminating data)

---

## 7. Dataset Considerations: Size, Balance, and Splits

### 7.1 Sample Sizes from Literature

| Dataset | Size | Split |
|---|---|---|
| WiMANS (arXiv 2402.09430) | Standard | 80/20 train/test |
| CSI-Bench (arXiv 2505.21866) | 461+ hours | 70/15/15 train/val/test |
| EHUNAM (Scientific Data, 2025) | 38 hours, 21 subjects | Multi-environment |
| Wallhack1.8k (Zenodo, 2024) | 1,806 spectrograms | 3 activity classes |
| General (traditional ML) | 500–1,000 windows/class | Minimum |
| General (deep learning) | 2,000–10,000+ windows/class | Recommended |

### 7.2 Recommended Splits

- **Holdout (simplest):** 70% train / 15% validation / 15% test — stratified, entire sessions in one split only
- **Cross-validation:** 5-fold or 10-fold stratified — recommended when dataset < 5,000 total windows
- **LOSO (leave-one-session-out):** Best for evaluating real-world deployment robustness
- **Leave-one-drone-out:** Tests generalization across drone types

### 7.3 Class Balance

If classes are unbalanced:
- Use SMOTE oversampling or undersampling
- Data augmentation for drone class: Gaussian noise, time warping (±10–15%), random subcarrier dropout
- C-DDPM augmentation (arXiv 2404.04829) significantly enhances accuracy for imbalanced CSI datasets
- Report **weighted F1-score**, not just accuracy

### 7.4 Subject Bias and Generalization

Models trained and tested on the same subjects significantly outperform cross-subject generalization. To avoid overoptimistic results:
- Use leave-one-subject-out validation for human activity data
- Collect from at least 8–10 different subjects
- Include diverse demographics (height, weight, gait, age)

---

## 8. Signal Preprocessing Pipeline

Based on ["Optimal preprocessing of WiFi CSI for sensing applications" (arXiv 2307.12126)](https://arxiv.org/abs/2307.12126).

### Step 1 — Packet Filtering
- Accept only CSI from the known transmitter MAC address
- Discard packets with `rx_state != 0`
- Discard packets where `sig_len` is outside expected range
- Log but do not use packets with anomalous `noise_floor` values

### Step 2 — Subcarrier Selection
- Map raw subcarrier indices to valid subcarriers (exclude null/pilot subcarriers)
- For LLTF HT20: use the 52 valid data subcarrier indices
- Optionally: rank subcarriers by variance during a quiet period and select top-k most dynamic

### Step 3 — Gain Correction *(Highest Priority)*
- If AGC is active: average first 100 quiet samples to estimate baseline gain, then subtract
- **Preferred:** configure forced gain control before collection
- Apply low-pass filter (cutoff 0.1 Hz) to capture slow amplifier drift; subtract from signal
- **Reduces noise by up to 40%** (arXiv 2307.12126)

### Step 4 — Phase Sanitization *(If Using Phase)*
- Unwrap phase across subcarriers for each packet
- Apply linear fit to the middle subcarriers
- Subtract the linear fit to remove deterministic hardware offset
- Apply temporal smoothing across consecutive packets

### Step 5 — Outlier Removal
- Apply Hampel filter or IQR filter to each subcarrier's amplitude time series
- Interpolate missing values using linear interpolation if gap < 5 packets; otherwise discard segment

### Step 6 — Temporal Smoothing
- Apply Savitzky-Golay filter or moving average (window 3–5 samples)
- Butterworth low-pass filter (cutoff: Nyquist/2 of your packet rate)

### Step 7 — Windowing and Segmentation
- Segment continuous CSI stream into fixed-length windows
- **Recommended window:** 2–5 seconds (20–50 samples at 10 Hz)
- Sliding window with 50% overlap for training data augmentation

### Step 8 — Normalization
- Per-window z-score normalization (subtract mean, divide by std) for each subcarrier independently
- Alternatively: min-max normalization to [0, 1] range
- Apply the same normalization parameters from training to test/inference

---

## 9. Key Distinctions: Drone vs. Human CSI Signatures

### Drone Signatures
- Propeller blade rotation creates periodic amplitude modulation at the **Blade Passing Frequency (BPF = RPM/60 × num_blades)**
  - DJI Mavic Mini hover: ~5,000–8,000 RPM, 2 blades → BPF = 83–133 Hz
  - ⚠️ **At 10 Hz sampling rate, BPF is NOT directly observable** — only the slow envelope of vibration is captured
  - Increase to 25+ Hz to capture low-frequency drone vibration signatures; 100+ Hz for Doppler
- Hovering drones produce a quasi-stationary but slightly oscillating CSI pattern
- Flying drones produce a transient Doppler shift as they cross the link
- Metallic frame and spinning blades cause RF scattering distinct from human body scattering

### Human Signatures
- Walking produces a periodic Doppler pattern (~1–2 Hz cadence)
- Stride pattern is distinctive: double-peak per step cycle
- Breathing (stationary): ~0.2–0.4 Hz periodic amplitude variation
- Large body surface area causes broader, slower Doppler smearing compared to drone frame

### Key Differentiating Features (at 10 Hz, 15 m TX-RX)

| Feature | Drone | Human |
|---|---|---|
| Amplitude change direction | Often increases (constructive multipath from propellers) | Often decreases (body absorption) |
| Periodicity | Fast, propeller-driven (if rate is sufficient) | Slow, gait-driven (~1–2 Hz) |
| Doppler spread | Narrow (small frame) | Wider (limb swing) |
| Subcarrier correlation | Metallic scatter pattern | Dielectric (water-based) scatter pattern |
| Duration | Variable (hover = sustained; pass = brief) | Variable |

---

## 10. Recommended Research Workflow

### Phase 1 — System Setup and Validation (Week 1–2)
- Flash both ESP32s with esp-csi get-started firmware
- Configure: channel 11, HT20, forced gain control, 10 Hz TX rate, baudrate 921,600
- Verify CSI reception using the esp-csi Python visualization tool
- Collect 5-minute empty baseline and verify no spurious motion artifacts
- Collect 2-minute human walking test and verify perturbations are visible
- Tune distance if needed (start at 5–8 m if 15 m seems insensitive)

### Phase 2 — Data Collection (Week 3–6)
- Follow the experimental protocol in Section 2
- Aim for: 3+ drone types, 5+ human subjects, 3+ environments
- Collect with synchronized video for ground truth verification
- Target: 60+ minutes per class (empty, human, drone)
- Total raw data: ~180+ minutes = ~108,000 CSI packets

### Phase 3 — Preprocessing and EDA (Week 5–7)
- Apply the full preprocessing pipeline (Section 8)
- Segment into 2-second windows with 50% overlap
- Visualize amplitude heatmaps for each class — can you see class differences by eye?
- Check PCA: do the first 2 principal components separate classes?

### Phase 4 — Baseline ML (Week 6–8)
- Train Random Forest and SVM with hand-crafted features
- 5-fold cross-validation; report weighted F1
- Establish baseline accuracy before applying deep learning

### Phase 5 — Deep Learning (Week 7–10)
- Train CNN-LSTM on amplitude heatmap inputs
- Experiment with window sizes (1s, 2s, 5s)
- Try amplitude-only vs. amplitude+phase
- Use leave-one-session-out validation for honest evaluation

### Phase 6 — Generalization Testing (Week 10–12)
- Test on a new drone model not seen in training
- Test with a new human subject not in training
- Test in a new environment
- Analyze confusion matrix: which class pairs are hardest to distinguish?

---

## 11. References

All links verified March 2026.

**[1]** Jian, Y., et al. (2023). "Low-cost UAV detection via WiFi traffic analysis and machine learning."
*Scientific Reports*, 13, 20857.
🔗 [Nature](https://www.nature.com/articles/s41598-023-47453-6) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10684492/)
> Random Forest with 9 trees; 99.93% detection at 280m LOS; 4 statistical packet header features

**[2]** Di Seglio, C., et al. (2024). "Comparing reference-free WiFi radar sensing approaches for monitoring people and drones."
*IET Radar, Sonar & Navigation.*
🔗 [IET Research](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/rsn2.12506)
> Non-coherent passive WiFi radar; detects humans and drones; 2.4 and 5 GHz experiments

**[3]** Diaz, G., et al. (2025). "EHUNAM, a WiFi CSI-based dataset for human and machine sensing."
*Scientific Data (Nature)*, 12.
🔗 [Nature](https://www.nature.com/articles/s41597-025-06238-4) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12711874/)
> 38 hours; 21 subjects; 9 machines; 8 environments; CNN >90% accuracy

**[4]** Niu, J., et al. (2022). "Placement Matters: Understanding the Effects of Device Placement for WiFi Sensing."
*ACM IMWUT*, 6(1).
🔗 [ACM Digital Library](https://dl.acm.org/doi/10.1145/3517237)
> Proper placement expands coverage by ~200%; best placement at midpoints of long walls

**[5]** Alhazbi, S., et al. (2022). "Drone Detection and Classification Using Physical-Layer Protocol Statistical Fingerprint."
*Sensors, MDPI*, 22(17), 6701.
🔗 [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9460464/)
> Cubic SVM; 6 drone types; 5 statistical packet-timing features; 2.4 and 5 GHz

**[6]** Rahman, S., and Robertson, D.A. (2020). "Classification of drones and birds using convolutional neural networks applied to radar micro-Doppler spectrogram images."
*IET Radar, Sonar & Navigation*, 14(6), 733–741.
🔗 [Semantic Scholar](https://www.semanticscholar.org/paper/Classification-of-drones-and-birds)
> GoogLeNet CNN on micro-Doppler spectrograms; distinguishes drones from birds

**[7]** Zhang, J., et al. (2021). "A CSI-Based Human Activity Recognition Using Deep Learning."
*Sensors, MDPI*, 21(21), 7225.
🔗 [MDPI](https://www.mdpi.com/1424-8220/21/21/7225)
> CNN-based HAR; sliding window of 300 samples; amplitude-only input

**[8]** "Wi-Fi Sensing Techniques for Human Activity Recognition: Brief Survey, Potential Challenges, and Research Directions."
*ACM Computing Surveys.*
🔗 [ACM Digital Library](https://dl.acm.org/doi/10.1145/3705893)
> Comprehensive survey; CNN, LSTM, SVM approaches; state of the art through 2024

**[9]** Restuccia, F., et al. (2022). "WiFi Sensing on the Edge."
*IEEE Communications Surveys & Tutorials (COMST).*
🔗 [PDF](https://ebulutvcu.github.io/COMST22_WiFi_Sensing_Survey.pdf)
> On-device ML with TinyML; energy consumption; real-world deployment challenges

**[10]** ⚠️ *Preprint* — "Optimal preprocessing of WiFi CSI for sensing applications." (2023).
*arXiv:2307.12126v2.*
🔗 [arXiv](https://arxiv.org/abs/2307.12126)
> Gain correction reduces noise by 40%; phase correction improves SNR by 20%

**[11]** "Motion Pattern Recognition via CNN-LSTM-Attention Model Using Array-Based Wi-Fi CSI Sensors." (2025).
*Electronics, MDPI*, 14(8), 1594.
🔗 [MDPI](https://www.mdpi.com/2079-9292/14/8/1594)
> CNN-LSTM-Attention achieves 98.2% accuracy; superior to single-receiver models

**[12]** Hernandez, S.M. (2020). "ESP32-CSI-Tool."
🔗 [GitHub](https://github.com/StevenMHernandez/ESP32-CSI-Tool) | [Website](https://stevenmhernandez.github.io/ESP32-CSI-Tool/)
> Active and passive modes; MicroSD logging; USB serial streaming; AP/Station/Passive modes

**[13]** "Wi-ESP: A Tool for CSI-based Device-Free Wi-Fi Sensing." (2020).
*Journal of Computational Design and Engineering*, 7(5), 644–656.
🔗 [Oxford Academic](https://academic.oup.com/jcde/article/7/5/644/5837600)
> ESP32 CSI capabilities; IQ data format; practical deployment guidance

**[14]** Espressif Systems. (2024). "ESP-CSI: Applications based on Wi-Fi CSI."
🔗 [GitHub](https://github.com/espressif/esp-csi) | [Get-Started Example](https://github.com/espressif/esp-csi/tree/master/examples/get-started)
> Official library; chip performance ranking: C5 > C6 > C3 ≈ S3 > ESP32

**[15]** Espressif Systems. (2024). "Wi-Fi Driver — ESP32." ESP-IDF Programming Guide v5.5.3.
🔗 [Espressif Docs](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi.html)
> API: `esp_wifi_set_csi_rx_cb()`, `esp_wifi_set_csi_config()`; LLTF/HT-LTF/STBC-HT-LTF training fields

**[16]** ⚠️ *Preprint* — "WiMANS: A Benchmark Dataset for WiFi-based Multi-user Activity Sensing." (2024).
*arXiv:2402.09430.*
🔗 [arXiv](https://arxiv.org/abs/2402.09430)
> 80/20 train/test split; 1,000 pps standard; multi-user scenarios

**[17]** ⚠️ *Preprint* — "CSI-Bench: A Large-Scale In-the-Wild Dataset for Multi-task WiFi Sensing." (2025).
*arXiv:2505.21866.*
🔗 [arXiv](https://arxiv.org/abs/2505.21866)
> 461+ hours; 70/15/15 train/val/test split; natural uncontrolled conditions

**[18]** Wu, D., Zeng, Y., Zhang, F., et al. (2022). "WiFi CSI-based device-free sensing: from Fresnel zone model to CSI-ratio model."
*CCF Transactions on Pervasive Computing and Interaction*, vol. 4, pp. 88–102.
🔗 [Springer](https://link.springer.com/article/10.1007/s42486-021-00077-z) | [DOI](https://doi.org/10.1007/s42486-021-00077-z)
> Fresnel zone governs sensing range; 0.5–4 m TX-RX gives 4m × 3m sensing ellipse

**[19]** "A Survey on Detection, Classification, and Tracking of UAVs using Radar and Communications Systems." (2024).
*arXiv:2402.05909.*
🔗 [arXiv](https://arxiv.org/abs/2402.05909)
> Comprehensive UAV detection survey; 280m detection range for WiFi-based systems

**[20]** "CSI Sanitization Tutorial." Wireless Sensing Tutorial (WST), Tsinghua University.
🔗 [Tsinghua WST](https://tns.thss.tsinghua.edu.cn/wst/docs/sanitization/)
> Phase unwrapping; linear detrending for hardware offset removal

**[21]** "Machine Learning Algorithms Applied for Drone Detection and Classification." (2024).
*Frontiers in Communications and Networks.*
🔗 [Frontiers](https://www.frontiersin.org/journals/communications-and-networks/articles/10.3389/frcmn.2024.1440727/full)
> CNN, LSTM, CNN-LSTM, KNN, SVM comparison; MFNet and MFNet-FA models

**[22]** "Deep Dive Into ESP-CSI: Channel State Information on ESP32 Chips."
*DEV Community, Pratha Maniar.*
🔗 [DEV Community](https://dev.to/pratha_maniar/a-deep-dive-into-esp-csi-channel-state-information-on-esp32-chips-5el1)
> ESP32 CSI overview; OFDM subcarrier explanation; amplitude vs. phase reliability

**[23]** "CSI Data Acquisition and Processing." DeepWiki / espressif/esp-csi.
🔗 [DeepWiki](https://deepwiki.com/espressif/esp-csi/2.1-csi-data-acquisition-and-processing)
> Default 20 Hz TX frequency; CSV output fields; gain calibration; known issues

**[24]** "RF/WiFi-based UAV surveillance systems: A systematic literature review." (2024).
*ScienceDirect.*
🔗 [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2542660524001422)
> Systematic review; 2.4 GHz and 5.8 GHz ISM band focus; detection range; system architectures

**[25]** "Radar micro-Doppler signatures of drones and birds at K-band and W-band." (2018).
*Scientific Reports.*
🔗 [Nature](https://www.nature.com/articles/s41598-018-35880-9)
> BPF for commercial drones ~100–300 Hz; distinct signatures for drones vs. birds

**[26]** "Evaluating Self-Supervised Learning for WiFi CSI-Based Human Activity Recognition." (2025).
*ACM Transactions on Sensor Networks.*
🔗 [ACM Digital Library](https://dl.acm.org/doi/10.1145/3715130)
> SSL competitive with supervised learning using only 10–20% labeled data

---

*Report compiled: March 2026*
*System: ESP32 dual-node WiFi CSI | 50 ft outdoor separation | ~10 pkts/sec*
*Base firmware: [espressif/esp-csi get-started](https://github.com/espressif/esp-csi/tree/master/examples/get-started)*
