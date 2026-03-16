# WiFi CSI Drone Detection — Research Summary
**March 2026**

---

## Overview

This document summarizes the key findings from a Claude Agent research session on WiFi CSI-based drone detection, filtered specifically for relevance to the current experimental setup: two ESP32 devices separated by 50 ft (~15.2 m) outdoors, collecting at ~10 packets/second using the Espressif esp-csi library.

---

## 1. The Single Most Important Finding

> **10 packets/second is insufficient for drone-specific detection.**

The literature classifies 10 pps as "passive/low-rate sniffing" — the lowest useful tier. The minimum required rates are:

| Use Case | Minimum Rate | Notes |
|---|---|---|
| Basic presence detection | 100–200 pps | Captures propeller amplitude modulation envelope |
| Full Doppler reconstruction | ≥ 2,000 pps | Required to resolve full micro-Doppler signature |
| De facto research standard | 1,000 pps | Intel 5300 / Atheros AR9300 literature |
| ESP32 practical ceiling | ~100–300 pps | Achievable by changing `send_frequency` in firmware |
| Current system | ~10 pps | ❌ Too low for Doppler — only amplitude/activity usable |

**Fix:** Change `send_frequency` from 20 to 200 in `esp-csi/examples/get-started/csi_send/main/app_main.c` and reflash the transmitter ESP32.

---

## 2. Why Our Results Still Make Sense

### 2.1 Why the Drone Caused an Amplitude Increase

At 50 ft (15.2 m) separation, the first Fresnel zone radius at the midpoint is approximately **68 cm**. A consumer drone (DJI Mavic class) has a body of ~20–30 cm, which is comparable to — but smaller than — the Fresnel zone. This means:

- The drone does **not fully block** the signal path
- Instead, spinning propellers and motor coils **reflect signal back** toward the receiver
- This creates **constructive multipath** → amplitude rises, not drops
- This is physically correct and consistent with published literature

### 2.2 Why the Person Had Almost No Effect

Walking at 50 ft separation with only 10 pps means:
- The Fresnel zone at 50 ft is wide (~68 cm radius) — a person walking near the midpoint causes a smaller relative perturbation
- At 10 pps we cannot capture fast variations — only slow trends are visible
- The RSSI difference between baseline and walking was only **−0.4 dBm**, which is within normal variation

**Both of these are valid scientific observations to include in the paper.**

---

## 3. Fresnel Zone — Key Numbers for Our Setup

| TX–RX Distance | FFZ Radius at Midpoint | Notes |
|---|---|---|
| 12 ft (3.7 m) indoor | ~34 cm | Drone body partially occludes FFZ |
| 50 ft (15.2 m) outdoor | ~68 cm | Drone fits inside FFZ |
| 264 ft (80.5 m) range test | ~158 cm | Very wide — drone is small relative to FFZ |

**Formula:** `r1 = sqrt(lambda × d1 × d2 / (d1 + d2))` where lambda = 0.125 m at 2.4 GHz

Key insight from literature: *"The Fresnel zone radius (~40 cm at 2.4 GHz, 5 m link) being comparable to a drone's physical size is the key geometric argument for why WiFi sensing is viable for drone detection at all."*

---

## 4. Path Loss — Validating Our n = 2.59

Our fitted path loss exponent from the 264 ft range test was **n = 2.59**.

| Environment | Expected n | Our Result |
|---|---|---|
| Free space | 2.0 | — |
| Outdoor LOS | 2.0–2.5 | ✅ Close |
| Outdoor with minor obstruction | 2.5–3.0 | ✅ Our value fits here |

**Interpretation:** n = 2.59 is consistent with outdoor line-of-sight with minor environmental effects (cold air, slight obstructions, snow ground reflection). This is a realistic and defensible result.

**Dual-slope model note:** Above the breakpoint distance `d_BP = 4 × h_TX × h_RX / lambda` (~32 m for 1 m antenna height), n rises to 3.5–4.0. Our 264 ft (~80 m) test exceeded this breakpoint, which partly explains our slightly elevated n.

---

## 5. Drone Propeller Doppler Signature

For a DJI Mavic-class drone:

| Parameter | Value |
|---|---|
| Propeller RPM | ~5,000–7,000 RPM |
| Blade passing frequency (BPF) | ~83–233 Hz (RPM × blades / 60) |
| Max micro-Doppler shift at 2.4 GHz | ~800–1,520 Hz |
| Nyquist-required sampling rate | ≥ 2,000 pps |
| Detectable at 100–200 pps? | ✅ Yes — amplitude envelope only, not full spectrum |

**At our current 10 pps:** Doppler analysis is completely blind. The propeller spins at 100+ Hz but we only sample 10 times per second — we cannot see it at all.

---

## 6. Recommended Next Experiments (Priority Order)

### Immediate (can do now at 10 pps)

1. **Clean 10-min baseline** — record empty environment, no people, no drone, outdoors. This is the most critical missing piece.
2. **Drone hovering stationary** — hover directly over the midpoint at 1 m, 2 m, 3 m height. Hovering is more consistent than flying for detection.
3. **Person walking perpendicular** — walk straight across the signal path at the midpoint (not parallel). This maximizes the Doppler effect even at low packet rates.

### After Fixing Packet Rate (100–200 pps)

4. **Repeat all above tests** — same experiments but now with Doppler analysis possible
5. **Drone at different distances from link** — 1 m, 3 m, 5 m, 10 m laterally from midpoint
6. **Drone takeoff and landing** — these transition events have distinctive signatures
7. **Multiple people vs. one drone** — tests false-positive rejection

### Data Labeling Protocol

Keep a phone stopwatch running during every recording and note:
- `T+00:00` — recording starts
- `T+00:25` — event starts (walking / drone on)
- `T+01:45` — event ends
- Any unusual events (wind gust, someone passing nearby)

---

## 7. Machine Learning — Future Path

**Not needed yet** — we need more labeled data first. Target: 20–30 recordings per class minimum.

When ready, recommended progression:

| Stage | Method | Expected Accuracy |
|---|---|---|
| 1 — Start simple | Random Forest / SVM on mean amp, std, activity score | Baseline |
| 2 — Intermediate | CNN on CSI amplitude heatmap | ~85–90% |
| 3 — Best known | CNN-LSTM-Attention on time series | ~98% (literature) |

**Key distinction:** Most published papers use Intel 5300 / Atheros hardware at 1,000 pps. Our ESP32 at 10 pps cannot replicate their results directly. This is an important limitation to state clearly in the paper.

---

## 8. How Our Work Compares to Literature

| Aspect | Published Literature | Our System |
|---|---|---|
| Hardware | Intel 5300, Atheros AR9300 | ESP32 (lower cost) |
| Packet rate | 100–1,000 pps | 10 pps (fixable to ~200) |
| Subcarriers | 30–52 | 192 (better!) |
| Drone detection method | Doppler + ML | Amplitude + activity (currently) |
| Cost | $100–$500 | ~$20 total |
| Accessibility | Research labs | Anyone |

**Our advantage:** 192 subcarriers vs. 30–52 in most literature — this is richer per-packet data. Our disadvantage is packet rate, which is fixable.

---

## 9. Key Citations for the Paper

1. Bisio et al., "Blind Detection: Advanced Techniques for WiFi-Based Drone Surveillance," **IEEE TVT**, vol. 68, pp. 938–946, 2019.
2. "Low-cost UAV detection via WiFi traffic analysis and machine learning," **Scientific Reports**, 2023. *(99.93% detection when drone streams video)*
3. Wu et al., "WiFi CSI-based device-free sensing: from Fresnel zone model to CSI-ratio model," **CCF TPCI**, Springer, 2021.
4. "Towards a Dynamic Fresnel Zone Model to WiFi-based Human Activity Recognition," **ACM IMWUT**, 2023.
5. NOT PEER REVIEWED - WiMANS benchmark dataset, arXiv:2402.09430, 2024. *(1,000 pps standard)*
6. NOT PEER REVIEWED "Optimal preprocessing of WiFi CSI for sensing applications," arXiv:2307.12126, 2023. *(gain correction reduces noise by 40%)*
7. Espressif Systems, "ESP-CSI: Applications based on Wi-Fi CSI," GitHub, 2024.

---

## 10. Summary 

> *"We built a low-cost WiFi CSI drone detection system using two ESP32 microcontrollers following the Espressif esp-csi example. Initial experiments at 10 packets/second over 50 ft outdoor separation revealed that while amplitude-based features show a measurable difference between drone presence (+0.75 amplitude units) and human walking (−0.40 amplitude units), the differences are too subtle for reliable automated detection at this packet rate. Literature confirms that 100–200 pps is the minimum for envelope-level drone detection, and ≥2,000 pps for full micro-Doppler reconstruction. The next phase involves increasing the packet rate to 200 pps via firmware modification and collecting properly labeled datasets for each class: empty environment, human walking, and drone hovering."*

---





# Key Citations — WiFi CSI Drone Detection Project

---

## Peer-Reviewed Publications

**[1]** I. Bisio, C. Garibotto, F. Lavagetto, A. Sciarrone, and S. Zappatore,
"Blind Detection: Advanced Techniques for WiFi-Based Drone Surveillance,"
*IEEE Transactions on Vehicular Technology*, vol. 68, no. 1, pp. 938–946, Jan. 2019.
🔗 [IEEE Xplore](https://ieeexplore.ieee.org/document/8556480) | [DOI: 10.1109/TVT.2018.2884767](https://doi.org/10.1109/TVT.2018.2884767)

---

**[2]** Anonymous et al.,
"Low-cost UAV detection via WiFi traffic analysis and machine learning,"
*Scientific Reports (Nature Portfolio)*, 2023.
*(Key finding: 99.93% detection probability when UAV streams video)*
🔗 [Scientific Reports](https://www.nature.com/articles/s41598-023-47453-6)

---

**[3]** D. Wu, Y. Zeng, F. Zhang et al.,
"WiFi CSI-based device-free sensing: from Fresnel zone model to CSI-ratio model,"
*CCF Transactions on Pervasive Computing and Interaction*, vol. 4, pp. 88–102, 2022.
🔗 [Springer](https://link.springer.com/article/10.1007/s42486-021-00077-z) | [DOI: 10.1007/s42486-021-00077-z](https://doi.org/10.1007/s42486-021-00077-z)

---

**[4]** J. Liu et al.,
"Towards a Dynamic Fresnel Zone Model to WiFi-based Human Activity Recognition,"
*Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, vol. 7, no. 2, Article 65, Jun. 2023.
🔗 [ACM Digital Library](https://dl.acm.org/doi/10.1145/3596258)

---

## Technical References *(Not Peer-Reviewed — Use as Supporting References Only)*

**[5]** ⚠️ *Preprint — not peer-reviewed*
X. Chen et al.,
"WiMANS: A Benchmark Dataset for WiFi-based Multi-user Activity Sensing,"
*arXiv preprint*, arXiv:2402.09430, 2024.
*(Key finding: 1,000 pps standard packet rate for CSI sensing research)*
🔗 [arXiv:2402.09430](https://arxiv.org/abs/2402.09430)

---

**[6]** ⚠️ *Preprint — not peer-reviewed*
Anonymous et al.,
"Optimal preprocessing of WiFi CSI for sensing applications,"
*arXiv preprint*, arXiv:2307.12126, 2023.
*(Key finding: gain correction reduces noise by 40%; phase correction improves SNR by 20%)*
🔗 [arXiv:2307.12126](https://arxiv.org/abs/2307.12126)

---

## Official Documentation

**[7]** Espressif Systems,
"ESP-CSI: Applications based on Wi-Fi CSI (Channel State Information),"
*GitHub Repository*, 2024.
🔗 [GitHub Repository](https://github.com/espressif/esp-csi) | [Get-Started Example](https://github.com/espressif/esp-csi/tree/master/examples/get-started)

---

## Notes on Citation Usage

| # | Venue | Peer-Reviewed | Safe for IEEE Paper |
|---|---|---|---|
| [1] | IEEE Transactions on Vehicular Technology | ✅ Yes | ✅ Yes |
| [2] | Scientific Reports (Nature) | ✅ Yes | ✅ Yes |
| [3] | CCF TPCI (Springer) | ✅ Yes | ✅ Yes |
| [4] | ACM IMWUT | ✅ Yes | ✅ Yes |
| [5] | arXiv preprint | ❌ No | ⚠️ Supporting only |
| [6] | arXiv preprint | ❌ No | ⚠️ Supporting only |
| [7] | GitHub (official docs) | ❌ No | ✅ Hardware reference |

---
*Report compiled from Claude Agent research session | March 2026*
*Base system: ESP32 dual-node, 50 ft outdoor, ~10 pps, esp-csi GitHub firmware*
