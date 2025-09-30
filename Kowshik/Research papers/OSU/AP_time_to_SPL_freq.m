clear; clc; close all;

% Load your two-column ASCII file (time, raw signal)
data = readmatrix('22Sep_16in_both_0deg_89.txt');  
time = data(:,1);        % time in seconds
pressure_Pa = data(:,2);  % raw mic signal (Volts)

% ---- Microphone sensitivity ----
%S_V_per_Pa = 46.5e-3;   % 46.5 mV/Pa = 0.0465 V/Pa

% ---- Convert to Pascals ----
%pressure_Pa = signal_raw ./ S_V_per_Pa;

% ---- Sampling parameters ----
dt = mean(diff(time));      % time step
Fs = 1/dt;                  % sampling frequency
N  = length(time);          % total samples

% ---- Select time window ----
t_start = 11.1;   % [s]  (example: start time of window)
t_end   = 11.25;   % [s]  (example: end time of window)

idx = (time >= t_start & time <= t_end);  
time_win = time(idx);
pressure_win = pressure_Pa(idx);

% ---- FFT ----
Nfft = length(pressure_win);
Y = fft(pressure_win, Nfft);
P2 = abs(Y/Nfft);          % two-sided spectrum
P1 = P2(1:floor(Nfft/2)+1); 
P1(2:end-1) = 2*P1(2:end-1);  % single-sided spectrum

f = Fs*(0:floor(Nfft/2))/Nfft;  % frequency vector

% ---- SPL calculation ----
p_ref = 20e-6;   % reference pressure (20 µPa)
SPL = 20*log10(P1 / p_ref);

% ---- Plot time window ----
figure('Color','w');
plot(time_win, pressure_win, 'k','LineWidth',1.5);
grid on;
xlabel('Time (s)');
ylabel('Amplitude (Pa)');
title('Selected Time Window');

% ---- Plot SPL spectrum ----
figure('Color','w');
semilogx(f, SPL, 'b','LineWidth',1.5); grid on;
xlabel('Frequency (Hz)');
ylabel('SPL (dB re 20 \muPa)');
title('SPL vs Frequency');
xlim([100 13000]);   % only meaningful range
