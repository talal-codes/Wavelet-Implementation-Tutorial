%% CWT for Time-Frequency Analysis of signals
clear ; close all; clc;

% Define signal parameters
frq1 = 32;
amp1 = 1;
frq2 = 64;
amp2 = 2;

Fs = 1e3;
t = 0:1/Fs:1;

% Create the signal with time-varying frequencies
x = amp1*sin(2*pi*frq1*t).*(t>=0.1 & t<0.3)+...
    amp2*sin(2*pi*frq2*t).*(t>0.6 & t<0.9);

% Compute the CWT using the Morlet wavelet
[wt, f] = cwt(x,Fs);

% Magnitude of the CWT
mag_wt = abs(wt);

% Plot the original signal
subplot(2,1,1)
plot(t,x)
xlabel('Time (s)')
ylabel('Signal')
title('Signal with Time-Varying Frequencies')

% Plot the magnitude of CWT in log scale
subplot(2,1,2)
pcolor(t,f,mag_wt)
shading('flat')
xlabel('Time (s)')
ylabel('Frequency (Hz)')
title('Magnitude of CWT (log scale)')
set(gca,'yscale','log')

% Customize the plot for better visualization
colormap('hot');
colorbar;

% Display the plots
sgtitle('1D CWT Analysis of Time-Varying Signal')



%% 1-D  single-level DWT
clear ; close all; clc;

% Define sampling frequency (fs)
fs = 1000;

% Define time vector
t = 0:1/fs:1;


% Define signal 1 (250-500 Hz)
f1 = 300; % Center frequency
% f1_low = f1 - 125; % Lower frequency bound
f1_high = f1 + 125; % Upper frequency bound
signal1 = sin(2*pi*f1*t) + 0.25*sin(2*pi*f1_high*t);

% Define signal 2 (0-250 Hz)
f2 = 100;
signal2 = sin(2*pi*f2*t) + 0.75*sin(2*pi*(f2/2)*t);

% Mix signals
mixed_signal = signal1 + signal2;



% Visualize original signals
figure
subplot(3,1,1);
plot(t,signal1);
title('Signal 1 (250-500 Hz)');
subplot(3,1,2);
plot(t,signal2);
title('Signal 2 (0-250 Hz)');
subplot(3,1,3);
plot(t,mixed_signal);
title('Mixed Signal');



% Choose wavelet (Here, 'db4' is used for demonstration)
wavelet_name = 'db4';

% Perform single-level DWT
[cA, cD] = dwt(mixed_signal, wavelet_name);

% Approximation coefficients (represents low-frequency content)
approx_signal = idwt(cA, [], wavelet_name);

% Detail coefficients (represents high-frequency content)
detail_signal = idwt([], cD, wavelet_name);


% Visualize separated signals
figure
subplot(2,1,1);
plot(t,approx_signal(1:end-1));
title('Approximation (Signal 2)');
subplot(2,1,2);
plot(t,detail_signal(1:end-1));
title('Detail (Signal 1)');



%% 1-D  multi-level DWT
clear ; close all; clc;


% Define sampling frequency (fs)
fs = 1000;

% Define time vector
t = 0:1/fs:0.25;

% Define multiple sine waves with different frequencies and amplitudes
f1 = 30; % Low frequency
f2 = 200; % Mid frequency
f3 = 300; % High frequency
signal1 = 2*sin(2*pi*f1*t);  % Double amplitude for better visualization
signal2 = sin(2*pi*f2*t);
signal3 = 0.5*sin(2*pi*f3*t);  % Half amplitude for better visualization

% Mix signals
mixed_signal = signal1 + signal2 + signal3;


% Visualize original signals
figure
subplot(4,1,3);
plot(t,signal1);
title('Signal 1 (30 Hz)');
subplot(4,1,2);
plot(t,signal2);
title('Signal 2 (200 Hz)');
subplot(4,1,1);
plot(t,signal3);
title('Signal 3 (300 Hz)');
subplot(4,1,4);
plot(t,mixed_signal);
title('Mixed Signal');



% Choose wavelet (Here, 'db4' is used for demonstration)
wavelet_name = 'db4';
n=3; % Number of levels
% n = wmaxlev(length(mixed_signal),wavelet_name);

% Perform 3-level DWT decomposition
[cA, cD] = wavedec(mixed_signal, n, wavelet_name);


% Access coefficients at different levels
approx_coeff = appcoef(cA, cD, wavelet_name);  % Approximation (Level 3)
detail_coeff = detcoef(cA, cD, (1:3));         % Detail (Level 1-3)

% Combine all coefficients into one cell array
Coeff_all = detail_coeff;
Coeff_all{1,n+1} = approx_coeff;          % All Coefficients Combined [cD1-cDn, cAn]


% Visualize separated signals
figure
for i = 1:n+1
    subplot(n+1,1,i);
    plot(Coeff_all{1,i});
    title(['Level ' num2str(i)]);
end




% Reconstruct each single detailed and approx coefficient
reconstructed_signals = cell(1, n+1);
for i = 1:n+1
    if i==n+1
        reconstructed_signals{i} = wrcoef('a',cA,cD,wavelet_name);
    else
        reconstructed_signals{i} = wrcoef('d', cA,cD,wavelet_name, i);
    end
end



% Plot the reconstructed signals
figure
for i = 1:n+1
    subplot(n+1,1,i);
    plot(t, reconstructed_signals{i});
    title(['Reconstructed Level ' num2str(i)]);
end






%% Feature extraction using wavelet transform
clear ; close all; clc;


load("EEGData.mat");

EEG_segment = eeg_data(1:15*Fs);


% Specify the wavelet type
wavelet_type = 'db4';
% Calculate the maximum level of decomposition
max_level = wmaxlev(length(EEG_segment),wavelet_type);

% Decompose the filtered eeg data into n = max levels
[c,l] = wavedec(EEG_segment,max_level,wavelet_type);

% Feature extraction, Extracting wavelet energy, wavelet entropy, wavelet standard deviation,
% wavelet variance and wavelet mean from each level of decomposition

% Calculate wavelet energy from each level of decomposition
[wavelet_energy_a, wavelet_energy_d] = wenergy(c,l);


% Calculate wavelet entropy from each level of decomposition
X = detcoef(c,l,"cells");
X{end+1} = appcoef(c,l,wavelet_type);
wavelet_entropy = (wentropy(X))';

% Calculate wavelet std from each level of decomposition
wavelet_std = zeros(1,max_level);
for j = 1:max_level+1
    wavelet_std(j) = std(X{j});
end


% Calculate wavelet skewness from each level of decomposition
wavelet_skewness = zeros(1,max_level);
for j = 1:max_level+1
    wavelet_skewness(j) = skewness(X{j});
end

% Calculate wavelet kurtosis from each level of decomposition
wavelet_kurtosis = zeros(1,max_level);
for j = 1:max_level+1
    wavelet_kurtosis(j) = kurtosis(X{j});
end

% Calculate wavelet Shape Factor from each level of decomposition
wavelet_SF = zeros(1,max_level);
for j = 1:max_level+1
    wavelet_SF(j) =  rms((X{j}))/mean(abs((X{j})));
end

% Concatenate all the features from each level of decomposition into a single feature vector
wavelet_features = [wavelet_energy_a wavelet_energy_d wavelet_std wavelet_entropy wavelet_skewness wavelet_kurtosis wavelet_SF];



%%

wname = 'db4';
lev = 5;

% [c,l] = wavedec(current_cropped(:,1),lev,wname);
% 
% sum(c.^2)

[Energy,Std] = ComputeFeatures(current,wname,lev)


% std(current_cropped(:,1))

