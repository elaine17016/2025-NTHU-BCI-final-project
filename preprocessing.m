function preprocessing()
% preprocessing.m
% Preprocess EEG data for motor imagery classification (e.g., from BCI Competition IV-2a)
% This script selects channels, extracts time window, applies sliding window segmentation,
% and saves the data in [samples × channels × time] format for training.

%% 0. (Optional) Confirm the numeric event field name
numericField = 'eventedftype';  % Usually 'eventtype' or 'eventedftype' for BCI IV-2a

%% 1. Select specific EEG channels (C3, Cz, C4)
EEG2 = pop_select(EEG, 'channel', {'C3','Cz','C4'});

%% 2. Extract data from 1 to 4 seconds
fs   = EEG2.srate;
idx1 = round((1 - EEG2.xmin) * fs);
idx2 = round((4 - EEG2.xmin) * fs);
data = EEG2.data(:, idx1:idx2, :);   % [channels × time × trials]

%% 3. Segment the data using a sliding window
win_sec  = 2.0;
step_sec = 0.25;
win_len  = round(win_sec  * fs);
step     = round(step_sec * fs);

[nChan, nTime, nTrials] = size(data);
seg_starts = 1 : step : (nTime - win_len + 1);
nSegs      = numel(seg_starts);

% Preallocate memory
segments = zeros(nTrials * nSegs, nChan, win_len);
labels   = zeros(nTrials * nSegs, 1);

%% 4. Extract each segment and assign labels
for tr = 1:nTrials
    cue = EEG2.epoch(tr).(numericField);  % e.g., 769=left, 770=right
    for si = 1:nSegs
        seg_idx = (tr - 1) * nSegs + si;
        start_i = seg_starts(si);
        segments(seg_idx, :, :) = data(:, start_i:start_i + win_len - 1, tr);
        labels(seg_idx) = cue;
    end
end

%% 5. Map labels: 769 → 0, 770 → 1
X = segments;
y = labels;
y(y == 769) = 0;
y(y == 770) = 1;

%% 6. Save result
save('MI_3.mat', 'X', 'y', '-v7.3');

% Optional: print summary
fprintf('Saved preprocessed data:\n');
fprintf('X size: [%d × %d × %d]\n', size(X));
fprintf('y distribution:\n');
tabulate(y)
end
