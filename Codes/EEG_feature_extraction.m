%% EEG feature extracting
clc
clear all

%% loading dataset
path = pwd;
cd '../Dataset'
load('DREAMER.mat');

cd '../Codes'

%% data managing /EEG_stimuli and baseline

number_of_participants = DREAMER.noOfSubjects;
number_of_trials = DREAMER.noOfVideoSequences;

eeg_baseline = cell(number_of_participants, number_of_trials);
eeg_stimuli = cell(number_of_participants, number_of_trials);

for p=1:number_of_participants
    for t=1:number_of_trials
        eeg_baseline{p,t} = DREAMER.Data{1,p}.EEG.baseline{t,1};
        eeg_stimuli{p,t} = DREAMER.Data{1,p}.EEG.stimuli{t,1};
    end     
end

%% extracting 60s of stimuli and 4s of baseline

for p=1:number_of_participants
    for t=1:number_of_trials
        % baseline
        baseline = cell2mat(eeg_baseline(p,t));
        baseline(1:(end-512),:) = [];
        eeg_baseline{p,t} = baseline;

        %stimuli
        stimuli = cell2mat(eeg_stimuli(p,t));
        stimuli(1:(end-7680),:) = [];
        eeg_stimuli{p,t} = stimuli;
    end     
end


%% feature matrix of EEG and baseline (3 frequency bands)

waveletName = 'db5';
level = 5;
    
number_of_channels = length(DREAMER.EEG_Electrodes);

for t=1:number_of_trials
    for p=1:number_of_participants
        for numofchannel=1:number_of_channels
            stimuli = cell2mat(eeg_stimuli(p,t));
            [c0,l0] = wavedec(stimuli(:,numofchannel),level,waveletName);
            D5(:,1) = wrcoef('d',c0,l0,waveletName,2); %GAMMA
            D5(:,2) = wrcoef('d',c0,l0,waveletName,3); %BETA
            D5(:,3) = wrcoef('d',c0,l0,waveletName,4); %ALPHA
            D5(:,4) = wrcoef('d',c0,l0,waveletName,5); %THETA
            D5(:,5) = wrcoef('a',c0,l0,waveletName,5); %DELTA

            for i=2:4
                feature_eeg(p+(t-1)*number_of_participants,(numofchannel-1)*3+(i-1)) = bandpower(D5(:,i));
            end
        end
    end
end

for t=1:number_of_trials
    for p=1:number_of_participants
        for numofchannel=1:number_of_channels
            baseline = cell2mat(eeg_baseline(p,t));
            [c0,l0] = wavedec(baseline(:,numofchannel),level,waveletName);
            d5(:,1) = wrcoef('d',c0,l0,waveletName,2); %GAMMA
            d5(:,2) = wrcoef('d',c0,l0,waveletName,3); %BETA
            d5(:,3) = wrcoef('d',c0,l0,waveletName,4); %ALPHA
            d5(:,4) = wrcoef('d',c0,l0,waveletName,5); %THETA
            d5(:,5) = wrcoef('a',c0,l0,waveletName,5); %DELTA

            for i=2:4
                feature_baseline(p+(t-1)*number_of_participants,(numofchannel-1)*3+(i-1)) = bandpower(d5(:,i));
            end
        end
    end
end

%% normalizing the feature matrix %dividing to baseline

norm_feature_eeg = (feature_eeg - feature_baseline)./feature_baseline;
save('feature_eeg.mat', 'norm_feature_eeg');
