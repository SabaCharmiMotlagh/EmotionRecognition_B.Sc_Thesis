%% ECG feature extracting
clc
clear all

%% loading dataset
path = pwd;
cd '../Dataset'
load('DREAMER.mat');

cd '../Codes'

%% data managing /ECG stimuli

number_of_participants = DREAMER.noOfSubjects;
number_of_trials = DREAMER.noOfVideoSequences;

ecg_stimuli = cell(number_of_participants, number_of_trials);

for p=1:number_of_participants
    for t=1:number_of_trials
        ecg_stimuli{p,t} = DREAMER.Data{1,p}.ECG.stimuli{t,1};
    end     
end


%% Extracting Valence and Arousal Labels

count = 0;

for t=1:number_of_trials
    for p=1:number_of_participants
        count = count + 1;
        LabelValence(1,count) = DREAMER.Data{1,p}.ScoreValence(t);
        LabelArousal(1,count) = DREAMER.Data{1,p}.ScoreArousal(t);
    end     
end


%% Notch PowerLine Noise

% filter parameters
Fs = 256; 
Fo = 50; 
Fon = Fo/Fs;
f1 = 45; 
f2 = 55; 
Bz = fir1(2000,[f1/Fs f2/Fs ],'stop'); 

for p=1:number_of_participants
    for t=1:number_of_trials

        stimuli = cell2mat(ecg_stimuli(p,t));

        % applying filter to ECG channels
        for channel=1:2
            ecg_fstimuli{p,t}(:,channel) = abs(filter(Bz,1,stimuli(:,channel))).^2;
        end
        
    end
end

%% HRV (Heart Rate Variability) Extraction

HRV = [];

for p=1:number_of_participants
    for t=1:number_of_trials

        for channel=1:2
            data = ecg_fstimuli{p,t};

            [~,locs] = findpeaks(data(:,channel));
            for k=2:size(locs,1)
                HRV.HRV{p,t}{channel}(k,1) = data(locs(k),channel) - data(locs(k-1),channel);
            end

            m = 0;
            for k = 1:size(locs,1)-1
                if((locs(k+1)/199)-(locs(k)/199)<0.05)
                    m = m + 1;
                end
            end
            
            HRV.HRVpNN50{p,t}(channel) = (m./size(locs,1)).*100;
            
        end
    end
end


%% Features for HRV

for p=1:number_of_participants
    for t=1:number_of_trials
        for channel=1:2

            HRV.mean{p,t}(channel) = mean(HRV.HRV{p,t}{channel});
            HRV.median{p,t}(channel) = median(HRV.HRV{p,t}{channel});
            HRV.std{p,t}(channel) = std(HRV.HRV{p,t}{channel});
            HRV.min{p,t}(channel) = min(HRV.HRV{p,t}{channel});
            HRV.max{p,t}(channel) = max(HRV.HRV{p,t}{channel});
            HRV.range{p,t}(channel) = range(HRV.HRV{p,t}{channel});
    
            PSD = periodogram(HRV.HRV{p,t}{channel});
            HRV.PSDmean{p,t}(channel) = mean(PSD);
            HRV.PSDmedian{p,t}(channel) = median(PSD);
            HRV.PSDstd{p,t}(channel) = std(PSD);
            HRV.PSDmin{p,t}(channel) = min(PSD);
            HRV.PSDmax{p,t}(channel) = max(PSD);
            HRV.PSDrange{p,t}(channel) = range(PSD);
        end
    end
end



%% Feature matrix
Feature1 = zeros(2,414);
count = 1;

for p=1:number_of_participants
    for t=1:number_of_trials
        for channel=1:2
            Feature1(channel,count) = HRV.mean{p,t}(channel);
            Feature2(channel,count) = HRV.median{p,t}(channel);
            Feature3(channel,count) = HRV.std{p,t}(channel);
            Feature4(channel,count) = HRV.min{p,t}(channel);
            Feature5(channel,count) = HRV.max{p,t}(channel);
            Feature6(channel,count) = HRV.range{p,t}(channel);
            Feature7(channel,count) = HRV.HRVpNN50{p,t}(channel);
            Feature8(channel,count) = HRV.PSDmean{p,t}(channel);
            Feature9(channel,count) = HRV.PSDmedian{p,t}(channel);
            Feature10(channel,count) = HRV.PSDstd{p,t}(channel);
            Feature11(channel,count) = HRV.PSDmin{p,t}(channel);
            Feature12(channel,count) = HRV.PSDmax{p,t}(channel);
            Feature13(channel,count) = HRV.PSDrange{p,t}(channel);
        end
        count = count + 1;
    end
end
%%
feature_ecg = [Feature1 ; Feature2 ; Feature3 ; Feature4 ; Feature5 ; Feature6 ; Feature7 ; Feature8 ; Feature9 ; Feature10 ; Feature11 ; Feature12 ; Feature13]';


%% Normalizing the feature matrix / Dimension reduction/ save the data

norm_ecg_feature = zscore(feature_ecg);
[U norm_ecg_feature] = pca(norm_ecg_feature,'numcomponents',20);
save('feature_ecg.mat', 'norm_ecg_feature');
