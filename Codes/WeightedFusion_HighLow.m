%% Loading Data 
clc
clear all

% go to dataset directory
load('DREAMER.mat');

%% Loading EEG and ECG feature matrices

path = pwd;
cd '../../../Emotion_Recognition_B.Sc_thesis/Codes/';

load('feature_eeg.mat');
load('feature_ecg.mat');


%% Manageing labels

LabelValence = [];
LabelArousal = [];
num_of_participants = 23;

for p=1:num_of_participants
    LabelValence = [LabelValence; DREAMER.Data{1,p}.ScoreValence];
    LabelArousal = [LabelArousal; DREAMER.Data{1,p}.ScoreArousal];
end

% converting numeric labels to High and Low values (1 for high and 0 for low)
LabelValence(LabelValence < 3) = 0;
LabelValence(LabelValence >= 3) = 1;

LabelArousal(LabelArousal < 3) = 0;
LabelArousal(LabelArousal >= 3) = 1;

%% Weighted Fusion for Arousal
%% ECG Data

% number of elements with label arousal of 0 (m1 = 100 - m0 which is number of elements with label arousal of 1)
split = 50;
negative = 0;
negative = sum(LabelArousal(1:split) == 0);

% split data for train and test
FeatureMatrix_split = norm_ecg_feature;
LabelArousal_split = LabelArousal;

test_data =  FeatureMatrix_split(1:split,:);
FeatureMatrix_split(1:split,:) = []; 
test_label = LabelArousal_split(1:split);
LabelArousal_split(1:split) = [];
    
train_data = FeatureMatrix_split;
train_label = LabelArousal_split;


% training the SVM classifier using training data
svm_arousal = fitcecoc(train_data, train_label);

for k=1:split
    true_SVM2(k) = predict(svm_arousal,test_data(k,:)) == test_label(k);
end

% calculating the number of true negative and true positive samples
true_negative = 0;
true_positive = 0;

for i=1:split
    if(true_SVM2(i) == 1 && test_label(i) == 0)
        true_negative = true_negative + 1;
    elseif(true_SVM2(i) == 1 && test_label(i) == 1)
        true_positive = true_positive + 1;
    end
end


% generating W matrix (ECG - Arousal)
w11 = true_negative/negative;
w22 = true_positive/(length(test_label) - negative);


W_ECG_Arousal = zeros(2);
W_ECG_Arousal(1,1) = w11;
W_ECG_Arousal(2,2) = w22;

%% EEG data

% split data for train and test
FeatureMatrix_split = norm_feature_eeg;
    
test_data =  FeatureMatrix_split(1:split,:);
FeatureMatrix_split(1:split,:) = []; 
    
train_data = FeatureMatrix_split;

% train the SVM classifier using train data
svm_arousal = fitcecoc(train_data,train_label);

for k=1:split
    true_SVM2(k) = predict(svm_arousal,test_data(k,:)) == test_label(k);
end

% calculating the number of true negative and true positive samples
true_negative = 0;
true_positive = 0;

for i=1:split
    if(true_SVM2(i) == 1 && test_label(i) == 0)
        true_negative = true_negative + 1;
    elseif(true_SVM2(i) == 1 && test_label(i) == 1)
        true_positive = true_positive + 1;
    end
end

% generating W matrix (EEG - Arousal)
w11 = true_negative/negative;
w22 = true_positive/(length(test_label) - negative);


W_EEG_Arousal = zeros(2);
W_EEG_Arousal(1,1) = w11;
W_EEG_Arousal(2,2) = w22;


%% Weighted Fusion for Valence
%% ECG Data

% number of elements with label arousal of 0 (m1 = split - m0 which is number of elements with label arousal of 1)
negative = 0;
negative = sum(LabelValence(1:split) == 0);

% split data for train and test
FeatureMatrix_split = norm_ecg_feature;
LabelValence_split = LabelValence;

test_data =  FeatureMatrix_split(1:split,:);
FeatureMatrix_split(1:split,:) = []; 
test_label = LabelValence_split(1:split);
LabelValence_split(1:split) = [];
    
train_data = FeatureMatrix_split;
train_label = LabelValence_split;


% training the SVM classifier using training data
svm_valence = fitcecoc(train_data, train_label);

for k=1:split
    true_SVM2(k) = predict(svm_valence,test_data(k,:)) == test_label(k);
end

% 
true_negative = 0;
true_positive = 0;

for i=1:split
    if(true_SVM2(i) == 1 && test_label(i) == 0)
        true_negative = true_negative + 1;
    elseif(true_SVM2(i) == 1 && test_label(i) == 1)
        true_positive = true_positive + 1;
    end
end


%% generating W matrix (ECG - Valence)

w11 = true_negative/negative;
w22 = true_positive/(length(test_label) - negative);


W_ECG_Valence = zeros(2);
W_ECG_Valence(1,1) = w11;
W_ECG_Valence(2,2) = w22;

%% EEG data

% split data for train and test
FeatureMatrix_split = norm_feature_eeg;
    
test_data =  FeatureMatrix_split(1:split,:);
FeatureMatrix_split(1:split,:) = []; 
    
train_data = FeatureMatrix_split;

% train the SVM classifier using train data
svm_valence = fitcecoc(train_data,train_label);

for k=1:split
    true_SVM2(k) = predict(svm_valence,test_data(k,:)) == test_label(k);
end

% calculating the number of true negative and true positive samples
true_negative = 0;
true_positive = 0;

for i=1:split
    if(true_SVM2(i) == 1 && test_label(i) == 0)
        true_negative = true_negative + 1;
    elseif(true_SVM2(i) == 1 && test_label(i) == 1)
        true_positive = true_positive + 1;
    end
end

% generating W matrix (EEG - Valence)
w11 = true_negative/negative;
w22 = true_positive/(length(test_label) - negative);


W_EEG_Valence = zeros(2);
W_EEG_Valence(1,1) = w11;
W_EEG_Valence(2,2) = w22;


W_EEG_V(2,2) = w22;

%% SVM / Weighted Fusion
%% Classification EEG/ECG (Arousal)

% ECG
SVM = fitcecoc(norm_ecg_feature, LabelArousal);
C_ECG_Arousal = zeros(414,2);

TestLabelArousalECG = zeros(414,1);
TestLabelArousalECG = predict(SVM, norm_ecg_feature);

for i=1:414
    C_ECG_Arousal(i,TestLabelArousalECG(i)+1) = 1;
end

% EEG
SVM = fitcecoc(norm_feature_eeg, LabelArousal);
C_EEG_Arousal = zeros(414,2);

TestLabelArousalEEG = zeros(414,1);
TestLabelArousalEEG = predict(SVM, norm_feature_eeg);

for i=1:414
    C_EEG_Arousal(i,TestLabelArousalEEG(i)+1) = 1;
end

% Weighted fusion
C_Arousal = (C_ECG_Arousal * W_ECG_Arousal) + (C_EEG_Arousal * W_EEG_Arousal);

for i=1:414    
    for j=1:2
        if(C_Arousal(i,j) == max(C_Arousal(i,:)))
            C_Arousal_final(i) = j-1;
        end
    end
end

true_prediction = sum(C_Arousal_final == LabelArousal);
ArousalDecisionEEG_ECG = (true_prediction(1)/414) * 100;

%% Classification EEG/ECG (Valence)

% ECG
SVM = fitcecoc(norm_ecg_feature,LabelValence);
C_ECG_Valence = zeros(414,2);

TestLabelValenceECG = zeros(414,1);
TestLabelValenceECG = predict(SVM,norm_ecg_feature);

for i=1:414
    C_ECG_Valence(i,TestLabelValenceECG(i)+1) = 1;
end

% EEG
SVM = fitcecoc(norm_feature_eeg,LabelValence);
C_EEG_Valence = zeros(414,2);

TestLabelValenceEEG = zeros(414,1);
TestLabelValenceEEG = predict(SVM,norm_feature_eeg);

for i=1:414
    C_EEG_Valence(i,TestLabelValenceEEG(i)+1) = 1;
end

% Weighted fusion
C_Valence = (C_ECG_Valence * W_ECG_Valence) + (C_EEG_Valence * W_EEG_Valence);

for i=1:414
    for j=1:2
        if(C_Valence(i,j) == max(C_Valence(i,:)))
            C_Valence_final(i) = j-1;
        end
    end
end

true_prediction = sum(C_Valence_final == LabelValence);
ValenceDecisionEEG_ECG = (true_prediction(1)/414) * 100;
