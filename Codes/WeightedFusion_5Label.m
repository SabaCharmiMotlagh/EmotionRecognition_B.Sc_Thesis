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

for label3=1:num_of_participants
    LabelValence = [LabelValence; DREAMER.Data{1,label3}.ScoreValence];
    LabelArousal = [LabelArousal; DREAMER.Data{1,label3}.ScoreArousal];
end

%% Weighted Fusion
%% Arousal
%% ECG data

% number of elements with label arousal of 1, 2, 3, 4, and 5 for test part
split = 50;
true_label1 = sum(LabelArousal(1:split) == 1);
true_label2 = sum(LabelArousal(1:split) == 2);
true_label3 = sum(LabelArousal(1:split) == 3);
true_label4 = sum(LabelArousal(1:split) == 4);
true_label5 = sum(LabelArousal(1:split) == 5);

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


% number of truly predicted elements with label arousal of 1, 2, 3, 4, and 5 for test part
label1 = 0;
laebl2 = 0;
label3 = 0;
label4 = 0;
label5 = 0;

for i=1:split
    if(true_SVM2(i) == 1 && LabelArousal(i) == 1)
        label1 = label1+1;
    elseif(true_SVM2(i) == 1 && LabelArousal(i) == 2)
        laebl2 = laebl2+1;
    elseif(true_SVM2(i) == 1 && LabelArousal(i) == 3)
        label3 = label3+1;
    elseif(true_SVM2(i) == 1 && LabelArousal(i) == 4)
        label4 = label4+1;
    elseif(true_SVM2(i) == 1 && LabelArousal(i) == 5)
        label5 = label5+1;
    end
end

% generating W matrix (ECG-Arousal)

w11 = label1/true_label1;
w22 = laebl2/true_label2;
w33 = label3/true_label3;
w44 = label4/true_label4;
w55 = label5/true_label5;

W_ECG_Arousal = zeros(5);
W_ECG_Arousal(1,1) = w11;
W_ECG_Arousal(2,2) = w22;
W_ECG_Arousal(3,3) = w33;
W_ECG_Arousal(4,4) = w44;
W_ECG_Arousal(5,5) = w55;

%% EEG data

% split data for train and test
FeatureMatrix_split = norm_feature_eeg;

test_data =  FeatureMatrix_split(1:split,:);
FeatureMatrix_split(1:split,:) = []; 
    
train_data = FeatureMatrix_split;


% training the SVM classifier using training data
svm_arousal = fitcecoc(train_data, train_label);

for k=1:split
    true_SVM2(k) = predict(svm_arousal,test_data(k,:)) == test_label(k);
end


label1 = 0;
laebl2 = 0;
label3 = 0;
label4 = 0;
label5 = 0;

for i=1:split
    if(true_SVM2(i) == 1 && LabelArousal(i) == 1)
        label1 = label1+1;
    elseif(true_SVM2(i) == 1 && LabelArousal(i) == 2)
        laebl2 = laebl2+1;
    elseif(true_SVM2(i) == 1 && LabelArousal(i) == 3)
        label3 = label3+1;
    elseif(true_SVM2(i) == 1 && LabelArousal(i) == 4)
        label4 = label4+1;
    elseif(true_SVM2(i) == 1 && LabelArousal(i) == 5)
        label5 = label5+1;
    end
end

% generating W matrix (EEG-Arousal)

w11 = label1/true_label1;
w22 = laebl2/true_label2;
w33 = label3/true_label3;
w44 = label4/true_label4;
w55 = label5/true_label5;

W_EEG_Arousal = zeros(5);
W_EEG_Arousal(1,1) = w11;
W_EEG_Arousal(2,2) = w22;
W_EEG_Arousal(3,3) = w33;
W_EEG_Arousal(4,4) = w44;
W_EEG_Arousal(5,5) = w55;

%% Valence
%% ECG data

% number of elements with label arousal of 1, 2, 3, 4, and 5 for test part
split = 207;
true_label1 = sum(LabelValence(1:split) == 1);
true_label2 = sum(LabelValence(1:split) == 2);
true_label3 = sum(LabelValence(1:split) == 3);
true_label4 = sum(LabelValence(1:split) == 4);
true_label5 = sum(LabelValence(1:split) == 5);

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


% number of truly predicted elements with label arousal of 1, 2, 3, 4, and 5 for test part
label1 = 0;
laebl2 = 0;
label3 = 0;
label4 = 0;
label5 = 0;

for i=1:split
    if(true_SVM2(i) == 1 && LabelValence(i) == 1)
        label1 = label1 + 1;
    elseif(true_SVM2(i) == 1 && LabelValence(i) == 2)
        laebl2 = laebl2 + 1;
    elseif(true_SVM2(i) == 1 && LabelValence(i) == 3)
        label3 = label3 + 1;
    elseif(true_SVM2(i) == 1 && LabelValence(i) == 4)
        label4 = label4 + 1;
    elseif(true_SVM2(i) == 1 && LabelValence(i) == 5)
        label5 = label5 + 1;
    end
end

% generating W matrix (ECG-Valence)

w11 = label1/true_label1;
w22 = laebl2/true_label2;
w33 = label3/true_label3;
w44 = label4/true_label4;
w55 = label5/true_label5;

W_ECG_Valence = zeros(5);
W_ECG_Valence(1,1) = w11;
W_ECG_Valence(2,2) = w22;
W_ECG_Valence(3,3) = w33;
W_ECG_Valence(4,4) = w44;
W_ECG_Valence(5,5) = w55;

%% EEG data

% split data for train and test
FeatureMatrix_split = norm_feature_eeg;

test_data =  FeatureMatrix_split(1:split,:);
FeatureMatrix_split(1:split,:) = []; 
    
train_data = FeatureMatrix_split;


% training the SVM classifier using training data
svm_valence = fitcecoc(train_data, train_label);

for k=1:split
    true_SVM2(k) = predict(svm_valence,test_data(k,:)) == test_label(k);
end


label1 = 0;
laebl2 = 0;
label3 = 0;
label4 = 0;
label5 = 0;

for i=1:split
    if(true_SVM2(i) == 1 && LabelValence(i) == 1)
        label1 = label1 + 1;
    elseif(true_SVM2(i) == 1 && LabelValence(i) == 2)
        laebl2 = laebl2 + 1;
    elseif(true_SVM2(i) == 1 && LabelValence(i) == 3)
        label3 = label3 + 1;
    elseif(true_SVM2(i) == 1 && LabelValence(i) == 4)
        label4 = label4 + 1;
    elseif(true_SVM2(i) == 1 && LabelValence(i) == 5)
        label5 = label5 + 1;
    end
end

% generating W matrix (EEG-Valence)

w11 = label1/true_label1;
w22 = laebl2/true_label2;
w33 = label3/true_label3;
w44 = label4/true_label4;
w55 = label5/true_label5;

W_EEG_Valence = zeros(5);
W_EEG_Valence(1,1) = w11;
W_EEG_Valence(2,2) = w22;
W_EEG_Valence(3,3) = w33;
W_EEG_Valence(4,4) = w44;
W_EEG_Valence(5,5) = w55;

%% SVM / Weighted Fusion
%% Classification EEG/ECG (Arousal)

% ECG
SVM = fitcecoc(norm_ecg_feature,LabelArousal);
C_ECG_Arousal = zeros(414,5);

TestLabelArousalECG = zeros(414,1);
TestLabelArousalECG = predict(SVM,norm_ecg_feature);

for i=1:414
    C_ECG_Arousal(i,TestLabelArousalECG(i)) = 1;
end

% EEG
SVM = fitcecoc(norm_feature_eeg,LabelArousal);
C_EEG_Arousal = zeros(414,5);

TestLabelArousalEEG = zeros(414,1);
TestLabelArousalEEG = predict(SVM,norm_feature_eeg);

for i=1:414
    C_EEG_Arousal(i,TestLabelArousalEEG(i)) = 1;
end

% weighted fusion
C_Arousal = (C_ECG_Arousal * W_ECG_Arousal) + (C_EEG_Arousal * W_EEG_Arousal);

for i=1:414

    for j=1:5
        if(C_Arousal(i,j)==max(C_Arousal(i,:)))
            C_Arousal_final(i)=j;
        end
    end
end

label1 = 0;

for i=1:414
    if(C_Arousal_final(i)==LabelArousal(i))
        label1 = label1 + 1;
    end
end


ArousalDecisionEEG_ECG = (label1/414)*100;


%% Classification EEG/ECG (Valence)
% ECG
SVM = fitcecoc(norm_ecg_feature,LabelValence);
C_ECG_Valence = zeros(414,5);

TestLabelValenceECG = zeros(414,1);
TestLabelValenceECG = predict(SVM,norm_ecg_feature);

for i=1:414
    C_ECG_Valence(i,TestLabelValenceECG(i)) = 1;
end

% EEG
SVM = fitcecoc(norm_feature_eeg,LabelValence);
C_EEG_Valence = zeros(414,5);

TestLabelValenceEEG = zeros(414,1);
TestLabelValenceEEG = predict(SVM,norm_feature_eeg);

for i=1:414
    C_EEG_Valence(i,TestLabelValenceEEG(i)) = 1;
end

% weighted fusion
C_Valence = (C_ECG_Valence * W_ECG_Valence) + (C_EEG_Valence * W_EEG_Valence);

for i=1:414

    for j=1:5
        if(C_Valence(i,j)==max(C_Valence(i,:)))
            C_Valence_final(i)=j;
        end
    end
end

label1 = 0;

for i=1:414
    if(C_Valence_final(i)==LabelValence(i))
        label1 = label1 + 1;
    end
end


ValenceDecisionEEG_ECG = (label1/414)*100;