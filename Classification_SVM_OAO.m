% Save this file in a folder called Code. Download indian pines hyperspectral image data and save in a folder called Data at the same level as Code with name Indian_pines_corrected and ground truth as Indian_pines_gt. 
% Denoised version of the indian pines should be stored as Denoised_Indian with the data as variable fs. If denoised data is not available, ust replace with Indian pines original data
clc
clearvars
close all
%% Loading data
cd ..
cd ./Data
load Indian_pines_gt

load Indian_pines_corrected
dat = indian_pines_corrected;

%load Indian_pines
%dat = indian_pines;
cd ..
cd ./Code
%% 
load('Denoised_Indian');
data= fs;
%% Data preprocessing

dat=norm_band(dat);

dim = size(dat);
%data= reshape(dat,dim(1)*dim(2),dim(3));

diml=size(indian_pines_gt);
labels=reshape(indian_pines_gt,diml(1)*diml(2),1);

% Removing unclassified data (corresponding to label 0)
% And classes with very few samples
indices = labels~=0 & labels~=1 & labels~=7 & labels~=9 & labels~=13 & labels~=16 & labels~=4;
data_fin = data(indices,:);
labels_fin=labels(indices);
%% Loading denoised data
% load('Denoised_Indiandsp');
% data_fin= fs;
% load('Denoised_Indiands2'); % Best
% data_fin= fs;
%% Removing buildings label 
ind_ext = labels_fin~= 15 ; 

data_fin = data_fin(ind_ext,:);

labels_fin=labels_fin(ind_ext);
Lbls = [2,3,5,6,8,10,11,12,14];

%% Dividing into train and test data
Lbls = [2,3,5,6,8,10,11,12,14,15];
Lbls_N = length(Lbls);

N_train_cl=zeros(Lbls_N,1);
N_test_cl=zeros(Lbls_N,1);
labels_test = [];
data_test = [];

labels_train = [];
data_train = [];
for i=1:Lbls_N
    label_no=Lbls(i);
    ind_SVM = find(labels_fin==label_no);
    labels_SVM=zeros(length(labels_fin),1);
    labels_SVM(ind_SVM)=label_no;
    
    %% Divide into test and train data
    
    % Percentage of data used for training
    
    per = 0.1; % 10% in the denoising paper, 50% in SVM paper
    
    % Using 1st per of samples for training
    N_train_cl(i) = round(per*length(ind_SVM));
    N_test_cl(i)= length(ind_SVM)-N_train_cl(i);
    
    order = false(length(ind_SVM),1);
    order(1:N_train_cl(i)) = true ;
    order = order(randperm(length(ind_SVM)));
    
    class_ind=ind_SVM(order);
    
    labels_train_cl(1:N_train_cl(i),i)=labels_SVM(class_ind);
    data_train_cl(1:N_train_cl(i),:,i)= data_fin(class_ind,:);
    
    class_ind_test=ind_SVM(~order);
    
    labels_test_cl(1:N_test_cl(i),i)=labels_SVM(class_ind_test);
    data_test_cl(1:N_test_cl(i),:,i)= data_fin(class_ind_test,:);
    
    labels_test=vertcat(labels_test,labels_test_cl(1:N_test_cl(i),i));
    data_test=vertcat(data_test,data_test_cl(1:N_test_cl(i),:,i));
    
end
N_train_tot = sum(N_train_cl);
N_test_tot = sum(N_test_cl);
%% SVM with OAO strategy

for i=1: Lbls_N-1
    for j = i+1:Lbls_N
        
        % SVM for classes i,j
        label1_no=Lbls(i);
        label2_no=Lbls(j);
        
        % Data of classes i,j
       
        % Using equal number of data points for each class 
        minN=min(N_train_cl(i),N_train_cl(j));
        
%         data_train=vertcat(data_train_cl(1:minN,:,i),data_train_cl(1:minN,:,j));
%         labels_train=vertcat(labels_train_cl(1:minN,i),labels_train_cl(1:minN,j));
        
        data_train=vertcat(data_train_cl(1:N_train_cl(i),:,i),data_train_cl(1:N_train_cl(j),:,j));
        labels_train=vertcat(labels_train_cl(1:N_train_cl(i),i),labels_train_cl(1:N_train_cl(j),j));
        
        %% Training the SVM
        
        % Box constraint = 50 for linear SVM, 40 for nonlinear
        % Linear SVM
       % SVMModel = fitcsvm(data_train,labels_train,'Standardize',true,'BoxConstraint',50);  
%         gamma=0.25;
%         KS = sqrt(1/gamma);
%         SVMModel = fitcsvm(data_train,labels_train,'Standardize',true,'KernelFunction','rbf','KernelScale',KS,'BoxConstraint',40);
%  

          c = cvpartition(length(labels_train),'KFold',10);
          opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
            'AcquisitionFunctionName','expected-improvement-plus','UseParallel',1);
      
          % SVM - RBF
      SVMModel = fitcsvm(data_train,labels_train,'Standardize',true,'KernelFunction','rbf',...
          'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
      
   %     SVM - linear
%        SVMModel = fitcsvm(data_train,labels_train,'Standardize',true,'KernelFunction','linear',...
%            'OptimizeHyperparameters',{'BoxConstraint'},'HyperparameterOptimizationOptions',opts);
%       
%         SVMModel = fitcsvm(data_train,labels_train,'Standardize',true,'KernelFunction','linear',...
%           'OptimizeHyperparameters',{'BoxConstraint'},'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus','UseParallel',1));
        save(strcat(num2str(i),'_',num2str(j)),'SVMModel')
    end
end

%% Testing data
N = length(labels_test);
S = zeros(N,Lbls_N);
for i=1: Lbls_N
    f = zeros(N,Lbls_N);
    label1_no=Lbls(i);
    for j = 1: Lbls_N
        if(i==j)
            continue;
        elseif(j<i)
            a = j;
            b = i;
        else
            a = i;
            b = j;
        end
        
        load(strcat(num2str(a),'_',num2str(b)))
        [label,score] = predict(SVMModel,data_test);
        
        A = label== label1_no;
        
        A = double(A);
        % Change all 0s of f to -1s
        A(A==0) = -1;
        
        f(:,j)= A;
    end
    
    S(:,i) = sum(f,2); % Sum all elts of a row
end

labels_tot = zeros(N,1);
for k=1:N
    [~,lbl_t]=max(S(k,:));
    labels_tot(k)= Lbls(lbl_t);
end

% Class accuracy
OA = zeros(Lbls_N,1);
for i=1:Lbls_N
    OA(i)= nnz((labels_tot==Lbls(i))&(labels_test==Lbls(i)))/nnz(labels_test==Lbls(i));
end
OA
OA_overall=nnz(labels_tot==labels_test)/length(labels_test)
OA_mean = mean(OA)

%% Matlab fn
% % gamma=0.25;
% % KS = sqrt(1/gamma);
% % t = templateSVM('Standardize',true,'KernelFunction','rbf','KernelScale',KS,'BoxConstraint',40);
% %Mdl = fitcecoc(data_train,labels_train,'Learners',t);
% Mdl = fitcecoc(data_train,labels_train);
% %%
% [label,score] = predict(Mdl,data_test);
% OA_overall=nnz(label==labels_test)/length(labels_test)
