% Save this file in a folder called Code. Download indian pines hyperspectral image data and save in a folder called Data at the same level as Code with name Indian_pines_corrected and ground truth as Indian_pines_gt. 
% Denoised version of the indian pines should be stored as Denoised_Indiandsp with the data as variable fs. If denoised data is not available, ust replace with Indian pines original data
clc
clearvars
close all
%% Loading data
cd ..
cd ./Data
load Indian_pines_gt

load Indian_pines_corrected
dat = indian_pines_corrected;

cd ..
cd ./Code

dat=norm_band(dat);

dim = size(dat);
data= reshape(dat,dim(1)*dim(2),dim(3));

%% 

%% Data preprocessing


diml=size(indian_pines_gt);
labels=reshape(indian_pines_gt,diml(1)*diml(2),1);

% Removing unclassified data (corresponding to label 0 )
% And classes with very few samples
indices = labels~=0 & labels~=1 & labels~=7 & labels~=9 & labels~=13 & labels~=16 & labels~=4;
data_fin = data(indices,:);

labels_fin=labels(indices);

%% Loading denoised data
load('Denoised_Indiandsp');
data_fin= fs;
%% Removing buildings label 
ind_ext = labels_fin~= 15 ; 

data_fin = data_fin(ind_ext,:);

labels_fin=labels_fin(ind_ext);
Lbls = [2,3,5,6,8,10,11,12,14];

%% Dividing into train and test data
%Lbls = [2,3,5,6,8,10,11,12,14,15];
Lbls_N = length(Lbls);

N_train_cl=zeros(Lbls_N,1);
N_test_cl=zeros(Lbls_N,1);

data_test = [];
%data_train=[];
labels_Test=[];

for i=1:Lbls_N
    label_no=Lbls(i);
    ind_SVM = find(labels_fin==label_no);
    labels_SVM=zeros(length(labels_fin),1);
    labels_SVM(ind_SVM)=label_no;
    
    %% Divide into test and train data
    
    % Percentage of data used for training
    
    per = 0.5; % 10% in the denoising paper, 50% in SVM paper
    
    % Use equal number of test
    % Using 1st per of samples for training
    N_train_cl(i) = round(per*length(ind_SVM));
    N_test_cl(i)= length(ind_SVM)-N_train_cl(i);
    
    order = false(length(ind_SVM),1);
    order(1:N_train_cl(i)) = true ;
    order = order(randperm(length(ind_SVM)));
    
    %class_ind=ind_SVM(1:N_train_cl(i));
    class_ind=ind_SVM(order);
    
    labels_train_cl(1:N_train_cl(i),i)=labels_SVM(class_ind);
    data_train_cl(1:N_train_cl(i),:,i)= data_fin(class_ind,:);
    
    % class_ind_test=ind_SVM(N_train_cl(i)+1:end);
    class_ind_test=ind_SVM(~order);
    
    labels_test_cl(1:N_test_cl(i),i)=labels_SVM(class_ind_test);
    data_test_cl(1:N_test_cl(i),:,i)= data_fin(class_ind_test,:);
    
    data_test=vertcat(data_test,data_test_cl(1:N_test_cl(i),:,i));
    labels_Test=vertcat(labels_Test,labels_test_cl(1:N_test_cl(i),i));
    
    % data_train= vertcat(data_train,data_train_cl(1:N_train_cl(i),:,i));
    
end
N_train_tot = sum(N_train_cl);
N_test_tot = sum(N_test_cl);
%% SVM with OAA strategy
tic;

OA = zeros(Lbls_N,1);

for i=1:Lbls_N
    %% Specifying training and testing data
    data_train=[];
    labels_train=zeros(2*N_train_cl(i),1);
    labels_test=zeros(N_test_tot,1);
    ind = 1;
    ind2=1;
    for n=1:Lbls_N
        
        if(i==n)
            data_train(1:N_train_cl(i),:)=data_train_cl(1:N_train_cl(i),:,i);
            labels_train(1:N_train_cl(i))= labels_train_cl(1:N_train_cl(i),i);
            labels_test(ind2:N_test_cl(i)+ind2-1)= labels_test_cl(1:N_test_cl(i),i);
            %break;
        else
            data_other(ind:N_train_cl(n)+ind-1,:)=data_train_cl(1:N_train_cl(n),:,n);
            ind = ind + N_train_cl(n);
            ind2 = ind2 + N_test_cl(n);
        end
    end
    
     order2 = false(length(data_other),1);
    order2(1:N_train_cl(i)) = true ;
    order2 = order2(randperm(length(data_other)));
    
   data_train=vertcat(data_train,data_other(order2,:,:));
    %% Training the SVM
    
    % Box constraint = 50 for linear SVM, 40 for nonlinear
    
    % Linear SVM
  %  SVMModel = fitcsvm(data_train,labels_train,'Standardize',true,'BoxConstraint',50);
    
    
    % SVM - RBF
%             gamma=0.25;
%             KS = sqrt(1/gamma);
%         SVMModel = fitcsvm(data_train,labels_train,'Standardize',true,'KernelFunction','rbf','KernelScale',KS,'BoxConstraint',40);
%     
                 c = cvpartition(length(labels_train),'KFold',10);
              opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
                'AcquisitionFunctionName','expected-improvement-plus');
    
%            SVMModel = fitcsvm(data_train,labels_train,'Standardize',true,'KernelFunction','rbf',...
%                'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
%     
 % SVM - linear
     %  SVMModel = fitcsvm(data_train,labels_train,'Standardize',true,'KernelFunction','linear',...
     %      'OptimizeHyperparameters',{'BoxConstraint'},'HyperparameterOptimizationOptions',opts);
     
      SVMModel = fitcsvm(data_train,labels_train,'Standardize',true,'KernelFunction','linear',...
          'OptimizeHyperparameters',{'BoxConstraint'},'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','UseParallel',1));
     
    save(num2str(i),'SVMModel')
    %% Testing data
    
    [label,score] = predict(SVMModel,data_test);
    
    %   OA(i)=nnz(label==labels_test)/length(labels_test);
    OA(i)=nnz(label & labels_test)/nnz(labels_test);
end
OA
OA_mean= mean(OA)
toc;
%% Testing data
N = length(labels_Test);
S = zeros(N,Lbls_N);
S_sc = zeros(N,Lbls_N);

for i=1: Lbls_N
    label1_no=Lbls(i);
    
    load(num2str(i))
    [label,score] = predict(SVMModel,data_test);
    
    cl_score = score(:,2); % Of belonging to class label1_no
    
    A = label== label1_no;
    %A = label == 1;
    
    A = double(A);
    
    S(:,i) = A;
    S_sc(:,i)= cl_score; 
end

labels_tot = zeros(N,1);
for k=1:N
   % [~,lbl_t]=max(S(k,:));
    [~,lbl_t]=max(S_sc(k,:));
    labels_tot(k)= Lbls(lbl_t);
end

% Class accuracy
OA2 = zeros(Lbls_N,1);
for i=1:Lbls_N
    OA2(i)= nnz((labels_tot==Lbls(i))&(labels_Test==Lbls(i)))/nnz(labels_Test==Lbls(i));
end
OA2
OA_overall=nnz(labels_tot==labels_Test)/length(labels_Test)
OA_mean2 = mean(OA2)
%% Results
% class 3 - 90.26% (linear SVM with C=1,50 without standardization)
% 91.94% with standardization; 91.37% (SVM-RBF); 92.71% (SVM-RBF with
% gamma=0.25)

% We are Additionally standardizing data here (not done in the papers)

% SVM RBF for all classes : mean OA = 90.85% for IP corrected, 90% for IP
