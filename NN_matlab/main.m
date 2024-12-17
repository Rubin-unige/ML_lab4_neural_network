%% Main Assignment Scripts

addpath("src\");
addpath("data\");

%% Task 1: Feedforward multi-layer networks (multi-layer perceptrons)

%% Prepare dataset for training (iris dataset, wine dataset and breast cancer dataset)

% Iris dataset
iris_data = 'data\iris_dataset\iris.data';
task1_iris_data(iris_data);

% Breast Cancer dataset
breast_cancer_data = "data\breast_cancer_dataset\wdbc.data";
task1_bc_data(breast_cancer_data);

% Wine dataset
input_file = "data\wine_dataset\wine.data"; 
task1_wine_data(input_file);

%% Load Dataseet

% choose which dataset to train
dataset_flag = 'iris';  % Change this flag to 'iris' or 'wine' as needed

switch dataset_flag
    case 'iris'
        load("data\iris_dataset\iris_prepared.mat");
    case 'breast_cancer'
        load("data\breast_cancer_dataset\breast_cancer_prepared.mat");
    case 'wine'
        load("data\wine_dataset\wine_prepared.mat");
    otherwise
        error('Dataset not found');
end

%% Neural Network training setting

hiddenLayerSize = 10; % experiment with this

trainRatio = 70/ 100;
valRatio = 15/100;
testRatio = 15/100;

task1_feedforward_mlp(X, T, hiddenLayerSize, trainRatio, valRatio, testRatio)

%% Task 2: Autoencoder

% choose which two digit you want to train
digit_1 = 3;
digit_2 = 9;

% fucntion to load MNIST data
[x , t] = task2_loadMNIST(digit_1, digit_2);
x = x';

% Setup before training, experiment here to train the best NN
hidden_units = 10; 
max_epochs = 1000;

% Autoencoder function
task2_autoencoder(x, hidden_units, max_epochs, digit_1, digit_2);