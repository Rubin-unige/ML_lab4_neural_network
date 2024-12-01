%% Main Assignment Scripts

addpath("src\");

%% Task 2: Autoencoder

% choose which two digit you want to train
digit_1 = 1;
digit_2 = 8;

% fucntion to load MNIST data
[x , t] = task2_loadMNIST(digit_1, digit_2);
x = x';
disp(size(x));

% Setup before training, experiment here to train the best NN
hidden_units = 10; 
max_epochs = 10;

% Autoencoder function
task2_autoencoder(x, t, hidden_units, max_epochs);