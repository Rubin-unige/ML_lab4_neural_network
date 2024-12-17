%% Task 2: autoencoder load MNIST data

function [ x , t] = task2_loadMNIST(digit_1, digit_2)

    addpath("data\mnist\");

    % load MNIST data using helper function
    [x, t] = loadMNIST(0, [digit_1, digit_2]);

end