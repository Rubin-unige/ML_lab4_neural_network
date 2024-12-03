%% Main Assignment Scripts

addpath("src\");
addpath("data\");

%% Task 1: Feedforward multi-layer networks (multi-layer perceptrons)

%% Iris Dataset
% prepare data
input_file = "data\iris_dataset\iris.data";

% Read the .data file
data = readtable(input_file, FileType="text");

% Extract feature columns (first four columns)
X = table2array(data(:, 1:4)); % Convert to a numerical array
X = X';

% Extract target labels (last column)
classes = data{:, 5}; % Extract the categorical labels

% Convert class labels to one-hot encoding
unique_classes = unique(classes); % Get unique class names
num_classes = numel(unique_classes);
num_samples = size(classes, 1);

T = zeros(num_classes, num_samples); % Initialize one-hot encoding matrix

for i = 1:num_samples
    class_index = find(strcmp(unique_classes, classes{i}));
    T(class_index, i) = 1;
end

save('data\iris_dataset\iris_prepared.mat', 'X', 'T');

%% Breast Cancer dataset

% File path
input_file = "data\breast_cancer_dataset\wdbc.data";

% Read the file
data = readtable(input_file, FileType="text");

% Extract feature columns (assume features are in columns 1:N)
X = table2array(data(:, 3:end)); % Adjust range based on dataset
X = X'; % Transpose to have observations as columns

% Extract target labels (assume labels are in the last column)
classes = data{:, 2}; % Adjust index if necessary

unique_classes = unique(classes); % Find unique class labels
num_classes = numel(unique_classes); % Number of unique classes
num_samples = size(classes, 1); % Number of samples

% Initialize one-hot encoding matrix
T = zeros(num_classes, num_samples);


for i = 1:num_samples
    class_index = find(strcmp(unique_classes, classes{i}));
    T(class_index, i) = 1;
end

save('data\breast_cancer_dataset\breast_cancer_prepared.mat', 'X', 'T');


%% Wine dataset
% File path
input_file = "data\wine_dataset\wine.data";  % Adjust path based on your dataset location

% Read the file
data = readtable(input_file, FileType="text");

% Extract feature columns (all except the first column which is the ID)
X = table2array(data(:, 2:end));  % Features are columns 2 to N

% Transpose to ensure the features are in rows and observations in columns
X = X'; 

% Extract target labels (class labels are in the first column)
classes = data{:, 1};  % Column 1 contains the class labels

% Find unique class labels
unique_classes = unique(classes);
num_classes = numel(unique_classes);  % Number of unique wine types (usually 3)
num_samples = size(classes, 1);  % Number of samples

% Initialize the one-hot encoding matrix
T = zeros(num_classes, num_samples);

% Encode each sample's class into the one-hot matrix
for i = 1:num_samples
    class_index = find(unique_classes == classes(i));  % Find the index of the class
    T(class_index, i) = 1;  % Set the corresponding one-hot position to 1
end
save('data\wine_dataset\wine_prepared.mat', 'X', 'T');  % Save the feature and target matrices


% %% Task 2: Autoencoder
% 
% % choose which two digit you want to train
% digit_1 = 1;
% digit_2 = 8;
% 
% % fucntion to load MNIST data
% [x , t] = task2_loadMNIST(digit_1, digit_2);
% x = x';
% disp(size(x));
% 
% % Setup before training, experiment here to train the best NN
% hidden_units = 10; 
% max_epochs = 10;
% 
% % Autoencoder function
% task2_autoencoder(x, t, hidden_units, max_epochs);