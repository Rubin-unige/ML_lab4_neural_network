%% Main assignment script

addpath("Data");
addpath("src");
    
%% Task 1: Feedforward multi-layer networks (multi-layer perceptrons)

%% Iris dataset
% Read the dataset
iris_dataset = readtable("Data\iris_dataset\iris.data", 'FileType', 'text');
% Rename the columns          
iris_dataset.Properties.VariableNames = {'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'};
% Convert the Species column to a categorical variable
iris_dataset.Species = categorical(iris_dataset.Species);
disp(head(iris_dataset)); % display for test
% Prepare data
x_iris = iris_dataset{:, 1:4};
t_iris = iris_dataset.Species;
% Call pattern recognition function
task1_feedforward_mlp(x_iris, t_iris);


