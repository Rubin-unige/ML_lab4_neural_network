%% Preprocessing iris data
function task1_iris_data(input_file)
    % Read the .data file
    data = readtable(input_file, 'FileType', 'text');
    
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
        class_index = strcmp(unique_classes, classes{i});
        T(class_index, i) = 1;
    end
    
    % Save the preprocessed data
    save('data\iris_dataset\iris_prepared.mat', 'X', 'T');
    disp('saved processed iris data !!!');
end
