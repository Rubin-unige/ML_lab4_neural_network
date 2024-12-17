%% Preprocessing breast cancer data
function task1_bc_data(input_file)

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
        class_index = strcmp(unique_classes, classes{i});
        T(class_index, i) = 1;
    end
    
    save('data\breast_cancer_dataset\breast_cancer_prepared.mat', 'X', 'T');
    disp('Saved preprocessed breast cancer data !!!');

end