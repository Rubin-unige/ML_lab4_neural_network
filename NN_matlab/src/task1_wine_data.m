function task1_wine_data(input_file)

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
        class_index = unique_classes == classes(i);  % Find the index of the class
        T(class_index, i) = 1;  % Set the corresponding one-hot position to 1
    end
    save('data\wine_dataset\wine_prepared.mat', 'X', 'T');  
    disp('Saved preprocessed wine dataset !!!');

end
