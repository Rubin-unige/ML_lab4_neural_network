function task2_autoencoder(x, hidden_units, max_epochs, digit_1, digit_2)
    % Train the autoencoder
    autoenc = trainAutoencoder(x, hidden_units, 'MaxEpochs', max_epochs); 
    
    % Encode the data using the trained autoencoder
    encodedData = encode(autoenc, x);
    
    % Get the total number of data points in the encoded data
    numDataPoints = size(encodedData, 2);  % Number of data points
    
    % Split data points into two classes, handle odd numbers
    numClass1 = floor(numDataPoints / 2);  % Class 1 size (rounding down)
    numClass2 = numDataPoints - numClass1; % Class 2 size (remaining points)
    
    % Create the label vector (1 for class 1, 2 for class 2)
    labels = [ones(1, numClass1), 2*ones(1, numClass2)];
    
    % Plot the encoded data with labels using plotcl
    figure;
    plotcl(encodedData', labels);  % Transpose encodedData for correct dimension
    title(sprintf('Encoded Data of Digits %d and %d', digit_1, digit_2));

    % Save the plotcl figure
    saveas(gcf, sprintf('results/results_task2/encoded_data_%d_%d_plot.png', digit_1, digit_2)); 

    % Decode the data using the trained autoencoder
    reconstructedData = predict(autoenc, x);
    
    % Number of images to display
    numImages = 10;  % You can change this to display more/less images
    
    % Plot original and reconstructed images side by side
    figure;
    for i = 1:numImages
        % Original image
        subplot(2, numImages, i);
        imshow(reshape(x(:, i), 28, 28));  % Reshape to 28x28 and display
        title(['Orig ', num2str(i)]);
        
        % Reconstructed image
        subplot(2, numImages, i + numImages);
        imshow(reshape(reconstructedData(:, i), 28, 28));  % Reshape to 28x28 and display
        title(['recon ', num2str(i)]);
    end

    % Save the entire figure as one image
    saveas(gcf, sprintf('results/results_task2/compare_image_%d_%d.png', digit_1, digit_2)); 
end
