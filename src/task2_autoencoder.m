function task2_autoencoder(x, t, hidden_units, max_epochs)
    
    % Train the autoencoder
    autoenc = trainAutoencoder(x, hidden_units, 'MaxEpochs', max_epochs);                 

    % Encode the data using the trained autoencoder
    encodedData = encode(autoenc, x);
    
     % Decode (reconstruct) the data using the trained autoencoder
    reconstructedData = predict(autoenc, x);
    
    % Plot the encoded data using plotcl (assuming 2D plot with classes 1 and 8)
    figure;
    subplot(1,2,1);
    plotcl(encodedData', t);
    title('Encoded Data (Hidden Layer Representation)');
    
    % Plot original vs. reconstructed images
    numImages = size(x, 1);  % Number of images
    imageIndex = 1;          % Choose an index for comparison (e.g., 1st image)
    
    % Reshape the original and reconstructed images to match the original image size
    originalImage = reshape(x(imageIndex, :), [28, 28]);  % Assuming images are 28x28 pixels
    reconstructedImage = reshape(reconstructedData(imageIndex, :), [28, 28]);
    
    % Display the original and reconstructed images
    subplot(1,2,2);
    montage({originalImage, reconstructedImage}, 'Size', [1 2]);
    title('Original (Left) vs Reconstructed (Right) Image');
   
end
