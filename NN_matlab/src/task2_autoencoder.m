function task2_autoencoder(x, t, hidden_units, max_epochs)
    
    % Train the autoencoder
    autoenc = trainAutoencoder(x, hidden_units, 'MaxEpochs', max_epochs);                 

    % Encode the data using the trained autoencoder
    encodedData = encode(autoenc, x);
    
end
