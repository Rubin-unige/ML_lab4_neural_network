%% Task 1: Feedforward multi-layer networks (multi-layer perceptrons)

function task1_feedforward_mlp(x, t)

    % Ensure targets are one-hot encoded if they are categorical
    if iscategorical(t)
        t = dummyvar(t);  % Converts categorical targets to one-hot encoding
    end
    
    % Transpose data
    x = x';
    t = t'; 
    
    % Choose a Training Function
    trainFcn = 'trainscg';  % Scaled Conjugate Gradient Backpropagation
    
    % Define the number of neurons in the hidden layer
    hiddenLayerSize = 18;  % Example, you can adjust this
    
    % Create a Pattern Recognition Network
    net = patternnet(hiddenLayerSize, trainFcn);
    
    % Setup Data Division for Training, Validation, and Testing
    net.divideParam.trainRatio = 70/100;  % 70% of data for training
    net.divideParam.valRatio = 15/100;    % 15% of data for validation
    net.divideParam.testRatio = 15/100;   % 15% of data for testing
    
    % Train the Network
    [net, tr] = train(net, x, t);
    
    % Test the Network
    y = net(x);  % Test the network with the same input data (you can use validation data as well)
    e = gsubtract(t, y);  % Compute the error (difference between target and predicted output)
    
    % Compute Performance (e.g., Mean Squared Error)
    performance = perform(net, t, y);
    fprintf('Network Performance (MSE): %.4f\n', performance);
    
    % Calculate misclassification error
    t_ind = vec2ind(t);  % Convert one-hot encoded targets to indices
    y_ind = vec2ind(y);  % Convert network outputs to indices
    percentErrors = sum(t_ind ~= y_ind) / numel(t_ind);  % Misclassification rate
    fprintf('Percent Errors: %.2f%%\n', percentErrors * 100);
    
    % Plot performance during training (optional)
    figure, plotperform(tr);
    
    % Plot confusion matrix (optional)
    figure, plotconfusion(t, y);
end
