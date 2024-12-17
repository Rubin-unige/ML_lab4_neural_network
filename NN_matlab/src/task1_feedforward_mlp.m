%% Task 1: Feedforward multi-layer networks (multi-layer perceptrons)

function task1_feedforward_mlp(x, t, hiddenLayerSize, trainRatio, valRatio, testRatio)

    % Choose a Training Function
    trainFcn = 'trainscg';  % Scaled Conjugate Gradient Backpropagation
    
    % Create a Pattern Recognition Network
    net = patternnet(hiddenLayerSize, trainFcn);
    
    % Setup Data Division for Training, Validation, and Testing
    net.divideParam.trainRatio = trainRatio;  
    net.divideParam.valRatio = valRatio;    
    net.divideParam.testRatio = testRatio;
    
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
    
    % Create a "results" folder if it doesn't exist
    if ~exist('results/results_task1', 'dir')
        mkdir('results/results_task1');
    end

    % Save performance and error metrics to a text file
    resultsFile = fullfile('results/results_task1', 'network_performance.txt');
    fileID = fopen(resultsFile, 'a');  % Open file for appending
    fprintf(fileID, 'Hidden Layer Size: %d\n', hiddenLayerSize);
    fprintf(fileID, 'Training Ratio: %.2f\n', trainRatio);
    fprintf(fileID, 'Validation Ratio: %.2f\n', valRatio);
    fprintf(fileID, 'Testing Ratio: %.2f\n', testRatio);
    fprintf(fileID, 'Network Performance (MSE): %.4f\n', performance);
    fprintf(fileID, 'Percent Errors: %.2f%%\n\n', percentErrors * 100);
    fclose(fileID);
    
    % Plot performance during training and save to file
    figure, plotperform(tr);
    saveas(gcf, fullfile('results/results_task1', 'training_performance.png'));
    
    % Plot confusion matrix and save to file
    figure, plotconfusion(t, y);
    saveas(gcf, fullfile('results/results_task1', 'confusion_matrix.png'));

    figure, plotroc(t, y);
    saveas(gcf, fullfile('results/results_task1', 'ROC.png'));
 
    % Optionally close figures after saving
    close all;
end
