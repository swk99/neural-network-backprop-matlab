clear; clc;

% Input
number_Pattern = input('Numbers of Input : ');
number_Input = 2;    % fixed for this assignment
number_Out = 1;

X = zeros(number_Pattern, number_Input);
T = zeros(number_Pattern, number_Out);

disp('Enter input:');
for i = 1:number_Pattern
    fprintf('Pattern %d\n', i);
    x1 = input(' x1 = ');
    x2 = input(' x2 = ');
    X(i,1) = x1;
    X(i,2) = x2;
end


disp('Enter outputs:');
for i = 1:number_Pattern
    fprintf('Pattern %d\n', i);
    T(i,1) = input(' Output = ');
end

eta = input('Enter Learning rate (eta): ');


% Initial Weights

disp('Enter initial weights:');
w.w1 = input('w1 = ');
w.w2 = input('w2 = ');
w.w3 = input('w3 = ');
w.w4 = input('w4 = ');
w.w5 = input('w5 = ');
w.w6 = input('w6 = ');
w.w7 = input('w7 = ');
w.w8 = input('w8 = ');
w.w9 = input('w9 = ');


fprintf('\nWeights:\n');
print_weights(w);

% backpropagation algorithms
for epoch = 1:100
    for p = 1:number_Pattern
        x1 = X(p,1);
        x2 = X(p,2);
        t  = T(p);
    
        % Forward: hidden layer
        h1 = sigmoid(w.w1 * x1);
        h2 = sigmoid(w.w3 * x1 + w.w4 * x2);
        h3 = sigmoid(w.w6 * x2);
    
        h = [h1; h2; h3];
    
        % Forward: output layer
        y = w.w2*x1 + w.w5*x2 + w.w9*h1 + w.w8*h2 + w.w7*h3;
    
        beta_out = (t - y);
    
        w_old = w;
        %eta x beta x x(input)
        w.w2 = w.w2 + eta * beta_out * x1;
        w.w5 = w.w5 + eta * beta_out * x2;
    
        w.w9 = w.w9 + eta * beta_out * h1;
        w.w8 = w.w8 + eta * beta_out * h2;
        w.w7 = w.w7 + eta * beta_out * h3;
    
        % Calculate beta
        beta1 = h1*(1-h1) * (beta_out * w_old.w9);
        beta2 = h2*(1-h2) * (beta_out * w_old.w8);
        beta3 = h3*(1-h3) * (beta_out * w_old.w7);
    
        % Input -> Hidden weight updates
        w.w1 = w.w1 + eta * beta1 * x1;
        w.w3 = w.w3 + eta * beta2 * x1;
        w.w4 = w.w4 + eta * beta2 * x2;
        w.w6 = w.w6 + eta * beta3 * x2;
    
        % Results
        fprintf('\n--- Pattern %d ---\n', p);
        fprintf('x = [%g %g], Out = %g\n', x1, x2, t);
        fprintf('Hidden outputs: [%.6f %.6f %.6f]\n', h1, h2, h3);
        fprintf('Output y = %.9f, beta_out = %.9f\n', y, beta_out);
        fprintf('Weights after update:\n');
        print_weights(w);
    end
end


% Functions

function s = sigmoid(x)
    s = 1.0/(1.0 + exp(-x));
end

function print_weights(w)
    fprintf(' w1=%f\n w2=%f\n w3=%f\n w4=%f\n w5=%f\n w6=%f\n w7=%f\n w8=%f\n w9=%f\n', w.w1, w.w2, w.w3, w.w4, w.w5, w.w6, w.w7, w.w8, w.w9);
end



