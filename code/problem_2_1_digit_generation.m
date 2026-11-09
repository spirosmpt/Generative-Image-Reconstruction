close all;
clc;
load('data21.mat');

% Generate 100 handwritten-digit images.
X2D_100im = zeros(280,280);
for j = 0:9
    for i = 0:9
        Z0 = randn(10,1);
        W1 = A_1 * Z0 + B_1;
        Z1 = max(W1,0);
        W2 = A_2 * Z1 + B_2;
        X = 1 ./ (1 + exp(W2));
        X2D = reshape(X,28,28);
        row_start = j * 28 + 1;
        row_end = row_start + 27;
        col_start = i * 28 + 1;
        col_end = col_start + 27;
        X2D_100image(row_start:row_end, col_start:col_end) = X2D;
    end
end
imshow(X2D_100image);