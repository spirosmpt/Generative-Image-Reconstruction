close all;

clear;

clc;

% Φόρτωση δεδομένων

load('data21.mat');

load('data23.mat');

% Αρχικοποίηση σταθερών

N = 49; % Αριθμός pixel της μετασχηματισμένης εικόνας (7x7 ανάλυση)

T = zeros(49, 784); % Πίνακας μετασχηματισμού

diagonal = zeros(7, 112);

step = 0;

% Δημιουργία διαγώνιου πίνακα

for row = 1:7

  for block = 0:3

       for pos = 1+step:4+step

         diagonal(row, block*28 + pos) = 1/16;

       end

  end

  step = pos;

end

% Συμπλήρωση του πίνακα μετασχηματισμού

for index = 0:6

  T(index*7 + 1:(index+1)*7, index*112 + 1:(index+1)*112) = diagonal;

end

% Επεξεργασία της επιλεγμένης στήλης

n = 4;

X_label = reshape(X_i(:, n), 28, 28);

X_test = reshape(X_n(:, n), 7, 7);

X_test_2D = kron(X_test, ones(4, 4)); % Μετατροπή σε πλήρη ανάλυση

% Παράμετροι Gradient Descent

s = 1e-2; % Μέγεθος βήματος

l = 0.1; % Συντελεστής εξομάλυνσης για το Adam

iterations = 2 * 1e3; % Μέγιστος αριθμός επαναλήψεων

Minimized_error = Inf(1, iterations); % Αρχικοποίηση ελαχιστοποιημένου κόστους

% Επαναλήψεις εύρεσης βέλτιστης λύσης

for trial = 1:10

  Z = randn(10, 1); % Τυχαία αρχικοποίηση

  error = zeros(1, iterations);

  for iter = 1:iterations

    % Υπολογισμός των δικτύων

    W1 = A_1 * Z + B_1;

    Z1 = max(0, W1); % Ενεργοποίηση ReLU

    W2 = A_2 * Z1 + B_2;

    X = 1 ./ (1 + exp(W2)); % Ενεργοποίηση Sigmoid

    % Εφαρμογή του μετασχηματισμού

    X_transformed = T * X;

    % Υπολογισμός του κόστους

    norm_error = sum((X_test(:) - X_transformed(:)).^2);

    error1 = N * log(norm_error);

    error2 = sum(Z.^2);

    error(iter) = error1 + error2;

   % Υπολογισμός Gradient

   U2 = -2 * T' * (X_test(:) - T * X) / norm_error;

   V2 = U2 .* diff_sigmoid(W2);

   U1 = A_2' * V2;

   V1 = U1 .* diff_Relu(W1);

   U0 = A_1' * V1;

   diff_error = N * U0 + 2 * Z;

   % Ενημέρωση με Adam

   if iter == 1 && trial == 1

      Adam = diff_error.^2;

   else

      Adam = (1 - l) * Adam + l * diff_error.^2;

   end

   a = 1e-20;

   Z = Z - s * diff_error ./ sqrt(a + Adam);

 end

 % Ενημέρωση βέλτιστης λύσης

 if Minimized_error(iterations) > error(iterations)

   Z_minimized = Z;

   Minimized_error = error;

 end

end

% Υπολογισμός της τελικής λύσης

W1 = A_1 * Z_minimized + B_1;

Z1 = max(0, W1);

W2 = A_2 * Z1 + B_2;

X_predicted = 1 ./ (1 + exp(W2));

X_predicted_2D = reshape(X_predicted, 28, 28);

% Απεικόνιση των αποτελεσμάτων

figure;

plot(Minimized_error);

figure;

subplot(1, 3, 1);

imshow(X_label);

subplot(1, 3, 2);

imshow(X_test_2D);

subplot(1, 3, 3);

imshow(X_predicted_2D);
