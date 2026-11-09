close all;

clear;

clc;

% Φόρτωση δεδομένων

load('data21.mat');

load('data22.mat');

% Αρχικοποίηση σταθερών και παραμέτρων

N = 500; % Πλήθος pixel που διατηρούνται

T = [eye(N), zeros(N, 784 - N)]; % Μετασχηματισμός

n = 4; % Επιλογή της στήλης X_n που χρησιμοποιείται

iterations = 2 * 1e3; % Μέγιστος αριθμός επαναλήψεων

m = 1e-2; % Βήμα εκπαίδευσης

l = 0.1; % Παράγοντας ομαλοποίησης (Adam)

% Διαμόρφωση δεδομένων εισόδου

X_test = [X_n(1:N, n); zeros(784 - N, 1)];

X_test_2D = reshape(X_test, 28, 28);

X_label = reshape(X_i(:, n), 28, 28);

% Αρχικοποίηση μεταβλητών για ελαχιστοποίηση κόστους

Minimized_Error = Inf(1, iterations);

% Επανάληψη για εύρεση βέλτιστης λύσης

for trial = 1:10

  % Αρχικοποίηση Gaussian εισόδου

  Z = randn(10, 1);

  error = zeros(1, iterations);

  for iter = 1:iterations

    % Υπολογισμός δικτύου (ReLU -> Sigmoid)

    W1 = A_1 * Z + B_1;

    Z1 = max(0, W1); % Ενεργοποίηση ReLU

    W2 = A_2 * Z1 + B_2;

 X = 1 ./ (1 + exp(W2)); % Ενεργοποίηση Sigmoid

 % Εφαρμογή μετασχηματισμού

 X_transformed = T * X;

 % Υπολογισμός κόστους

 norm_error = sum((X_test(1:N) - X_transformed).^2);

 error1 = N * log(norm_error);

 error2 = sum(Z.^2);

 error(iter) = error1 + error2;

 % Υπολογισμός gradient

 U2 = -2 * T' * (X_test(1:N) - T * X) / norm_error;

 V2 = U2 .* diff_sigmoid(W2);

 U1 = A_2' * V2;

 V1 = U1 .* diff_Relu(W1);

 U0 = A_1' * V1;

 diff_Cost = N * U0 + 2 * Z;

 % Ενημέρωση με Adam

 if iter == 1 && trial == 1

   Adam = diff_Cost.^2;

 else

   Adam = (1 - l) * Adam + l * diff_Cost.^2;

 end

 a = 1e-20; % Μικρή σταθερά για αποφυγή διαίρεσης με το 0

 Z = Z - m * diff_Cost ./ sqrt(a + Adam);

end

  % Ενημέρωση βέλτιστης λύσης

  if Minimized_Error(iterations) > error(iterations)

    Z_minimized = Z;

    Minimized_Error = error;

  end

end

% Υπολογισμός της τελικής εικόνας

W1 = A_1 * Z_minimized + B_1;

Z1 = max(0, W1);

W2 = A_2 * Z1 + B_2;

X_predicted = 1 ./ (1 + exp(W2));

% Μετατροπή σε μορφή εικόνας

X_predicted_2D = reshape(X_predicted, 28, 28);

% Απεικόνιση των αποτελεσμάτων

figure;

plot(Minimized_Error);

figure;

subplot(1,3,1);

imshow(X_label)

subplot(1,3,2);

imshow(X_test_2D);

subplot(1,3,3);

imshow(X_predicted_2D);
