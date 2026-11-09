function y = diff_sigmoid(x)

  y = zeros(1,length(x));

  for i = 1:length(x)

       y(i) = -(exp(x(i)) / (1 + exp(x(i)))^2);

  end

  y = y'; %σε στήλη

end
