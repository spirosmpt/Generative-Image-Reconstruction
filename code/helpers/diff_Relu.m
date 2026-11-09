function y = diff_Relu(k)

  y = zeros(size(k));

  indices = (k > 0);

  y(indices) = 1;

end
