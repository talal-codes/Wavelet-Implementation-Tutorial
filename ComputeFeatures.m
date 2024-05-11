function [Energy,Std] = ComputeFeatures(InputSignal,wname,lev)

% Loop through each signal channel or Phases
for i=1:min(size(InputSignal))
    % Get a single channel/Phase signal
    sig = InputSignal(:,i);

    % Apply 1-D Multilevel DWT on the siganl and get all the detailed and
    % approx coefficients in 'c'
    [c,~] = wavedec(sig,lev,wname);
    
    % Compute Energy of all the coefficients combined
    Energy(i) = sum(c.^2);

    % Compute Standard Deviation of all the coefficients combined
    Std(i) = std(sig);

end

end