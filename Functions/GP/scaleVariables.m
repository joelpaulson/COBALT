function [Xnew,Ynew] = scaleVariables(X,Y,lb,ub)
% Copyright (c) by Eric Bradford, Artur M. Schweidtmann and Alexei Lapkin, 2017-13-12.

%% Scales input and output variabels
Xnew = zeros(size(X)) ; % scaled inputs
Ynew = zeros(size(Y)) ; % scaled outputs

%% Scale input variables to [0,1]
for i = 1:size(X,2)
    Xnew(:,i) = (X(:,i)-lb(i)) / (ub(i)-lb(i)) ;
end

%% Scale output variables to zero mean and unit variance
ny = size(Y,2);
MeanOfOutputs = zeros(ny,1) ;
stdOfOutputs = zeros(ny,1) ;
for i = 1:ny
    MeanOfOutputs(i) = mean(Y(:,i)); % calculate mean
    stdOfOutputs(i) = std(Y(:,i)) ; % calculate standard deviation
    Ynew(:,i) = (Y(:,i) - MeanOfOutputs(i)) / stdOfOutputs(i) ; % scale outputs
end

end