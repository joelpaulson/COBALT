function Opt = setOptionStructure(Opt,X,Y)
% Copyright (c) by Eric Bradford, Artur M. Schweidtmann and Alexei Lapkin, 2017-13-12.

%% Extraction of input and output dimensions from X and Y
Opt.Gen.NoOfGPs = size(Y,2);           % Number of GPs to be trained
Opt.Gen.NoOfInputDim = size(X,2);      % Number of inputs dimensions
for i = 1 : Opt.Gen.NoOfGPs
    %% Set up GP options
    Opt.GP(i).cov = Opt.GP(i).matern;      % Matern type 1 / 3 / 5 / inf
    
    %% Set hyperpriors (MAP)
    Opt.GP(i).noiselimit = 0;                      % Upper bound on noise
    Opt.GP(i).var        = 10;                     % Upper bound on signal variance
    Opt.GP(i).h1         = Opt.Gen.NoOfInputDim+1; % Number of hyperparameters from covariance
    Opt.GP(i).h2         = 1;                      % Number of hyperparameters from likelihood
    
    %% priorGauss (mean, var)
    Opt.GP(i).priorlik  = [-6 ,Opt.GP(i).var];
    Opt.GP(i).priorcov  = [ 0 ,Opt.GP(i).var];
    
    %% Initial values for hyperparameters
    Opt.GP(i).hyp.cov       = zeros(1, Opt.GP(i).h1);
    Opt.GP(i).hyp.lik       = log(1e-2);
end

end