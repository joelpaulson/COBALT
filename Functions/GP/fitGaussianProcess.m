function [OptGPhyp] = fitGaussianProcess(Xnew,Ynew,OptGP)
% Copyright (c) by Eric Bradford, Artur M. Schweidtmann and Alexei Lapkin, 2017-13-12.

% Function which minimizes the neg-loglikelihood to find hyperparameters
%% Initialize variables
Opt.GP = OptGP;
[n,D] = size(Xnew) ;

%% Set initial hyperparameters
h1 = Opt.GP.h1; % number of hyperparameters from covariance
h2 = Opt.GP.h2; % number of hyperparameters from likelihood

%% Calculation of squared-distance matrix
a = Xnew' ;
K_M = zeros(n,n*D) ;
for i = 1:D
    K_M(:,(i-1)*n+1:i*n) = sqdist(a(i,:),a(i,:)) ;
end

%% Minimize log-negative likeliehood
% Objective Function
obj_fun.f = @(hypVar)NLikelihood(hypVar,Xnew,Ynew,K_M,Opt.GP);

% Define bounds
lb              = ones(h1+h2,1) *  log(sqrt(10^(-3))) ;  % see Jones paper
ub              = ones(h1+h2,1) *  log(sqrt(10^(3)))  ;  % see Jones paper
lb(h1+h2)    = -6;
ub(h1+h2)    = Opt.GP.noiselimit;
bounds = [lb,ub];
opts.maxevals = Opt.GP.fun_eval*(h1+h2);
opts.maxits =  100000*(h1+h2);
opts.maxdeep = 100000*(h1+h2);
opts.showits = 0;

% Defintion of options for global search
[~,x0] = Direct(obj_fun,bounds,opts);

% Defintion of options for fmincon solver
LSoptions.Algorithm = 'interior-point';
LSoptions.DerivativeCheck = 'off';
LSoptions.TolCon = 1e-12;
LSoptions.Display = 'off';
LSoptions.Hessian = 'bfgs';
LSoptions.TolFun = 1e-12;
LSoptions.PlotFcns = [];
LSoptions.GradConstr = 'off';
LSoptions.GradObj = 'on';
LSoptions.TolX = 1e-14;
LSoptions.UseParallel = 0;

% Solve optimization problem
hypResult = fmincon(obj_fun.f,x0,[],[],[],[],lb,ub,[],LSoptions);

%% Return optimal hyperparameters
Opt.GP.hyp.cov  = hypResult(1:h1);
Opt.GP.hyp.lik  = hypResult(h1+1:h1+h2);
OptGPhyp = Opt.GP.hyp ;

end