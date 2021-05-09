function [NLL,dNLL] = NLikelihood(hypVar, Xnew, Ynew, K_M, OptGP)
% Copyright (c) by Eric Bradford, Artur M. Schweidtmann and Alexei Lapkin, 2017-13-12.

% Calculates the log-negative likelihood
%% Initialize variables
[n,D]       = size(Xnew) ;
Opt.GP      = OptGP ;
if Opt.GP.cov ~= inf
    d           = Opt.GP.cov;    % type of Martern
else
    d           = 1;
end
h1          = Opt.GP.h1 ;    % number of hyperparameters from covariance
h2          = Opt.GP.h2;     % number of hyperparameters from likelihood
hyp.cov     = hypVar(1:h1);
hyp.lik     = hypVar(h1+1:h1+h2);
ell         = exp(hyp.cov(1:D));
sf2         = exp(2*hypVar(D+1));
K           = zeros(n,n) ;

%% Calculate covariance matrix
for i = 1:D
    K = K_M(:,(i-1)*n+1:i*n) * d/ell(i)^2 + K;
end

if Opt.GP.cov ~= inf
    sqrtK = sqrt(K) ;
    expnK = exp(-sqrtK) ;
else
    expnK = exp(-1/2*K);
    sqrtK = [];
end

if      Opt.GP.cov == 3, t = sqrtK ; m =  (1 + t).*expnK;
elseif  Opt.GP.cov == 1,             m =  expnK;
elseif  Opt.GP.cov == 5, t = sqrtK ; m =  (1 + t.*(1+t/3)).*expnK;
elseif  Opt.GP.cov == inf,           m  = expnK;
end
K = sf2*m;
K =  K + eye(n)*exp(hyp.lik*2) ;
K = (K+K')/2 ; % This guarantees a symmetric matrix

%% Calculate inverse of covariance matrix
try
    CH = chol(K) ;
    invK = CH\(CH'\eye(n));
catch
    CH = chol(K+eye(n)*1e-4);
    invK = CH\(CH'\eye(n));
    warning('Covariance matrix in Nlikelihood is not positive semi-definite')
end

%% Calculate determinant of covariance matrix
logDetK = 2*sum(log(abs(diag(CH)))) ;

%% Calculate hyperperpriors
logprior = 0 ;
dlogpriorcov = zeros(1,h1) ;
for i = 1 : h1
    [A, dlogpriorcov(i)] =  priorGauss(Opt.GP.priorcov(1),Opt.GP.priorcov(2), hyp.cov(i) ) ;
    logprior = logprior + A ;
end

dlogpriorlik = zeros(1,h2);
for i = 1 : h2
    [A, dlogpriorlik(i)] =  priorGauss(Opt.GP.priorlik(1),Opt.GP.priorlik(2), hyp.lik(i) ) ;
    logprior = logprior + A ;
end

%% Calculate negative log-likeliehood
NLL = n/2*log(2*pi) + 1/2 * logDetK + 1/2 * Ynew'*invK*Ynew - logprior ;

%% Gradient calculation
if nargout == 2 % do only if No of output variable is 2 (if necessary)
    dsq_M = zeros(n,n*D) ;
    for i = 1 : D
        dsq_M(:,(i-1)*n+1:i*n)  = K_M(:,(i-1)*n+1:i*n) * (d)/ell(i)^2 ;
    end
    
    c = invK*Ynew;
    for i = 1 : h1
        dK = covMaternanisotropic(Opt.GP.cov,hyp.cov, sqrtK, expnK, dsq_M, Xnew, [], i);
        b = invK* dK ;
        dNLL_f.cov(i) = 1/2*trace(b) - 1/2*Ynew'*b*c ;
    end
    
    for i = 1 : h2
        dK = 2 * exp(hyp.lik(i)) * eye(n) * exp(hyp.lik(i));
        b = invK* dK ;
        dNLL_f.lik(i) = 1/2*trace(b) - 1/2*Ynew'*b*c ;
    end
    
    dNLL = [dNLL_f.cov';dNLL_f.lik'] - [dlogpriorcov';dlogpriorlik'];
end

end