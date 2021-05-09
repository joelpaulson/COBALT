function [f,varf] = getTrueMeanFunction(X, Y, Opt)

% Initialize variables

[n,D]       = size(X) ;
Opt.GP      = Opt ;
if Opt.GP.cov ~= inf
    d           = Opt.GP.cov;    % type of Martern
else
    d           = 1;
end
h1          = Opt.GP.h1 ;    % number of hyperparameters from covariance
h2          = Opt.GP.h2;     % number of hyperparameters from likelihood
hyp.cov     = Opt.GP.hyp.cov;
hyp.lik     = Opt.GP.hyp.lik;
ell         = exp(hyp.cov(1:D));
sf2         = exp(2*hyp.cov(D+1));
K           = zeros(n,n) ;

% Calculate covariance matrix
a = X' ;
K_M = zeros(n,n*D) ;
for i = 1:D
    K_M(:,(i-1)*n+1:i*n) = sqdist(a(i,:),a(i,:)) ;
end

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

% Calculate inverse of covariance matrix
try
    CH = chol(K) ;
    invK = CH\(CH'\eye(n));
catch
    CH = chol(K+eye(n)*1e-4);
    invK = CH\(CH'\eye(n));
    warning('Covariance matrix in Nlikelihood is not positive semi-definite')
end

% calculate covariance between training and test set
Ktetr = @(x)(evaluateKernel(x, X, Opt));
Ktete = @(x)(evaluateKernel(x, x, Opt));

% calculate mean and covariance
alpha = CH\(CH'\Y);
f = @(x)(Ktetr(x)*alpha);
varf = @(x)(Ktete(x) - Ktetr(x)*invK*(Ktetr(x))');

end