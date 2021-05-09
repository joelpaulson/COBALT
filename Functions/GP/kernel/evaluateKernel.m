function K = evaluateKernel(X, Y, OptGP)

Opt.GP = OptGP;

[n1,D]       = size(X);
n = size(Y,1);
hyp.cov     = Opt.GP.hyp.cov;
hyp.lik     = Opt.GP.hyp.lik;
ell         = exp(hyp.cov(1:D));
sf2         = exp(2*hyp.cov(D+1));
K           = zeros(n1,n) ;

if Opt.GP.cov ~= inf
    d           = Opt.GP.cov;    % type of Martern
else
    d           = 1;
end

a = X';
b = Y';
% K_M = zeros(n1,n*D) ;
% for i = 1:D
%     K_M(:,(i-1)*n+1:i*n) = sqdist(a(i,:),b(i,:)) ;
% end
% 
% for i = 1:D
%     K = K_M(:,(i-1)*n+1:i*n) * d/ell(i)^2 + K;
% end

for i = 1:D
    K = distSquaredGP(a(i,:)',b(i,:)') * d/ell(i)^2 + K;
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

end