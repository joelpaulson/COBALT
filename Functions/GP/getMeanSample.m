function [f,varf] = getMeanSample(Xnew,Ynew,Opt)
% Copyright (c) by Eric Bradford, Artur M. Schweidtmann and Alexei Lapkin, 2020-23-05.

% extration of variables from problem structure
nSpectralpoints = Opt.nSpectralpoints;
[n,D] = size(Xnew);
ell = exp(Opt.hyp.cov(1:D));
sf2 = exp(2*Opt.hyp.cov(D+1));
sn2 = exp(2*Opt.hyp.lik);

% Sampling of W and b
sW1  = lhsdesign(nSpectralpoints,D,'criterion','none');
sW2  = lhsdesign(nSpectralpoints,D,'criterion','none');
if Opt.cov ~= inf
    W = repmat(1./(ell)', nSpectralpoints, 1).*norminv(sW1).*sqrt(Opt.cov./chi2inv(sW2,Opt.cov));
else
    W = randn(nSpectralpoints,D) .* repmat(1./ell', nSpectralpoints, 1);
end

b = 2*pi*lhsdesign(nSpectralpoints,1,'criterion','none');

% Calculation of phi
phi = sqrt(2 * sf2 / nSpectralpoints) * cos(W * Xnew' + repmat(b, 1, n));

% Sampling of theta according to phi
A = phi * phi' + sn2 * eye(nSpectralpoints);

invA      = invChol(A);
mu_theta  = invA*phi*Ynew;

% Mean approximation (function) according to theta
phi_x = @(x) sqrt(2 * sf2 / nSpectralpoints) * cos(W * x' + repmat(b,1,size(x,1)));
f     = @(x) (mu_theta' * phi_x(x))';
varf  = @(x) sn2 + sn2*phi_x(x)'*invA*phi_x(x); 

end