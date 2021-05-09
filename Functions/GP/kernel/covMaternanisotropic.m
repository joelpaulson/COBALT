function K = covMaternanisotropic(d, hyp, sqrtK,expnK, dsq_M, x, z, i)
% Copyright (c) 2005-2017 Carl Edward Rasmussen & Hannes Nickisch. All rights reserved.
%
% Redistribution and use in source and binary forms, with or without modification,
% are permitted provided that the following conditions are met:
%    1. Redistributions of source code must retain the above copyright notice,
%       this list of conditions and the following disclaimer.
%    2. Redistributions in binary form must reproduce the above copyright notice,
%      this list of conditions and the following disclaimer in the documentation
%      and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY CARL EDWARD RASMUSSEN & HANNES NICKISCH ``AS IS''
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
% IN NO EVENT SHALL CARL EDWARD RASMUSSEN & HANNES NICKISCH OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
% ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
% EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
% The views and conclusions contained in the software and documentation
% are those of the authors and should not be interpreted as representing official policies,
% either expressed or implied, of Carl Edward Rasmussen & Hannes Nickisch.
%
% The code and associated documentation is available from http://gaussianprocess.org/gpml/code.</pre>

[n,D] = size(x);
sf2 = exp(2*hyp(D+1));

if nargin<7                                                        % covariances
    if      d == 3, t = sqrtK ; m =  (1 + t).*expnK;
    elseif  d == 1,             m =  expnK;
    elseif  d == 5, t = sqrtK ; m =  (1 + t.*(1+t/3)).*expnK;
    elseif  d == inf, m = expnK;
    end
    K = sf2*m;
else                                                               % derivatives
    if i<=D                                               % length scale parameter
        Ki = dsq_M(:,(i-1)*n+1:i*n) ;
        if     d == 3,             dm = expnK;
        elseif d == 1, t = sqrtK ; dm = (1./t).*expnK;
        elseif d == 5, t = sqrtK ; dm = ((1+t)/3).*expnK;
        elseif d == inf; dm = -1/2*expnK;
        end
        
        K = sf2*dm.*Ki;
        K(Ki<1e-12) = 0;                                    % fix limit case for d=1
    elseif i==D+1                                            % magnitude parameter
        if      d == 3, t = sqrtK ; m =  (1 + t).*expnK;
        elseif  d == 1,             m =  expnK;
        elseif  d == 5, t = sqrtK ; m =  (1 + t.*(1+t/3)).*expnK;
        elseif  d == inf,           m = expnK;
        end
        K = 2*sf2*m;
    end
end

end