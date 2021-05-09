
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Examples from the paper titled "COBALT: COnstrained Bayesian optimizAtion of %%%
%%% computationaLly expensive grey-box models exploiting derivaTive information  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear all currently defined variables
clear

% fix random seed
rng(100, 'twister')

% select problem example
% unconstrained examples: 'GoldsteinPrice', 'Rastrigin3D', 'Rosenbrock6D'
% constrained examples: 'ToyHydrology', 'RosenSuzuki', 'Colville'
problem = "RosenSuzuki";

% maximum number of iterations
Nmax = 20;

% number of monte carlo runs for optimizer
Nmc = 3;

% what optimizer? pick 1 or more: 'ei', 'pi', 'random', 'eicf', 'cobalt'
optimizer_list = {'cobalt', 'ei'};

% define problem
% min_{x,y,z} f(x,y)
%    s.t.     g(x,y) <= 0
%             y = d(z)
%             z = A*x
switch problem
    case "GoldsteinPrice"
        % problem objective and constraints
        nx = 2;
        ny = 2;
        A = [1,0;0 1];
        g = @(x,y)([]);
        f = @(x, y)((1 +(x(1)+x(2)+1).^2.*(19-14.*x(1)+3.*x(1).^2+y(2))) * (30 + y(1)*(18-32*x(1)+12*x(1).^2+48*x(2)-36*x(1).*x(2)+27*x(2).^2)));
        d = @(z)([(2*z(1)-3*z(2)).^2 ; -14.*z(2)+6.*z(1).*z(2)+3.*z(2).^2]);
        
        % lower and upper bounds
        x_min = [-2 ; -2];
        x_max = [ 2 ;  2];
        
        % initial guess
        x0 = [0,-1]';
        
    case "Rastrigin3D"
        % problem objective and constraints
        Acoeff = 10;
        f = @(x,y)(3*Acoeff + x(1).^2 - Acoeff*cos(2*pi*x(1)) + 3*Acoeff + x(2).^2 - Acoeff*cos(2*pi*x(2)) + y(2));
        d = @(z)([z.^2 - Acoeff*cos(2*pi*z)]);
        g = @(x,y)([]);
        nx = 3;
        ny = 2;
        A = [0, 0, 1];
        
        % lower and upper bounds
        x_min = [-5.12 ; -5.12 ; -5.12];
        x_max = [5.12 ; 5.12 ; 5.12];
        
        % initial guess
        x0 = [0,0,0];
        
    case "Rosenbrock6D"
        % problem objective and constraints
        f = @(x,y)(100*(y(1))^2 + (1 - x(1))^2 +...
                   100*(y(2))^2 + (1 - x(2))^2 +...
                   100*(y(3))^2 + (1 - x(3))^2 +...
                   100*(x(5) - x(4)^2)^2 + y(4) +...
                   100*(x(6) - x(5)^2)^2 + (1 - x(5))^2);
        d = @(z)([z(2) - z(1)^2;...
                  z(3) - z(2)^2;...
                  z(4) - z(3)^2;...
                  (1 - z(4))^2]);
        g = @(x,y)([]);
        nx = 6;
        ny = 4;
        A = [1 0 0 0 0 0;0 1 0 0 0 0; 0 0 1 0 0 0; 0 0 0 1 0 0];
        
        % bounds
        x_min = -2*ones(6,1);
        x_max = 2*ones(6,1);
        
        % initial guess
        x0 = ones(6,1);
        
    case 'ToyHydrology'
        % problem objective and constraints
        f = @(x,y)(x(1) + x(2));
        d = @(z)(2*pi*z(1).^2);
        g = @(x,y)([3/2 - x(1) - 2*x(2) - 1/2*sin(-2*pi*(2*x(2)) + y(1)) ; x(1).^2 + x(2).^2 - 3/2]);
        nx = 2;
        ny = 1;
        A = [1, 0];
        
        % lower and upper bounds
        x_min = [0 ; 0];
        x_max = [1 ; 1];
        
        % initial guess
        x0 = [0.5 ; 0.5];
        
    case 'RosenSuzuki'
        % problem objective and constraints
        f = @(x,y)(x(1)^2 + x(2)^2 + x(4)^2 - 5*x(1) - 5*x(2) + y(1));
        d = @(z)([2*z(1)^2 - 21*z(1) + 7*z(2) ; z(1)^2 + 2*z(2)^2]);
        g = @(x,y)([-(8 - x(1)^2 - x(2)^2 - x(3)^2 - x(4)^2 - x(1) + x(2) - x(3) + x(4)) ; -(10 - x(1)^2 - 2*x(2)^2 - y(2) + x(1) + x(4)) ; -(5 - 2*x(1)^2 - x(2)^2 - x(3)^2 - 2*x(1) + x(2) + x(4))]);
        nx = 4;
        ny = 2;
        A = [0, 0, 1, 0 ; 0, 0, 0, 1];
 
        % lower and upper bounds
        x_min = [-2 ; -2 ; -2 ; -2];
        x_max = [ 2 ;  2 ;  2 ;  2];
        
        % initial guess
        x0 = [0 ; 0 ; 0 ; 0];
        
    case "Colville"
        % problem objective and constraints
        f = @(x,y)(5.3578*x(3)^2 + y(1));
        d = @(z)([0.8357*z(1)*z(4) + 37.2392*z(1);...
                  0.00002584*z(3)*z(4) - 0.00006663*z(2)*z(4);...
                  2275.1327/(z(3)*z(4))- 0.2668*z(1)/z(4);...
                  1330.3294/(z(2)*z(4)) - 0.42*z(1)/z(4)]);
        g = @(x,y)([y(2) - 0.0000734*x(1)*x(4) - 1;...
                    0.000853007*x(2)*x(5) + 0.00009395*x(1)*x(4) - 0.00033085*x(3)*x(5) - 1;...
                    y(4) - 0.30586*x(3)^2/(x(2)*x(5)) - 1;...
                    0.00024186*x(2)*x(5) + 0.00010159*x(1)*x(2) + 0.00007379*x(3)^2 - 1;...
                    y(3) - 0.40584*x(4)/x(5) - 1;...
                    0.00029955*x(3)*x(5) + 0.00007992*x(1)*x(3) + 0.00012157*x(3)*x(4) - 1]);
        nx = 5;
        ny = 4;
        A = [1 0 0 0 0;0 1 0 0 0;0 0 1 0 0; 0 0 0 0 1];
        
        % lower and upper bounds
        x_min = [78;33;27;27;27];
        x_max = [102;45;45;45;45];
        
        % initial guess
        x0 = [78;33;29.998;45;36.7673];
end

% add required
addpath('../Functions/GP')
addpath('../Functions/COBALT')
addpath('../Functions/GP/kernel')

% solve exact [only possible if function d(.) is known and
% implementable in casadi]
solve_exact = 1;

% number of initial points
nz = size(A,1);
Ninit = max(3, nz+1);

% loop over number of optimizers
Fmin_list = cell(length(optimizer_list),1);
for j = 1:length(optimizer_list)
    % create empty array
    Fmin = zeros(Nmax,Nmc);
    
    % run algorithm for the Monte Carlo runs
    for i = 1:Nmc
        
        % create new grey box class instance
        gb = GreyboxModel(f, d, g, nx, ny, A, x_min, x_max, x0, Ninit, Nmax);
        
        % decide which optimizer to call
        switch optimizer_list{j}
            case 'cobalt'
                gb.optimize();
            case 'ei'
                gb.optimize_blackbox();
            case 'pi'
                gb.optimize_blackbox_pi();
            case 'random'
                gb.Ninit = Nmax;
                gb.optimize_blackbox();
            case 'eicf'
                gb.optimize_eicf();
        end
        Fmin(:,i) = gb.Fmin;
    end
    
    % store data
    Fmin_list{j} = Fmin;
end

% optimize true function
if solve_exact == 1
    [x_best, f_best] = gb.solve_exact();
else
    f_best = min(min([Fmin_list{:}]));
end

% plot regret over iteration
Regret_list = cell(length(optimizer_list),1);
for j = 1:length(optimizer_list)
    Fmin = Fmin_list{j};
    Regret = max(Fmin - f_best, 1e-5); % set a floor based on tolerances
    Regret(isnan(Fmin)) = nan;
    Regret = Regret';
    Regret_list{j} = Regret;
end
figure;
hold on;
color_list = {'-b', '-r', '-g', '-k','-m', '-c', '-y'};
h_list = [];
log_list = Regret_list;
for j = 1:length(optimizer_list)
    log_list{j} = log10(Regret_list{j});
    h = stairs(1:Nmax, mean(log_list{j}(:,1:Nmax)), color_list{j}, 'linewidth', 4, 'DisplayName', optimizer_list{j});
    errorbar(1:Nmax, mean(log_list{j}(:,1:Nmax)), 1.96*std(log_list{j}(:,1:Nmax))/sqrt(Nmc), color_list{j}, 'CapSize', 10, 'linewidth', 1, 'linestyle', 'none')
    h_list = [h_list ; h];
end
leg = legend(h_list);
xlabel('iteration')
ylabel('log10(regret)')
set(gcf,'color','w');
set(gca,'FontSize',20)
set(leg,'color','none');
xlim([1 Nmax])
grid on

% remove added folders from path
rmpath('../Functions/GP')
rmpath('../Functions/GP/kernel')
rmpath('../Functions/COBALT')
