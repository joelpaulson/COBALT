classdef GreyboxModel < matlab.mixin.Copyable
   properties
       % functions
       f function_handle = @(z,w,y)([])
       d function_handle = @(w)([])
       g function_handle = @(z,w,y)([])
       
       % casadi function names
       f_ca
       d_ca
       g_ca
       x_ca
       z_ca
       y_ca
       conversion_ca
       
       % dimensions
       nx(1,1) {mustBeInteger, mustBeNonnegative} = 0
       nz(1,1) {mustBeInteger, mustBeNonnegative} = 0
       ny(1,1) {mustBeInteger, mustBeNonnegative} = 0
       ng(1,1) {mustBeInteger, mustBeNonnegative} = 0
       
       % matrix connecting
       A
       
       % bounds
       x_min
       x_max
       z_min
       z_max
       
       % initial guess for optimizers (if needed)
       x0
       
       % list of data
       X
       Z
       Y
       F
       G
       Incumbent
       S
       timeIter
       Fmin
       
       % exact solution (if possible to obtain)
       x_exact
       f_exact
       conv_exact
       exact_solver_used
       
       % gaussian process properties
       optgp
       dmean_ca
       dvar_ca
       
       % COBALT parameters
       Nmc_eic(1,1) {mustBeInteger, mustBeNonnegative} = 100   % number of MC samples for expected improvement composite (EIC)
       Ninit(1,1) {mustBeInteger, mustBeNonnegative} = 5
       Nmax(1,1) {mustBeInteger, mustBeNonnegative} = 30
       tolG = 1e-3
       
   end
   methods
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Object definition, with defaults specified
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function obj = GreyboxModel(varargin)           
           % first argument is objective f(x,y)
           if nargin > 0
               obj.f = varargin{1};
           end
           % second argument is black-box function y=d(z)
           if nargin > 1
               obj.d = varargin{2};
           end
           % third argument is inequality g(x,y)<=0
           if nargin > 2 && ~isempty(varargin{3})
               obj.g = varargin{3};
           end
           % fourth argument is number of variables
           if nargin > 3
               obj.nx = varargin{4};
           end
           % fifth argument is number of black-box outputs
           if nargin > 4
               obj.ny = varargin{5};
           end
           % sixth argument is matrix A where z = A*x
           if nargin > 5
               obj.A = varargin{6};
           end
           obj.nz = size(obj.A,1);

           % set default bound values
           x_min = zeros(obj.nx,1);
           x_max = ones(obj.nx,1);
           
           % seventh argument is lower bound on white-box variables
           if nargin > 6
               x_min = varargin{7};
           end
           % eight argument is upper bound on white-box variables
           if nargin > 7
               x_max = varargin{8};
           end
           
           % calculate bounds on z
           z_min = obj.A*x_min;
           z_max = obj.A*x_max;

           obj.x_min = x_min;
           obj.x_max = x_max;
           obj.z_min = z_min;
           obj.z_max = z_max;

           % set default initial guess values
           x0 = zeros(obj.nx,1);
           
           % ninth argument is initial guess for variables
           if nargin > 8
               x0 = varargin{9};
           end
           obj.x0 = x0;

           % tenth argument is Ninit
           if nargin > 9
               Ninit = varargin{10};
           end
           obj.Ninit = Ninit;           

           % eleventh argument is Nmax
           if nargin > 10
               Nmax = varargin{11};
           end
           obj.Nmax = Nmax;                      
           
           % set size of constraints automatically
           obj.ng = length(obj.g(zeros(obj.nx,1), zeros(obj.ny,1)));
           
           % convert functions to casadi function (if possible)
           obj.convert_to_casadi()
       end

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Converts functions specified in terms of Matlab functions into
       %%% casadi objects, which can be used by gradient-based solvers 
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
       function convert_to_casadi(obj)
           % import casadi
           import casadi.*
           
           % create casadi variables
           obj.x_ca = SX.sym('x_ca',obj.nx);
           obj.z_ca = SX.sym('z_ca',obj.nz);
           obj.y_ca = SX.sym('y_ca',obj.ny);
           
           % try to evaluate matlab functions, otherwise flag as not usable
           % by casadi
           counter = 0;
           try 
               obj.f_ca = Function('f_ca', {obj.x_ca, obj.y_ca}, {obj.f(obj.x_ca, obj.y_ca)});
           catch
               counter = counter + 1;
           end
           try 
               obj.d_ca = Function('d_ca', {obj.z_ca}, {obj.d(obj.z_ca)});
           catch
               counter = counter + 1;
           end
           if obj.ng > 0
               try
                   obj.g_ca = Function('g_ca', {obj.x_ca, obj.y_ca}, {obj.g(obj.x_ca, obj.y_ca)});
               catch
                   counter = counter + 1;
               end               
           end
           if counter == 0
               obj.conversion_ca = 1;
           else
               obj.conversion_ca = 0;
           end      
       end

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Solve the exact problem, if possible
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                     
       function [x_opt, f_opt] = solve_exact(obj, varargin)
           if obj.conversion_ca == 0
               fprintf('CANNOT WTIH CASADI!\n')
           else
               import casadi.*
               opti = casadi.Opti();
               X = opti.variable(obj.nx);
               opti.set_initial(X, obj.x0);
               opti.minimize( obj.f_ca(X,obj.d(obj.A*X)) )
               if obj.ng > 0
                   opti.subject_to( obj.g_ca(X,obj.d_ca(obj.A*X)) <= 0 )
               end
               opti.subject_to( obj.x_min <= X <= obj.x_max )
               p_opts = struct('expand',true,'print_time',0);
               s_opts = struct('max_iter',1000,'tol',1e-8);
               s_opts.print_level = 0;
               opti.solver('ipopt',p_opts,s_opts);
               sol = opti.solve();
               x_opt = sol.value(X);
               f_opt = sol.value(opti.f);
           end
       end
       
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Run the black-box algorithm bayesopt
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                     
       function optimize_blackbox(obj, varargin)
           
           % TODO: override defaults with variable input arguments
           % ONLY allows real numbers for now
           
           % Bayesian optimization variables
           xbo = [];
           for i = 1:obj.nx
               xbo_i = optimizableVariable(['x' num2str(i)],[obj.x_min(i),obj.x_max(i)],'Type','real');
               xbo = [xbo ; xbo_i];
           end
           
           % objective function and constraint handle
           fun = @(x)obj.blackBoxFunction(x, obj.f, obj.g, obj.d, obj.A, obj.ng);
           
           % call Bayesian optimization solver
           results = bayesopt(fun,xbo,...
               'AcquisitionFunctionName', 'expected-improvement-plus',...
               'IsObjectiveDeterministic', 1,...
               'ExplorationRatio', 0.5,...
               'GPActiveSetSize', 300,...
               'UseParallel', false,...
               'MaxObjectiveEvaluations', obj.Nmax,...
               'NumSeedPoints', obj.Ninit,...
               'NumCoupledConstraints',obj.ng,...
               'PlotFcn',{});
           
           % store information
           obj.X = table2array(results.XTrace);
           obj.F = results.ObjectiveTrace;
           obj.G = results.ConstraintsTrace;
           obj.Fmin = results.ObjectiveMinimumTrace;           
       end 
       
       % bayesopt with PI acquisition
       function optimize_blackbox_pi(obj, varargin)
           
           % TODO: override defaults with variable input arguments
           % ONLY allows real numbers for now
           
           % Bayesian optimization variables
           xbo = [];
           for i = 1:obj.nx
               xbo_i = optimizableVariable(['x' num2str(i)],[obj.x_min(i),obj.x_max(i)],'Type','real');
               xbo = [xbo ; xbo_i];
           end
           
           % objective function and constraint handle
           fun = @(x)obj.blackBoxFunction(x, obj.f, obj.g, obj.d, obj.A, obj.ng);
           
           % call Bayesian optimization solver
           results = bayesopt(fun,xbo,...
               'AcquisitionFunctionName', 'probability-of-improvement',...
               'IsObjectiveDeterministic', 1,...
               'ExplorationRatio', 0.5,...
               'GPActiveSetSize', 300,...
               'UseParallel', false,...
               'MaxObjectiveEvaluations', obj.Nmax,...
               'NumSeedPoints', obj.Ninit,...
               'NumCoupledConstraints',obj.ng,...
               'PlotFcn',{});
           
           % store information
           obj.X = table2array(results.XTrace);
           obj.F = results.ObjectiveTrace;
           obj.G = results.ConstraintsTrace;
           obj.Fmin = results.ObjectiveMinimumTrace;           
       end 
       
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Run the proposed COBALT algorithm 
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%              
       function optimize(obj, varargin)

           % TODO: override defaults with variable input arguments           
           
           % extract parameters from objective
           nx = obj.nx;
           ng = obj.ng;
           ny = obj.ny;
           nz = obj.nz;
           x_min = obj.x_min;
           x_max = obj.x_max;
           z_min = obj.z_min;
           z_max = obj.z_max;
           tolG = obj.tolG;
           Ninit = obj.Ninit;
           Nmax = obj.Nmax;
           A = obj.A;
           f = obj.f;
           d = obj.d;
           g = obj.g;
           Nmc_eic = obj.Nmc_eic;
           x = obj.x_ca;
           z = obj.z_ca;
           y = obj.y_ca;

           % import casadi
           import casadi.*

           % tolerance, iteration count, and printlevel for ipopt
           ipopt_tol = 1e-8;
           ipopt_maxiter = 1000;
           ipopt_printlevel = 0;
           
           % generate initial dataset
           fprintf('Random sampling first %g evaluations...', Ninit)
           X = repmat((x_max-x_min)', [Ninit, 1]).*lhsdesign(Ninit,nx) + repmat(x_min', [Ninit, 1]);
           Z = X*A';
           Y = zeros(Ninit,ny);
           F = zeros(Ninit,1);
           G = zeros(Ninit,ng);
           for k = 1:size(Z,1)
               Y(k,:) = d(Z(k,:)); % evaluate black-box function
               F(k) = f(X(k,:), Y(k,:));
               if ng > 0
                   G(k,:) = g(X(k,:), Y(k,:));
               end
           end
           fprintf('done\n')
                                 
           % loop over maximum iteration
           Nadd = Nmax-Ninit;
           timeIter = zeros(Nadd,1);
           Incumbent = nan*ones(Nadd,1);
           S = nan*ones(Nadd,1);
           for i = 1:Nadd
               % print statement about iteration
               fprintf('Starting iteration %g of %g...', i+Ninit, Nmax)
               
               % record time
               tic;
               
               % fit Gaussian process models
               obj.fit_gaussian_process(Z, Y)
               dmean_func = obj.dmean_ca;
               dvar_func = obj.dvar_ca;
               
               % compute the incumbent
               if ng > 0
                   index_feas = sum(G > tolG, 2); % first find indices of feasible points
                   index_feas = find(index_feas == 0);
               else
                   index_feas = (1:length(F))';
               end
               if sum(index_feas) == 0
                   incumbent = nan; % does not exist since no feasible points
               else
                   incumbent = min(F(index_feas)); % take the incumbent as minimum of feasible
               end
               Incumbent(i) = incumbent;
               
               % approximate the mean objective function using linearization
               ymean = dmean_func(A*x);
               fmean_func = Function('fmean_func', {x}, {obj.f_ca(x, ymean)});
               
               % calculate mean and variance of constraints using linearization
               if ng > 0
                   gmean_func = Function('gmean_func', {x}, {obj.g_ca(x, ymean)});
                   
                   var_g = [];
                   for j = 1:ng
                       gj = g(x,y);
                       gj = gj(j);
                       gj_func = Function('gj_func', {x, y}, {gj});
                       J_gj_y = Function(['J_g' num2str(j) '_y'], {x, y}, {jacobian(gj, y)});
                       var_gj = J_gj_y(x, dmean_func(A*x)) * diag(dvar_func(A*x)) * J_gj_y(x, dmean_func(A*x))';
                       var_g = vertcat(var_g, var_gj);
                   end
                   gvar_func = Function('gvar_func', {x}, {var_g});
               end
               
               % define nonlinear constraints
               taug = -(3-3/Nadd*i);   % taug = norminv(1-0.9);
               if ng > 0
                   nonlincon = @(x)(obj.my_nonlincon(x, gmean_func, gvar_func, taug, ng));
               end
               
               % if incumbent does not exist, we set S=0 and thus only use mean-based
               % search
               if isnan(incumbent)
                   % specify acquisition function
                   s = 0;
                   acquisition_function = @(x)(-reshape(full(fmean_func(x')),[size(x,1),1]));
                   
                   % otherwise we need to include EI and mean in objective
               else
                   % draw random samples to be used for SAA
                   eta = randn(Nmc_eic,ny);

                   % construct the EIC function using MC sampling
                   EICF_samp = Function('EICF_samp', {x}, {max(incumbent - (obj.f_ca(x,dmean_func(A*x)+sqrt(dvar_func(A*x)).*eta')), 0)});
                   expected_improvement_composite = @(x)(mean(reshape(full(EICF_samp(x')),[length(eta),size(x,1)]),1)');

                   % calculate scaling factor
                   X_lhs = repmat((x_max-x_min)', [100*nx, 1]).*lhsdesign(100*nx,nx) + repmat(x_min', [100*nx, 1]);
                   if ng > 0
                       G_lhs = nonlincon(X_lhs);
                       index_lhs_feas = sum(G_lhs > tolG, 2); % first find indices of feasible points
                       index_lhs_feas = find(index_lhs_feas == 0);
                   else
                       index_lhs_feas = (1:size(X_lhs,1))';
                   end
                   if sum(index_lhs_feas) == 0
                       EI_lhs_max = 0; % set to zero if we did not find any feasible points
                   else
                       X_lhs = X_lhs(index_lhs_feas,:);
                       EI_lhs = expected_improvement_composite(X_lhs);
                       [EI_lhs_max,index_lhs] = max(EI_lhs);
                   end
                   if EI_lhs_max <= 1e-1
                       s = 1;
                   else
                       x_EI_lhs_max = X_lhs(index_lhs,:);
                       s = 100 * abs(full(fmean_func(x_EI_lhs_max))) / EI_lhs_max;
                   end
                   
                   % specify acquisition function
                   acquisition_function = @(x)(s * expected_improvement_composite(x) - reshape(full(fmean_func(x')),[size(x,1),1]));
               end
               S(i) = s; % store s value
               
               % optimize using ga if specified
               fitness = @(x)(-acquisition_function(x));
               if 0
                   % optimize using genetic algorithm
                   if ng > 0
                       [x_ga, f_ga] = ga(fitness, nx, [], [], [], [], x_min', x_max', nonlincon, ga_opts);
                   else
                       [x_ga, f_ga] = ga(fitness, nx, [], [], [], [], x_min', x_max', [], ga_opts);
                   end
                   
                   % otherwise optimize using random evaluation + then ipopt
               else
                   X_lhs = repmat((x_max-x_min)', [10000*nx, 1]).*lhsdesign(10000*nx,nx) + repmat(x_min', [10000*nx, 1]);
                   if ng > 0
                       G_lhs = nonlincon(X_lhs);
                       index_lhs_feas = sum(G_lhs > tolG, 2); % first find indices of feasible points
                       index_lhs_feas = find(index_lhs_feas == 0);
                   else
                       index_lhs_feas = (1:size(X_lhs,1))';
                       fitness_0 = 1e6;
                   end
                   if sum(index_lhs_feas) == 0
                       Fitness_lhs = fitness(X_lhs);
                       [~,index_lhs] = min(Fitness_lhs + 100*max([G_lhs,zeros(size(G_lhs,1),1)], [], 2).^2);
                   else
                       X_lhs = X_lhs(index_lhs_feas,:);
                       Fitness_lhs = fitness(X_lhs);
                       [fitness_0, index_lhs] = min(Fitness_lhs);
                   end
                   x0_fitness = X_lhs(index_lhs,:);
                   
                   opti_wb2s = casadi.Opti();
                   X_wb2s = opti_wb2s.variable(nx);
                   opti_wb2s.set_initial(X_wb2s, x0_fitness);
                   opti_wb2s.minimize( fitness(X_wb2s') )
                   if ng > 0
                       opti_wb2s.subject_to( nonlincon(X_wb2s') <= 0 )
                   end
                   opti_wb2s.subject_to( x_min <= X_wb2s <= x_max )
                   p_opts = struct('expand',true,'print_time',0);
                   s_opts = struct('max_iter',ipopt_maxiter,'tol',ipopt_tol);
                   s_opts.print_level = ipopt_printlevel;
                   opti_wb2s.solver('ipopt',p_opts,s_opts);
                   try
                       sol_wb2s = opti_wb2s.solve();
                       f_ga = sol_wb2s.value(opti_wb2s.f);
                       if f_ga < fitness_0
                           x_ga = sol_wb2s.value(X_wb2s)';
                       else
                           x_ga = x0_fitness;
                       end
                   catch
                       f_ga = opti_wb2s.debug.value(opti_wb2s.f);
                       if f_ga < fitness_0
                           x_ga = opti_wb2s.debug.value(X_wb2s)';
                       else
                           x_ga = x0_fitness;
                       end
                   end
               end
               
               % add data to X and then scale inputs to prepare for black-box
               % evluation
               X = [X ; x_ga];
               xnewtrue = x_ga;
               znewtrue = x_ga*A';
               for j = 1:nz
                   zNew(:,j) = (znewtrue(:,j) - z_min(j)) / (z_max(j) - z_min(j));
               end
               
               % evaluate black-box function at new inputs
               for l = 1:size(znewtrue,1)
                   ytrue(l,:) = d(znewtrue(l,:));
                   ftrue(l) = f(xnewtrue, ytrue(l,:));
                   if ng > 0
                       gtrue(l,:) = g(xnewtrue, ytrue(l,:));
                   end
               end
               Z = [Z ; znewtrue];
               Y = [Y ; ytrue];
               F = [F ; ftrue];
               if ng > 0
                   G = [G ; gtrue];
               end
                              
               % record time for each iteration
               timeIter(i) = toc;
               
               % print end statement
               fprintf('took %g seconds...incumbent = %g...scaling = %g\n', timeIter(i), Incumbent(i), S(i))
           end
           
           % compute the minimum feasible solution at every iteration
           Fmin = nan*ones(Nmax,1);
           for i = 1:Nmax
               if ng > 0
                   if sum(G(i,:) > tolG) == 0
                       if i == 1
                           Fmin(i) = F(i);
                       else
                           Fmin(i) = min(F(i),Fmin(i-1));
                       end
                   else
                       if i > 1
                           Fmin(i) = Fmin(i-1);
                       end
                   end
               else
                   if i == 1
                       Fmin(i) = F(i);
                   else
                       Fmin(i) = min(F(i),Fmin(i-1));
                   end
               end
           end
           
           % store information
           obj.X = X;
           obj.Z = Z;
           obj.Y = Y;
           obj.F = F;
           obj.G = G;
           obj.Incumbent = Incumbent;
           obj.S = S;
           obj.timeIter = timeIter;
           obj.Fmin = Fmin;
       end
       
       % COBALT with eicf acquisition
       
       function optimize_eicf(obj, varargin)

           % TODO: override defaults with variable input arguments           
           
           % extract parameters from objective
           nx = obj.nx;
           ng = obj.ng;
           ny = obj.ny;
           nz = obj.nz;
           x_min = obj.x_min;
           x_max = obj.x_max;
           z_min = obj.z_min;
           z_max = obj.z_max;
           tolG = obj.tolG;
           Ninit = obj.Ninit;
           Nmax = obj.Nmax;
           A = obj.A;
           f = obj.f;
           d = obj.d;
           g = obj.g;
           Nmc_eic = obj.Nmc_eic;
           x = obj.x_ca;
           z = obj.z_ca;
           y = obj.y_ca;

           % import casadi
           import casadi.*

           % tolerance, iteration count, and printlevel for ipopt
           ipopt_tol = 1e-8;
           ipopt_maxiter = 1000;
           ipopt_printlevel = 0;
           
           % generate initial dataset
           fprintf('Random sampling first %g evaluations...', Ninit)
           X = repmat((x_max-x_min)', [Ninit, 1]).*lhsdesign(Ninit,nx) + repmat(x_min', [Ninit, 1]);
           Z = X*A';
           Y = zeros(Ninit,ny);
           F = zeros(Ninit,1);
           G = zeros(Ninit,ng);
           for k = 1:size(Z,1)
               Y(k,:) = d(Z(k,:)); % evaluate black-box function
               F(k) = f(X(k,:), Y(k,:));
               if ng > 0
                   G(k,:) = g(X(k,:), Y(k,:));
               end
           end
           fprintf('done\n')
                                 
           % loop over maximum iteration
           Nadd = Nmax-Ninit;
           timeIter = zeros(Nadd,1);
           Incumbent = nan*ones(Nadd,1);
           S = nan*ones(Nadd,1);
           for i = 1:Nadd
               % print statement about iteration
               fprintf('Starting iteration %g of %g...', i+Ninit, Nmax)
               
               % record time
               tic;
               
               % fit Gaussian process models
               obj.fit_gaussian_process(Z, Y)
               dmean_func = obj.dmean_ca;
               dvar_func = obj.dvar_ca;
               
               % compute the incumbent
               if ng > 0
                   index_feas = sum(G > tolG, 2); % first find indices of feasible points
                   index_feas = find(index_feas == 0);
               else
                   index_feas = (1:length(F))';
               end
               if sum(index_feas) == 0
                   incumbent = nan; % does not exist since no feasible points
               else
                   incumbent = min(F(index_feas)); % take the incumbent as minimum of feasible
               end
               Incumbent(i) = incumbent;
               
               % approximate the mean objective function using linearization
               ymean = dmean_func(A*x);
               fmean_func = Function('fmean_func', {x}, {obj.f_ca(x, ymean)});
               
               % calculate mean and variance of constraints using linearization
               if ng > 0
                   gmean_func = Function('gmean_func', {x}, {obj.g_ca(x, ymean)});
                   
                   var_g = [];
                   for j = 1:ng
                       gj = g(x,y);
                       gj = gj(j);
                       gj_func = Function('gj_func', {x, y}, {gj});
                       J_gj_y = Function(['J_g' num2str(j) '_y'], {x, y}, {jacobian(gj, y)});
                       var_gj = J_gj_y(x, dmean_func(A*x)) * diag(dvar_func(A*x)) * J_gj_y(x, dmean_func(A*x))';
                       var_g = vertcat(var_g, var_gj);
                   end
                   gvar_func = Function('gvar_func', {x}, {var_g});
               end
               
               % define nonlinear constraints
               taug = -(3-3/Nadd*i);   % taug = norminv(1-0.9);
               if ng > 0
                   nonlincon = @(x)(obj.my_nonlincon(x, gmean_func, gvar_func, taug, ng));
               end
               
               % if incumbent does not exist, we set S=0 and thus only use mean-based
               % search
               if isnan(incumbent)
                   % specify acquisition function
                   s = 0;
                   acquisition_function = @(x)(-reshape(full(fmean_func(x')),[size(x,1),1]));
                   
                   % otherwise we need to include EI and mean in objective
               else
                   % draw random samples to be used for SAA
                   eta = randn(Nmc_eic,ny);

                   % construct the EIC function using MC sampling
                   EICF_samp = Function('EICF_samp', {x}, {max(incumbent - (obj.f_ca(x,dmean_func(A*x)+sqrt(dvar_func(A*x)).*eta')), 0)});
                   expected_improvement_composite = @(x)(mean(reshape(full(EICF_samp(x')),[length(eta),size(x,1)]),1)');
                   
                   % specify acquisition function
                   acquisition_function = expected_improvement_composite;
               end
               S(i) = 0; % store s value
               
               % optimize using ga if specified
               fitness = @(x)(-acquisition_function(x));
               if 0
                   % optimize using genetic algorithm
                   if ng > 0
                       [x_ga, f_ga] = ga(fitness, nx, [], [], [], [], x_min', x_max', nonlincon, ga_opts);
                   else
                       [x_ga, f_ga] = ga(fitness, nx, [], [], [], [], x_min', x_max', [], ga_opts);
                   end
                   
                   % otherwise optimize using random evaluation + then ipopt
               else
                   X_lhs = repmat((x_max-x_min)', [10000*nx, 1]).*lhsdesign(10000*nx,nx) + repmat(x_min', [10000*nx, 1]);
                   if ng > 0
                       G_lhs = nonlincon(X_lhs);
                       index_lhs_feas = sum(G_lhs > tolG, 2); % first find indices of feasible points
                       index_lhs_feas = find(index_lhs_feas == 0);
                   else
                       index_lhs_feas = (1:size(X_lhs,1))';
                       fitness_0 = 1e6;
                   end
                   if sum(index_lhs_feas) == 0
                       Fitness_lhs = fitness(X_lhs);
                       [~,index_lhs] = min(Fitness_lhs + 100*max([G_lhs,zeros(size(G_lhs,1),1)], [], 2).^2);
                   else
                       X_lhs = X_lhs(index_lhs_feas,:);
                       Fitness_lhs = fitness(X_lhs);
                       [fitness_0, index_lhs] = min(Fitness_lhs);
                   end
                   x0_fitness = X_lhs(index_lhs,:);
                   
                   opti_wb2s = casadi.Opti();
                   X_wb2s = opti_wb2s.variable(nx);
                   opti_wb2s.set_initial(X_wb2s, x0_fitness);
                   opti_wb2s.minimize( fitness(X_wb2s') )
                   if ng > 0
                       opti_wb2s.subject_to( nonlincon(X_wb2s') <= 0 )
                   end
                   opti_wb2s.subject_to( x_min <= X_wb2s <= x_max )
                   p_opts = struct('expand',true,'print_time',0);
                   s_opts = struct('max_iter',ipopt_maxiter,'tol',ipopt_tol);
                   s_opts.print_level = ipopt_printlevel;
                   opti_wb2s.solver('ipopt',p_opts,s_opts);
                   try
                       sol_wb2s = opti_wb2s.solve();
                       f_ga = sol_wb2s.value(opti_wb2s.f);
                       if f_ga < fitness_0
                           x_ga = sol_wb2s.value(X_wb2s)';
                       else
                           x_ga = x0_fitness;
                       end
                   catch
                       f_ga = opti_wb2s.debug.value(opti_wb2s.f);
                       if f_ga < fitness_0
                           x_ga = opti_wb2s.debug.value(X_wb2s)';
                       else
                           x_ga = x0_fitness;
                       end
                   end
               end
               
               % add data to X and then scale inputs to prepare for black-box
               % evluation
               X = [X ; x_ga];
               xnewtrue = x_ga;
               znewtrue = x_ga*A';
               for j = 1:nz
                   zNew(:,j) = (znewtrue(:,j) - z_min(j)) / (z_max(j) - z_min(j));
               end
               
               % evaluate black-box function at new inputs
               for l = 1:size(znewtrue,1)
                   ytrue(l,:) = d(znewtrue(l,:));
                   ftrue(l) = f(xnewtrue, ytrue(l,:));
                   if ng > 0
                       gtrue(l,:) = g(xnewtrue, ytrue(l,:));
                   end
               end
               Z = [Z ; znewtrue];
               Y = [Y ; ytrue];
               F = [F ; ftrue];
               if ng > 0
                   G = [G ; gtrue];
               end
                              
               % record time for each iteration
               timeIter(i) = toc;
               
               % print end statement
               fprintf('took %g seconds...incumbent = %g...scaling = %g\n', timeIter(i), Incumbent(i), S(i))
           end
           
           % compute the minimum feasible solution at every iteration
           Fmin = nan*ones(Nmax,1);
           for i = 1:Nmax
               if ng > 0
                   if sum(G(i,:) > tolG) == 0
                       if i == 1
                           Fmin(i) = F(i);
                       else
                           Fmin(i) = min(F(i),Fmin(i-1));
                       end
                   else
                       if i > 1
                           Fmin(i) = Fmin(i-1);
                       end
                   end
               else
                   if i == 1
                       Fmin(i) = F(i);
                   else
                       Fmin(i) = min(F(i),Fmin(i-1));
                   end
               end
           end
           
           % store information
           obj.X = X;
           obj.Z = Z;
           obj.Y = Y;
           obj.F = F;
           obj.G = G;
           obj.Incumbent = Incumbent;
           obj.S = 0;
           obj.timeIter = timeIter;
           obj.Fmin = Fmin;
       end
       
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Specify noninear constraint (with mean+variance); this is used
       %%% to call potentially internal Matlab solvers (like ga)
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function [c, ceq] = my_nonlincon(obj, x, gmean_func, gvar_func, taug, ng)
           ceval = full(gmean_func(x') + taug*gvar_func(x'));
           c = ceval';
           ceq = [];
       end
       
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Fits Gaussian process (GP) regression model to the current (Z,Y)
       %%% data from the black-box function. Uses the DIRECT optimizer +
       %%% fmincon to find hyperparameters. We first scale the data (both
       %%% input and output) to make problem easier. We then store the them
       %%% as casadi functions that can be used for further optimization.
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function fit_gaussian_process(obj, Z, Y, varargin)
           % check if gp options have been set, if not then set them
           if isempty(obj.optgp)
               % create default structure
               for i = 1:obj.ny
                   opt.GP(i).nSpectralpoints = 1000;   % Number of spectral sampling points for output i
                   opt.GP(i).matern = inf;             % Matern type 1 / 3 / 5 / inf for output i
                   opt.GP(i).fun_eval = 200;           % Function evaluations by direct algorithm per input dimension for output i
               end
               
               % initialization options structure
               opt = setOptionStructure(opt,Z,Y);
               
               % store in the object
               obj.optgp = opt;
           end
           
           % scale variables
           [Znew, Ynew] = scaleVariables(Z,Y,obj.z_min',obj.z_max');
           
           % train Gaussian process (GP) models for each output
           for j = 1:obj.ny
               if sum(isnan(Ynew(:,j))) == 0
                   obj.optgp.GP(j).hyp = fitGaussianProcess(Znew,Ynew(:,j),obj.optgp.GP(j));
               end
           end

           % import casadi
           import casadi.*           
           
           % get current GP mean and variance function as a casadi function
           % which generally maps vector of size w to vector of size y
           z = SX.sym('z',obj.nz);
           znew = (z - obj.z_min)./(obj.z_max - obj.z_min);
           dmean = [];
           dvar = [];
           for j = 1:obj.ny
               if sum(isnan(Ynew(:,j))) == 0
                   [dmean_j, dvar_j] = getTrueMeanFunction(Znew,Ynew(:,j),obj.optgp.GP(j));
               else
                   dmean_j = Function('dmean_j', {z}, {0});
                   dvar_j = Function('dvar_j', {z}, {0});
               end
               dmean = vertcat(dmean, std(Y(:,j))'.*dmean_j(znew') + mean(Y(:,j))');
               dvar = vertcat(dvar, var(Y(:,j))'.*dvar_j(znew'));
           end
           
           % store mean and variance as casadi functions
           obj.dmean_ca = Function('dmean_ca',{z},{dmean});
           obj.dvar_ca = Function('dvar_ca',{z},{dvar}); 
       end

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Simple helper function to predict only mean of each individual
       %%% GP to avoid computations
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function ypred = predict_blackbox_mean(obj, z)
           ypred = full(obj.dmean_ca(z));
       end       
       
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Simple helper function to predict mean and standard deviation of
       %%% each individual GP model stacked into a vector for a set of inputs w
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function [ypred,ystd] = predict_blackbox(obj, z)
           ypred = full(obj.dmean_ca(z));
           ystd = sqrt(max(full(obj.dvar_ca(z)),0));
       end

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Simple function to help visualize the 1d GP predictions in the
       %%% relevant subspace; must specify the rest of the w variables to be
       %%% fixed at some values
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function plot_gaussian_process_1d(obj, varargin)
           % set default for input dimension to be first dimension
           zdim = 1;
           % set default for output dimension to be first dimension
           ydim = 1;
           % set defaults of other dimensions to be 0
           z_default = zeros(obj.nz,1);
           % set default for plotting true function to be false
           plot_true = 0;
           % second argument is input dimension of interest
           if nargin > 1 && ~isempty(varargin{1})
               zdim = varargin{1};
           end
           % thrid argument is output dimension of interest
           if nargin > 2 && ~isempty(varargin{2})
               ydim = varargin{2};
           end
           % fourth argument is fixed w vaues
           if nargin > 3 && ~isempty(varargin{3})
               z_default = varargin{3};
           end
           % fifth argument is whether or not to plot true values
           if nargin > 4 && ~isempty(varargin{4})
               plot_true = varargin{4};
           end
           
           % get list of w values in the relevant dimension
           z_list = linspace(obj.z_min(zdim), obj.z_max(zdim), 101);
           Z_list = repmat(z_default, [1, 101]);
           Z_list(zdim,:) = z_list;
           
           % evaluate mean and standard deviation 
           [Ypred_list, Ystd_list] = obj.predict_blackbox(Z_list);
           ypred_list = Ypred_list(ydim,:);
           ystd_list = Ystd_list(ydim,:);
           
           % plot the GP predictions (mean and variance +/- 3*std clouds)           
           figure; hold on;
           plot(z_list, ypred_list, '--r', 'linewidth', 2)
           patch([z_list' ; flipud(z_list')], [ypred_list+3.*ystd_list, fliplr(ypred_list-3.*ystd_list)]','r','FaceAlpha',0.1)           
           if obj.nz == 1
               scatter(obj.Z(:,zdim), obj.Y(:,ydim), 50, 'ko')
           end
           if plot_true == 1
               Ytrue_list = obj.d(Z_list);
               ytrue_list = Ytrue_list(ydim,:);
               plot(z_list, ytrue_list, '-k', 'linewidth', 2)
           end
       end

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Simple function to help visualize the 2d predictions of the GP
       %%% mean function only
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function plot_gaussian_process_mean_2d(obj, varargin)
           % set default for input dimension to be first and second dimension
           z1dim = 1;
           z2dim = 2;
           % set default for output dimension to be first dimension
           ydim = 1;
           % set defaults of other dimensions to be 0
           z_default = zeros(obj.nz,1);
           % set default for plotting true function to be false
           plot_true = 0;
           % second argument is first input dimension of interest
           if nargin > 1 && ~isempty(varargin{1})
               z1dim = varargin{1};
           end
           % third argument is second input dimension of interest
           if nargin > 2 && ~isempty(varargin{2})
               z2dim = varargin{2};
           end
           % fourth argument is output dimension of interest
           if nargin > 3 && ~isempty(varargin{3})
               ydim = varargin{3};
           end
           % fifth argument is fixed w vaues
           if nargin > 4 && ~isempty(varargin{4})
               z_default = varargin{4};
           end
           % sixth argument is whether or not to plot true grid
           if nargin > 5 && ~isempty(varargin{5})
               plot_true = varargin{5};
           end
           
           % construct meshgrid and evaluate 
           Z_ij = z_default;
           z1_list = linspace(obj.z_min(z1dim),obj.z_max(z1dim),51);
           z2_list = linspace(obj.z_min(z2dim),obj.z_max(z2dim),51);
           [Z1_grid, Z2_grid] = meshgrid(z1_list, z2_list);
           Ypred_grid = zeros(size(Z1_grid));
           if plot_true == 1
               Ytrue_grid = zeros(size(Z1_grid));
           end
           for i = 1:length(z1_list)
               for j = 1:length(z2_list)
                   Z_ij(z1dim) = Z1_grid(i,j);
                   Z_ij(z2dim) = Z2_grid(i,j);
                   Ycurr = obj.predict_blackbox_mean(Z_ij);
                   Ypred_grid(i,j) = Ycurr(ydim);
                   if plot_true == 1                       
                       Ytrue = obj.d(Z_ij);
                       Ytrue_grid(i,j) = Ytrue(ydim);
                   end
               end
           end
           
           % plot the grid values using a countour plot
           figure; hold on;
           h = pcolor(Z1_grid, Z2_grid, Ypred_grid);
           scatter(obj.Z(:,z1dim), obj.Z(:,z2dim), 100, 'kx', 'linewidth', 2)
           colorbar;
           set(h, 'EdgeColor', 'none');       
           if plot_true == 1
               figure; hold on;
               h = pcolor(Z1_grid, Z2_grid, Ytrue_grid);
               colorbar;
               set(h, 'EdgeColor', 'none');
           end
           set(gcf,'color','w');
           set(gca,'FontSize',20)           
       end       
   end
   
   methods(Static)
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%% Run the black-box algorithm bayesopt
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function [f,c] = blackBoxFunction(x, ffunc, cfunc, dfunc, A, nc)
           x = table2array(x)';
           f = ffunc(x,dfunc(A*x));
           if nc > 0
               c = cfunc(x,dfunc(A*x));
           else
               c = [];
           end
       end
   end
end