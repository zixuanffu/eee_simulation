%%
% Logistic Location Model for Efficiency
%
% Section 3.1.1 of Kaji, Manresa, and Pouliot (2023) "An Adversarial
% Approach to Structural Estimation"
%
% Produces Figures 1 and 2.
%
% Written by Tetsuya Kaji


%%
% Setup
%

% To save the source code in the data
%
code = fileread('main_logistic.m');

% For replication
%
rng default

% Sample sizes
%
n = 300;
M = n*[1 2];

% Simulation size
%
S = 500;

% Transformation map from parameter to simulated data
%
T = @(th,x) th+x;

% Neural network configuration
%
% We estimate discriminator [g] times and take average. Then we minimize
% this average discriminator over parameter values. The neural network
% consists of [node] number of nodes.
%
g = [20 250];
node = 5;

assert(isequal(size(M),size(g)))

% True parameter
%
th0 = 0;


%%
% Analytic calculation
%

% Likelihood
%
log_p0 = @(x) loglogisticpdf(x,th0,1);
log_p = @(th,x) loglogisticpdf(x,th,1);

% Score and Hessian
%
ld = @(th,x) -1+2*logisticcdf(x-th,0,1);
ldd = @(th,x) -2*logisticcdf(x-th,0,1).*logisticcdf(-(x-th),0,1);

% Miscellaneous intermediate functions
%
log_D = @(th,x) log_p0(x)-logaddexp(log_p0(x),log_p(th,x));     % log(D)
log_UmD = @(th,x) log_p(th,x)-logaddexp(log_p0(x),log_p(th,x)); % log(1-D)
p0 = @(x) exp(log_p0(x));           % p_0
p = @(th,x) exp(log_p(th,x));       % p_theta
D = @(th,x) exp(log_D(th,x));       % Oracle D

% The middle term of the sandwich formula in Theorem 3
%
% It computes the version that does not assume "n/m -> 0" (Assumption 2),
% which is given in Theorem 3 of arXiv:2007.06169v2.
%
meat = @(x) 4*exp(log_D(th0,x)+log_UmD(th0,x)).*ld(th0,x).^2;
Meat_1 = integral(@(x) meat(x).*p(th0,x),-Inf,Inf,'AbsTol',1e-20);
Meat_2 = integral(@(x) meat(x).*p0(x),-Inf,Inf,'AbsTol',1e-20);
V = Meat_1+(n./M).*Meat_2;

% The wing terms of the sandwich formula in Theorem 3
%
bun = @(x) 2*(D(th0,x).*ld(th0,x).^2+(ldd(th0,x)+ld(th0,x).^2).*log_UmD(th0,x));
I_tilde = integral(@(x) bun(x).*p(th0,x),-Inf,Inf,'AbsTol',1e-20);

% Asymptotic variance of Adv in Theorem 3
%
V_Adv = I_tilde\V/I_tilde;

% Asymptotic standard deviation of MLE
%
V_MLE = -1/integral(@(x) ldd(th0,x).*p0(x),-Inf,Inf);


%%
% Orthogonality
%

K = 50;
th_grid = linspace(-0.7,0.7,K);
LL_grid = zeros(1,K);
oD_grid = zeros(numel(M),K);
cD_grid = zeros(numel(M),K);
NND_grid = zeros(numel(M),K);

% Real data
%
X = logisticrnd(0,1,n,1);

% Latent data
%
Z = logisticrnd(0,1,max(M),1);

tic
for k = 1 : K

    th = th_grid(k);

    % Log likelihood
    %
    LL_grid(k) = -mean(log_p(th,X));

    for m = 1 : numel(M)

        % Oracle Adv objective function
        %
        oD_grid(m,k) = mean(log_D(th,X))+mean(log_UmD(th,T(th,Z(1:M(m)))));

        % Correctly specified Adv objective function
        %
        cD_grid(m,k) = loss(X,T(th,Z(1:M(m))));

        % Neural network Adv objective function
        %
        NND_grid(m,k) = NND(X',T(th,Z(1:M(m)))',g(m),node);

    end
end

LL0_grid = -mean(log_p(th0,X));
oD0_grid = zeros(numel(M),1);
cD0_grid = zeros(numel(M),1);
NND0_grid = zeros(numel(M),1);
for m = 1 : numel(M)
    oD0_grid(m) = mean(log_D(th0,X))+mean(log_UmD(th0,T(th0,Z(1:M(m)))));
    cD0_grid(m) = loss(X,T(th0,Z(1:M(m))));
    NND0_grid(m) = NND(X',T(th0,Z(1:M(m)))',g(m),node);
end
toc


%%
% Figures
%

% Figure 1a
%
figure(1)
subplot(1,3,1)
m = 1;
plot(th_grid, [cD_grid(m,:);oD_grid(m,:)])
xlim([-0.7 0.7])
ylim([-1.4 -1.3])
hold on
yyaxis right
plot(th_grid, LL_grid/2,'LineWidth',1)
ylim([0.96 1.06])
hold off
yyaxis left
xlabel('$\theta$','Interpreter','latex')
title(sprintf('n=%d, m=%d', n, M(m)))
legend(...
    '${\bf M}_\theta(\hat{D}_\theta)$',...
    '${\bf M}_\theta(D_\theta)$',...
    '${\bf L}_\theta$',...
    'Interpreter','latex')
legend('boxoff')

% Figure 2a
%
figure(2)
subplot(1,3,1)
m = 1;
plot(th_grid, [NND_grid(m,:);oD_grid(m,:)])
xlim([-0.7 0.7])
ylim([-1.4 -1.3])
hold on
yyaxis right
plot(th_grid, LL_grid/2)
ylim([0.96 1.06])
hold off
yyaxis left
xlabel('$\theta$','Interpreter','latex')
title(sprintf('n=%d, m=%d', n, M(m)))
legend(...
    '${\bf M}_\theta(\hat{D}_\theta)$',...
    '${\bf M}_\theta(D_\theta)$',...
    '${\bf L}_\theta$',...
    'Interpreter','latex')
legend('boxoff')

% Figure 2b
%
figure(2)
subplot(1,3,2)
m = 2;
plot(th_grid, [NND_grid(m,:);oD_grid(m,:)])
xlim([-0.7 0.7])
ylim([-1.55 -1.3])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('n=%d, m=%d', n, M(m)))
legend(...
    '${\bf M}_\theta(\hat{D}_\theta)$',...
    '${\bf M}_\theta(D_\theta)$',...
    'Interpreter','latex')
legend('boxoff')

% Figure 2c
%
figure(2)
subplot(1,3,3)
m = 2;
plot(th_grid, [NND_grid(m,:)-NND0_grid(m);oD_grid(m,:)-oD0_grid(m);(LL_grid-LL0_grid)/2])
xlim([-0.7 0.7])
ylim([-0.02 0.08])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('n=%d, m=%d', n, M(m)))
legend(...
    '${\bf M}_\theta(\hat{D}_\theta)$',...
    '${\bf M}_\theta(D_\theta)$',...
    '${\bf L}_\theta$',...
    'Interpreter','latex')
legend('boxoff')


% Export csv for Figures 1a and 2
%
writetable(array2table([th_grid;LL_grid/2;oD_grid;cD_grid;NND_grid]','VariableNames',...
    {'theta' 'LL/2' 'oD(1)' 'oD(2)' 'cD(1)' 'cD(2)' 'NND(1)' 'NND(2)'}),...
    'logistic_orthogonality.csv')


%%
% Estimation
%

% Memory allocation
%
th_MLE = zeros(S,1);
th_oAdv = zeros(S,numel(M));
th_AdvL = zeros(S,numel(M));
th_AdvN = zeros(S,numel(M));

% Initialize progress tracker
%
fID = fopen('main_logistic.txt','w');
fprintf(fID,'main_logistic''s routine started at %s\n',datetime('now'));
fprintf(fID,[...
    '+----------+----------+----------------+-----------+----------------+\n' ...
    '|   Step   |  Passed  |       At       | Remaining |     Until      |\n' ...
    '+----------+----------+----------------+-----------+----------------+\n']);
fclose(fID);
fprintf([...
    '+----------+----------+----------------+-----------+----------------+\n' ...
    '|   Step   |  Passed  |       At       | Remaining |     Until      |\n' ...
    '+----------+----------+----------------+-----------+----------------+\n'])

tic
for s = 1 : S

    % Real data
    %
    X = logisticrnd(0,1,n,1);

    % Latent data
    %
    Z = logisticrnd(0,1,max(M),1);

    % Initialization
    %
    th = (-1)^binornd(1,0.5);

    % MLE
    %
    th_MLE(s) = fminsearch(@(th) -sum(loglogisticpdf(X,th,1)),th);

    for m = 1 : numel(M)

        % Oracle adversarial estimator
        %
        th_oAdv(s,m) = fminsearch(@(th) mean(log_D(th,X))+mean(log_UmD(th,T(th,Z(1:M(m))))),th_MLE(s));

        % Correctly specified adversarial estimator with logistic discriminator
        %
        th_AdvL(s,m) = fminsearch(@(th) loss(X,T(th,Z(1:M(m)))),th_oAdv(s,m));

        % Adversarial estimator with neural network discriminator
        %
        th_AdvN(s,m) = fminsearchgrid(@(th) NND(X',T(th,Z(1:M(m)))',g(m),node),th_oAdv(s,m)-2.5/sqrt(n),th_oAdv(s,m)+2.5/sqrt(n),15);

    end

    % Update progress tracker
    %
    if mod(s,1) == 0
        sec_psst = toc;  % seconds passed since tic
        h_psst = floor(sec_psst/60/60);
        m_psst = floor(mod(sec_psst,60*60)/60);
        s_psst = floor(mod(sec_psst,60));
        sec_togo = (S-s)/s*sec_psst;
        h_togo = floor(sec_togo/60/60);
        m_togo = floor(mod(sec_togo,60*60)/60);
        s_togo = floor(mod(sec_togo,60));
        tnow = datetime('now');
        tdone = tnow+seconds(sec_togo);
        fID = fopen('main_logistic.txt','a+');
        fprintf(fID,'| %3d /%3d |%3d:%02d:%02d | %s | %3d:%02d:%02d | %s |\n',s,S,h_psst,m_psst,s_psst,datetime(tnow,'Format','MM/dd HH:mm:ss'),h_togo,m_togo,s_togo,datetime(tdone,'Format','MM/dd HH:mm:ss'));
        fclose(fID);
        fprintf('| %3d /%3d |%3d:%02d:%02d | %s | %3d:%02d:%02d | %s |\n',s,S,h_psst,m_psst,s_psst,datetime(tnow,'Format','MM/dd HH:mm:ss'),h_togo,m_togo,s_togo,datetime(tdone,'Format','MM/dd HH:mm:ss'))
    end
end

% Finalize progress tracker
%
fID = fopen('main_logistic.txt','a+');
fprintf(fID,'+----------+----------+----------------+-----------+----------------+\n');
fclose(fID);
fprintf('+----------+----------+----------------+-----------+----------------+\n')

toc


%%
% Figures
%

% Figure 1b
%
figure(1)
subplot(1,3,2)
bins = linspace(-0.7,0.7,24);
m = 1;
histogram(th_oAdv(:,m),bins,'Normalization','pdf')
hold on
histogram(th_MLE,bins,'Normalization','pdf')
hold off
ylim([0 6])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('n=%d, m=%d', n, M(m)))
legend(sprintf('oAdv (%.2f)', sqrt(n)*std(th_oAdv(:,m))), ...
       sprintf('MLE (%.2f)', sqrt(n)*std(th_MLE)))
legend('boxoff')

% Figure 1c
%
figure(1)
subplot(1,3,3)
bins = linspace(-0.7,0.7,24);
m = 1;
histogram(th_AdvL(:,m),bins,'Normalization','pdf')
hold on
histogram(th_MLE,bins,'Normalization','pdf')
hold off
ylim([0 6])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('n=%d, m=%d', n, M(m)))
legend(sprintf('Adv (%.2f)', sqrt(n)*std(th_AdvL(:,m))), ...
       sprintf('MLE (%.2f)', sqrt(n)*std(th_MLE)))
legend('boxoff')


% Export csv for Figures 1b and 1c
%
writetable(array2table([th_MLE th_oAdv th_AdvL th_AdvN],'VariableNames',...
    {'MLE' 'oAdv(1)' 'oAdv(2)' 'AdvL(1)' 'AdvL(2)' 'AdvN(1)' 'AdvN(2)'}),...
    'logistic_estimator.csv')


%%
% Save Matlab workspace and figures
%

seed = rng;
save(sprintf('main_logistic_%s.mat',datetime('now','format','yyyyMMddHHmmss')))

saveas(figure(1),'Figure1.png')
saveas(figure(2),'Figure2.png')


%%
% Functions
%

%
% Classification accuracy with logistic discriminator
%
function [v,beta] = loss(X1,X2)

    [beta,v] = fminsearch(@(l) mean(softplus(indexf(X1,l)))+mean(softplus(-indexf(X2,l))),[1;2;1]);
    v = -v;

    % Nonlinear index for correctly specified discriminator
    %
    function idx = indexf(x,l)
        idx = l(1)+l(2)*(softplus(-x)-softplus(-x+l(3)));
    end
end
