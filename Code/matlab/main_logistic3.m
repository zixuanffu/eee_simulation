%%
% Logistic Location Model for Efficiency (Bootstrap validity)
%
% Section 3.1.1 of Kaji, Manresa, and Pouliot (2023) "An Adversarial
% Approach to Structural Estimation"
%
% Produces the bootstrap standard errors 2.28 for logistic discriminator
% and 2.55 for neural network discriminator.
%
% Written by Tetsuya Kaji


%%
% Setup
%

% To save the source code in the data
%
code = fileread('main_logistic3.m');

% For replication
%
rng default

% Sample sizes
%
n = 300;
m = n;

% Bootstrap size
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
g = 20;
node = 5;

% True parameter
%
th0 = 0;

% Likelihood
%
log_p0 = @(x) loglogisticpdf(x,th0,1);
log_p = @(th,x) loglogisticpdf(x,th,1);

% Miscellaneous intermediate functions
%
log_D = @(th,x) log_p0(x)-logaddexp(log_p0(x),log_p(th,x));     % log(D)
log_UmD = @(th,x) log_p(th,x)-logaddexp(log_p0(x),log_p(th,x)); % log(1-D)


%%
% Estimation
%

% Real data
%
X = logisticrnd(0,1,n,1);

% Latent data
%
X_tilde = logisticrnd(0,1,m,1);

% MLE
%
MLE = fminsearch(@(th) -sum(loglogisticpdf(X,th,1)),1);

% Oracle adversarial estimator
%
oAdv = fminsearch(@(th) mean(log_D(th,X))+mean(log_UmD(th,T(th,X_tilde))),MLE);

% Correctly specified adversarial estimator with logistic discriminator
%
AdvL = fminsearch(@(th) loss(X,T(th,X_tilde)),oAdv);

% Adversarial estimator with neural network discriminator
%
AdvN = fminsearchgrid(@(th) NND(X',T(th,X_tilde)',g,node),oAdv-2.5/sqrt(n),oAdv+2.5/sqrt(n),15);


%%
% Bootstrap
%

% Memory allocation
%
th_MLE = zeros(S,1);
th_oAdv = zeros(S,1);
th_AdvL = zeros(S,1);
th_AdvN = zeros(S,1);

% Initialize progress tracker
%
fID = fopen('main_logistic3.txt','w');
fprintf(fID,'main_logistic3''s routine started at %s\n',datetime('now'));
fprintf(fID,[...
    '  Step    Passed        At       To Go     Until\n' ...
    '-------+---------+--------------+------+-----------\n']);
fclose(fID);
fprintf([...
    '  Step    Passed        At       To Go     Until\n' ...
    '-------+---------+--------------+------+-----------\n'])

tic
for s = 1 : S

    % Index for bootstrap
    %
    bi1 = unidrnd(n,n,1);
    bi2 = unidrnd(m,m,1);

    % Real data
    %
    bX = X(bi1,:);

    % Latent data
    %
    bU = X_tilde(bi2,:);

    % MLE
    %
    th_MLE(s) = fminsearch(@(th) -sum(loglogisticpdf(bX,th,1)),1);

    % Oracle adversarial estimator
    %
    th_oAdv(s) = fminsearch(@(th) mean(log_D(th,bX))+mean(log_UmD(th,T(th,bU))),th_MLE(s));

    % Correctly specified adversarial estimator with logistic discriminator
    %
    th_AdvL(s) = fminsearch(@(th) loss(bX,T(th,bU)),th_oAdv(s));

    % Adversarial estimator with neural network discriminator
    %
    th_AdvN(s) = fminsearchgrid(@(th) NND(bX',T(th,bU)',g,node),th_oAdv(s)-2.5/sqrt(n),th_oAdv(s)+2.5/sqrt(n),15);

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
        tnow = datetime('now');
        tdone = tnow+seconds(sec_togo);
        fID = fopen('main_logistic3.txt','a+');
        fprintf(fID,'%3d/%3d %3d:%02d:%02d %s %3d:%02d %s\n',s,S,h_psst,m_psst,s_psst,datetime(tnow,'Format','MM/dd HH:mm:ss'),h_togo,m_togo,datetime(tdone,'Format','MM/dd HH:mm'));
        fclose(fID);
        fprintf('%3d/%3d %3d:%02d:%02d %s %3d:%02d %s\n',s,S,h_psst,m_psst,s_psst,datetime(tnow,'Format','MM/dd HH:mm:ss'),h_togo,m_togo,datetime(tdone,'Format','MM/dd HH:mm'))
    end
end

% Finalize progress tracker
%
fID = fopen('main_logistic3.txt','a+');
fprintf(fID,'-------+---------+--------------+------+-----------\n');
fclose(fID);
fprintf('-------+---------+--------------+------+-----------\n')

toc


%%
% Figures
%

bins = linspace(-0.7,0.7,24);

figure(1)
subplot(1,3,1)
histogram(th_oAdv,bins,'Normalization','pdf')
hold on
histogram(th_MLE,bins,'Normalization','pdf')
hold off
xlabel('$\theta$','Interpreter','latex')
title(sprintf('n=%d, m=%d', n, m))
legend(sprintf('oAdv (%.2f)', sqrt(n)*std(th_oAdv)), ...
       sprintf('MLE (%.2f)', sqrt(n)*std(th_MLE)))
legend('boxoff')

figure(1)
subplot(1,3,2)
histogram(th_AdvL,bins,'Normalization','pdf')
hold on
histogram(th_MLE,bins,'Normalization','pdf')
hold off
xlabel('$\theta$','Interpreter','latex')
title(sprintf('n=%d, m=%d', n, m))
legend(sprintf('AdvL (%.2f)', sqrt(n)*std(th_AdvL)), ...
       sprintf('MLE (%.2f)', sqrt(n)*std(th_MLE)))
legend('boxoff')

figure(1)
subplot(1,3,3)
histogram(th_AdvN,bins,'Normalization','pdf')
hold on
histogram(th_MLE,bins,'Normalization','pdf')
hold off
xlabel('$\theta$','Interpreter','latex')
title(sprintf('n=%d, m=%d', n, m))
legend(sprintf('AdvN (%.2f)', sqrt(n)*std(th_AdvN)), ...
       sprintf('MLE (%.2f)', sqrt(n)*std(th_MLE)))
legend('boxoff')


%%
% Save Matlab workspace
%

seed = rng;
save(sprintf('main_logistic3_%s.mat',datetime('now','format','yyyyMMddHHmmss')))


%%
% Functions
%

%
% Classification accuracy with logistic discriminator
%
function [v,beta] = loss(X1,X2)

    [beta,v] = fminsearch(@(l) mean(softplus(indexf(X1,l)))+mean(softplus(-indexf(X2,l))),[1;1]);
    v = -v;

    % Nonlinear index for correctly specified discriminator
    %
    function idx = indexf(x,l)
        idx = l(1)+2*(softplus(-x)-softplus(-x+l(2)));
    end
end
