%%
% Logistic Misspecification Model
%
% Section 3.1.2 of Kaji, Manresa, and Pouliot (2023) "An Adversarial
% Approach to Structural Estimation"
%
% Produces Figure 3.
% Also produces Figure 13a of the earlier version arXiv:2007.06169v2.
%
% Written by Tetsuya Kaji


%%
% Setup
%

% To save the source code in the data
%
code = fileread('main_misspec.m');

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
g = [20 100];
node = 5;
input = @(x) [x x.^2 softplus(x)]';


%%
% Analytic calculation
%

% Likelihood
%
log_p0 = @(x) loglogisticpdf(x,0,1);
log_p = @(th,x) lognormpdf(x,th,1);

% Score and Hessian
%
ld = @(th,x) x-th;
ldd = @(th,x) -ones(size(x+th));

% Density derivatives
%
dpdxp = @(th,x) -(x-th);                % (dp/dx)/p
dp0dxp0 = @(x) 1-2*logisticcdf(x,0,1);  % (dp0/dx)/p0
dlddx = @(th,x) -ones(size(x+th));      % (dld/dx)

% Pseudo-parameter
%
th0 = 0;

% Miscellaneous intermediate functions
%
log_D = @(th,x) log_p0(x)-logaddexp(log_p0(x),log_p(th,x));     % log(D)
log_UmD = @(th,x) log_p(th,x)-logaddexp(log_p0(x),log_p(th,x)); % log(1-D)
p0 = @(x) exp(log_p0(x));           % p_0
p = @(th,x) exp(log_p(th,x));       % p_theta
D = @(th,x) exp(log_D(th,x));       % Oracle D
UmD = @(th,x) exp(log_UmD(th,x));   % Oracle 1-D
tau = @(x) D(th0,x).*(dpdxp(th0,x)-dp0dxp0(x)); % \tau_n\circ T_{\theta_0}^{-1}

% The middle term of the sandwich formula in Theorem 3
%
% It computes the version that does not assume "n/m -> 0" (Assumption 2),
% which is given in Theorem 3 of arXiv:2007.06169v2.
%
meat = @(x) 4*exp(log_D(th0,x)+log_UmD(th0,x)).*ld(th0,x).^2;
Meat_1 = integral(@(x) meat(x).*p(th0,x),-Inf,Inf,'AbsTol',1e-20);
Meat_2 = integral(@(x) meat(x).*p0(x),-Inf,Inf,'AbsTol',1e-20);
Meat_a = integral(@(x) 4*(2*D(th0,x).*ld(th0,x).*tau(x)+tau(x).^2).*p(th0,x),-Inf,Inf,'AbsTol',1e-20);
V = Meat_1+(n./M).*(Meat_2+Meat_a);

% The wing terms of the sandwich formula of Theorem 3
%
bun = @(x) 2*(D(th0,x).*ld(th0,x).^2+(ldd(th0,x)+ld(th0,x).^2).*log_UmD(th0,x));
I_tilde = integral(@(x) bun(x).*p(th0,x),-Inf,Inf,'AbsTol',1e-20);

% Asymptotic variance of Adv
%
V_Adv = I_tilde\V/I_tilde;

% Asymptotic variance of qMLE
%
V_qMLE = pi^2/3;


% Function to calculate P_theta log(1-D_theta_0) for the high-level
% condition in Assumption 4 of arXiv:2007.06169v2
%
% (This condition is only necessary when "n/m -> 0" is not satified,
% the discussion of which is removed in the published version.)
%
P_log_UmD = @(th) integral(@(x) log_UmD(th0,x).*p(th,x),-Inf,Inf,'ArrayValued',1);

% Function to calculate P_theta D_theta_0 ld_theta_0 for the high-level
% condition in Assumption 6 of arXiv:2007.06169v2
%
% (This condition is only necessary when "n/m -> 0" is not satified,
% the discussion of which is removed in the published version.)
%
P_D_ld = @(th) integral(@(x) D(th0,x).*ld(th0,x).*p(th,x),-Inf,Inf,'ArrayValued',1);


%%
% Orthogonality
%

% Real data
%
X = logisticrnd(0,1,n,1);

% Latent data
%
Z = normrnd(0,1,max(M),1);

K = 50;
th_grid = linspace(-0.7,0.7,K);
h_grid = sqrt(n)*(th_grid-th0);
LL_grid = zeros(1,K);
qLL_grid = zeros(1,K);
oD_grid = zeros(numel(M),K);
cD_grid = zeros(numel(M),K);
NND_grid = zeros(numel(M),K);

tic
for k = 1 : K

    th = th_grid(k);

    % Log likelihood
    %
    LL_grid(k) = -mean(loglogisticpdf(X,th,1));

    % Quasi log likelihood
    %
    qLL_grid(k) = -mean(log_p(th,X));

    for m = 1 : numel(M)

        % Oracle Adv objective function
        %
        oD_grid(m,k) = mean(log_D(th,X))+mean(log_UmD(th,T(th,Z(1:M(m)))));

        % Correctly specified Adv objective function
        %
        [cD_grid(m,k),beta] = loss(X,T(th,Z(1:M(m))));

        % Neural network Adv objective function
        %
        NND_grid(m,k) = NND(input(X),input(T(th,Z(1:M(m)))),g(m),node);

    end
end

% For normalization
%
LL0_grid = -mean(loglogisticpdf(X,0,1));
qLL0_grid = -mean(log_p(th0,X));
oD0_grid = zeros(numel(M),1);
cD0_grid = zeros(numel(M),1);
NND0_grid = zeros(numel(M),1);
for m = 1 : numel(M)
    oD0_grid(m) = mean(log_D(th0,X))+mean(log_UmD(th0,T(th0,Z(1:M(m)))));
    cD0_grid(m) = loss(X,T(th0,Z(1:M(m))));
    NND0_grid(m) = NND(input(X),input(T(th0,Z(1:M(m)))),g(m),node);
end
toc


%%
% Figures
%

% Figure 3a
%
figure(3)
subplot(1,3,1)
m = 1;
plot(th_grid, [cD_grid(m,:);oD_grid(m,:)],'LineWidth',1)
xlim([-0.7 0.7])
ylim([-1.28 -1.18])
hold on
yyaxis right
plot(th_grid, qLL_grid/2,'LineWidth',1)
ylim([1.16 1.26])
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


% Export csv for Figure 3a
%
writetable(array2table([th_grid;qLL_grid/2;oD_grid;cD_grid;NND_grid]','VariableNames',...
    {'theta' 'qLL/2' 'oD(1)' 'oD(2)' 'cD(1)' 'cD(2)' 'NND(1)' 'NND(2)'}),...
    'misspec_orthogonality.csv')


%%
% Verifying smooth synthetic data generation in arXiv:2007.06169v2
%

asm4 = zeros(numel(M),K);
asm6 = zeros(numel(M),K);

for m = 1 : numel(M)

    % Assumption 4 of arXiv:2007.06169v2
    %
    asm4(m,:) = (mean(log_UmD(th0,T(th_grid,Z(1:M(m)))),1)-P_log_UmD(th_grid)) ...
              - (mean(log_UmD(th0,T(th0,Z(1:M(m)))),1)-P_log_UmD(th0));

    % Assumption 6 of arXiv:2007.06169v2
    %
    asm6(m,:) = (mean(D(th0,T(th_grid,Z(1:M(m)))).*ld(th0,T(th_grid,Z(1:M(m)))),1)-P_D_ld(th_grid)) ...
              - (mean(D(th0,T(th0,Z(1:M(m)))).*ld(th0,T(th0,Z(1:M(m)))),1)-P_D_ld(th0));
    asm6(m,:) = asm6(m,:).*(th_grid-th0);

end


% Figure 13a of arXiv:2007.06169v2
%
figure(13)
m = 1;
plot(th_grid,oD_grid(m,:)-oD0_grid(m),'LineWidth',1)
hold on
plot(th_grid,[asm4(m,:);asm6(m,:)],'LineWidth',1.5,'LineStyle',':')
hold off
xlim([-0.7 0.7])
ylim([-0.02 0.08])
xlabel('$\theta$','Interpreter','latex')
ylabel('loss')
title(sprintf('n=%d, m=%d', n, M(m)))
legend('${\bf M}_\theta(D_\theta)-{\bf M}_{\theta_0}(D_{\theta_0})$',...
    'Assumption 4','Assumption 6','Interpreter','latex')
legend('boxoff')


%%
% Estimation
%

% Memory allocation
%
th_MLE = zeros(S,1);
th_qMLE = zeros(S,1);
th_oAdv = zeros(S,numel(M));
th_AdvL = zeros(S,numel(M));

% Initialize progress tracker
%
fID = fopen('main_misspec.txt','w');
fprintf(fID,'main_misspec''s routine started at %s\n',datetime('now'));
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
    Z = normrnd(0,1,max(M),1);

    % Initialization
    %
    th = (-1)^binornd(1,0.5);

    % MLE
    %
    th_MLE(s) = fminsearch(@(th) -sum(loglogisticpdf(X,th,1)),th);

    % Quasi MLE
    %
    th_qMLE(s) = fminsearch(@(th) -sum(log_p(th,X)),th);

    for m = 1 : numel(M)

        % Oracle Adv
        %
        th_oAdv(s,m) = fminsearch(@(th) mean(log_D(th,X))+mean(log_UmD(th,T(th,Z(1:M(m))))),th_qMLE(s));

        % Correctly Specified Adv
        %
        th_AdvL(s,m) = fminsearch(@(th) loss(X,T(th,Z(1:M(m)))),th_oAdv(s,m));

    end

    % Update progress tracker
    %
    if mod(s,5) == 0
        sec_psst = toc;  % seconds since tic
        h_psst = floor(sec_psst/60/60);
        m_psst = floor(mod(sec_psst,60*60)/60);
        s_psst = floor(mod(sec_psst,60));
        sec_togo = (S-s)/s*sec_psst;
        h_togo = floor(sec_togo/60/60);
        m_togo = floor(mod(sec_togo,60*60)/60);
        s_togo = floor(mod(sec_togo,60));
        tnow = datetime('now');
        tdone = tnow+seconds(sec_togo);
        fID = fopen('main_misspec.txt','a+');
        fprintf(fID,'| %3d /%3d |%3d:%02d:%02d | %s | %3d:%02d:%02d | %s |\n',s,S,h_psst,m_psst,s_psst,datetime(tnow,'Format','MM/dd HH:mm:ss'),h_togo,m_togo,s_togo,datetime(tdone,'Format','MM/dd HH:mm:ss'));
        fclose(fID);
        fprintf('| %3d /%3d |%3d:%02d:%02d | %s | %3d:%02d:%02d | %s |\n',s,S,h_psst,m_psst,s_psst,datetime(tnow,'Format','MM/dd HH:mm:ss'),h_togo,m_togo,s_togo,datetime(tdone,'Format','MM/dd HH:mm:ss'))
    end
end

% Finalize progress tracker
%
fID = fopen('main_misspec.txt','a+');
fprintf(fID,'+----------+----------+----------------+-----------+----------------+\n');
fprintf('+----------+----------+----------------+-----------+----------------+\n')

toc


%%
% Figures
%

% Figure 3b
%
figure(3)
subplot(1,3,2)
bins = linspace(-0.7,0.7,24);
m = 1;
histogram(th_oAdv(:,m),bins,'Normalization','pdf')
hold on
histogram(th_qMLE,bins,'Normalization','pdf')
hold off
ylim([0 6])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('n=%d, m=%d', n, M(m)))
legend(sprintf('oAdv (%.2f)', sqrt(n)*std(th_oAdv(:,m))), sprintf('qMLE (%.2f)', sqrt(n)*std(th_qMLE)))
legend('boxoff')

% Figure 3c
%
figure(3)
subplot(1,3,3)
bins = linspace(-0.7,0.7,24);
m = 1;
histogram(th_AdvL(:,m),bins,'Normalization','pdf')
hold on
histogram(th_qMLE,bins,'Normalization','pdf')
hold off
ylim([0 6])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('n=%d, m=%d', n, M(m)))
legend(sprintf('Adv (%.2f)', sqrt(n)*std(th_AdvL(:,m))), sprintf('qMLE (%.2f)', sqrt(n)*std(th_qMLE)))
legend('boxoff')


% Export csv for Figures 3b and 3c
%
writetable(array2table([th_MLE th_qMLE th_oAdv th_AdvL],'VariableNames',...
    {'MLE' 'qMLE' 'oAdv(1)' 'oAdv(2)' 'AdvL(1)' 'AdvL(2)'}),...
    'misspec_estimator.csv')


%%
% Save Matlab workspace and figures
%

seed = rng;
save(sprintf('main_misspec_%s.mat',datetime('now','format','yyyyMMddHHmmss')))

saveas(figure(3),'Figure3.png')
saveas(figure(13),'Figure13a.png')


%%
% Function
%

%
% Classification accuracy with logistic discriminator
%
function [v,beta] = loss(X1,X2)

    n = size(X1,1); m = size(X2,1);
    [beta,v,~] = fminsearch(@(b) -(mean(loglogisticcdf([ones(n,1) X1 X1.^2 softplus(X1)]*b))+mean(loglogisticcdf(-[ones(m,1) X2 X2.^2 softplus(X2)]*b))),zeros(4,1));
    v = -v;

end
