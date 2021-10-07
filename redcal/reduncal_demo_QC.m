%% Simple demonstration of redundancy calibration for QC project
%
% This script provides a simple demonstration of redundancy calibration
% using a 5-element uniform linear array. It uses conventional redundancy
% calibration based on the logarithm of the elements of the measured array
% covariance matrix (the visibilities) as that allows to estimate the gain
% magnitudes and phases directly without resorting to iterative estimateion
% of the true values of the measured visibilites and the gains. The latter
% approach is more robust to low-SNR scenarios, but is more complicated and
% therefore less suitable for the initial exploration of QC approaches to
% radio interferometric calibration. Both implementations of redundancy
% calibration are described in [1].
%
% The basic idea of redundancy calibration is that identical baselines
% should measure the same visibility. For a 5-element uniform linear array,
% there are 3 distinct types of redundant baselines:
%
% array configuration   x   x   x   x   x
% baseline type 1        <-> <-> <-> <->
% baseline type 2        <-----> <----->
%                            <----->
%                                <----->
% baseline type 3        <--------->
%                            <--------->
%
% Each element of the array covariance matrix is a product of two gains and
% the unperturbed value of the corresponding visibility. Conventional
% redundancy calibration takes the logarithm to transform this product of
% factors into a summation of terms, resulting in a very simple linear
% relationship with a sparse measurement matrix.
%
% Reference
% [1]   Stefan J. Wijnholds and Parisa Noorishad, "Statistically optimal
% self-calibration of regular imaging arrays", European Signal Processing
% Conference (EuSiPCo), 2012.
%
% Stefan J. Wijnholds, October 7, 2021

%% simulation setup

% start with a clean workspace
clear
close all

% antenna positions in m of 5-element uniform linear array as column vector
xpos = (-2:1:2).';
% number of antennas
% Hardcoded as the measurement matrix will be hardcoded for this example as
% well, so we want to trigger an error when an inconsistent simulation
% setup is defined.
Nant = 5;

% source positions as column vector of directional cosines
% A 3-source model is used to ensure that different baselines have
% different true visibilities. This may not be true for a 1-source model.
l = [-0.5; 0.2; 0.7];

% source powers as column vector
sigma = [0.8; 1; 0.4];

% gain column vector
% This is what we like to estimate
g = 1 + 0.3 * (randn(Nant, 1) + 1i * randn(Nant, 1));

% geometrical delays of source signals over the array
freq = 150e6;   % measurement frequency in MHz
c = 2.99792e8;  % speed of light in m/s
A = exp(-(2 * pi * 1i * freq / c) * (xpos * l.'));

% noise-free "measured" visibilities
R = diag(g) * A * diag(sigma) * A' * diag(g)';

%% perform redundancy calibration of gain magnitudes

% measurement matrix for gain magnitudes
Mmag = [1 1 0 0 0 1 0 0; ... % baseline type 1 (4 rows)
        0 1 1 0 0 1 0 0; ...
        0 0 1 1 0 1 0 0; ...
        0 0 0 1 1 1 0 0; ...
        1 0 1 0 0 0 1 0; ... % baseline type 2 (3 rows)
        0 1 0 1 0 0 1 0; ...
        0 0 1 0 1 0 1 0; ...
        1 0 0 1 0 0 0 1; ... % baseline type 3 (1 row)
        0 1 0 0 1 0 0 1; ...
        1 0 0 0 0 0 0 0];    % magnitude constriant
% This measurement matrix applies to the elements in the upper triangle of
% the array covariance matrix to which redundancy is applicable. The
% following selection matrix selects the appropriate elements from the
% array covariance matrix
sel = [6, 12, 18, 24, 11, 17, 23, 16, 22].';

% solve for gain magnitudes
theta = Mmag \ [log10(abs(R(sel))); 0];
gmag = 10.^theta(1:Nant);

% show comparison
figure
% Normalise true gain values to match constraint that the gain of the first
% element is unity
gmag_true = abs(g) / abs(g(1));
plot(1:5, gmag_true, 'b-', 1:5, gmag, 'ro');
set(gca, 'FontSize', 16);
xlabel('antenna index');
ylabel('gain magnitude');
legend('true gain', 'estimated gain');

%% perform redundancy calibration of gain phases

% measurement matrix for gain phases
Mph = [ 1 -1  0  0  0  1  0  0; ... % baseline type 1 (4 rows)
        0  1 -1  0  0  1  0  0; ...
        0  0  1 -1  0  1  0  0; ...
        0  0  0  1 -1  1  0  0; ...
        1  0 -1  0  0  0  1  0; ... % baseline type 2 (3 rows)
        0  1  0 -1  0  0  1  0; ...
        0  0  1  0 -1  0  1  0; ...
        1  0  0 -1  0  0  0  1; ... % baseline type 3 (1 row)
        0  1  0  0 -1  0  0  1; ...
        0  0  1  0  0  0  0  0; ... % phase constraint on first element
       [xpos.' 0 0 0]];             % phase gradient constraint

% solve for gain magnitudes
theta = Mph \ [angle(R(sel)); 0; 0];
gph = theta(1:Nant);

% show comparison
figure
% apply constraint that central element is phase reference
gph_true = angle(g) - angle(g(3));
% find (arbitrary) phase gradient ambiguity
theta = xpos \ (gph_true - gph);
% correct true gains for this gradient ambiguity
gph_true = gph_true - theta * xpos;
plot(1:5, gph_true, 'b-', 1:5, gph, 'ro');
set(gca, 'FontSize', 16);
xlabel('antenna index');
ylabel('gain phases (rad)');
legend('true gain', 'estimated gain');
