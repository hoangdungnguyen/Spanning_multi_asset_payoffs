# Spanning_multi_asset_payoffs

This github repository contains code and technical implementation specification of the paper "Spanning Multi-Asset Payoffs With ReLUs" by Sébastien Bossu, Stéphane Crépey, Nisrine Madhar and Hoang-Dung Nguyen.

## Data Sampling

We use synthetic data to generate our training and test sets, respectively denoted as $\mathcal{D}^{train}$ and $\mathcal{D}^{test}$ with sizes $n^{train}, n^{test}$. Each data point is a bundle of asset performances and corresponding payoff value $\left (\vec x, F(\vec x) \right) \in \R^{d+1}$. Table \ref{tab:training-specs} provides the regular grid sampling specifications used to generate the training and test sets in low dimension $d=2$ to 5.  This approach ensures there are no gaps in sampling and facilitates comparisons between option payoffs.  Note that some payoffs require a different $\vec x$-sampling range to curb the number of zero payoff values.  Estimates are evaluated on a test set of 50,000 to 200,000  points drawn uniformly in the same range as the training grid.

