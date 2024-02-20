# Spanning_multi_asset_payoffs

This github repository provides code and technical implementation specification of the paper "Spanning Multi-Asset Payoffs With ReLUs" by Sébastien Bossu, Stéphane Crépey, Nisrine Madhar and Hoang-Dung Nguyen.

## Overview

`fig` and `result` folders contain some figures and results of our numerical experiments. 

`paper` folder contains the current version of the paper and its complementary material.

`DC_theoretical_test.ipynb` notebook implements and compares the pratical solution of the spanning problem for the dispersion call to the theoretical formulas deduced in Section 4 of the paper. Other Python scripts are detailed as below

- `data_generator.py` contains an object to generate spanning data with different sampling techniques;
- `model.py` contains all spanning strategies;
- `payoff_spec.py` stores the data sampling specification of each payoff;
- `spanningengine.py` defines spanning engine;
- `tool.py` contains some useful functions;
- run `run_file.py` to implement the spanning for a selected payoff in dimension $d=2,3,4,5,20,50$;
- run `run_file_nbbasket.py` to implement the spanning for a selected payoff in dimension $d=2,3,4,5$ while varying the number of spanning baskets.

