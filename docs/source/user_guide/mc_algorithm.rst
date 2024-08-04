.. _mc_algorithm:

Monte Carlo Method for Obtaining the Cutoff Value
==================================================

Monte Carlo simulation method for obtaining the cutoff value :math:`c_{univ,j}^+` in Step 2 of the MDDC method. 

1. Obtain the marginals :math:`n_{1\bullet}, \ldots, n_{I\bullet}, n_{\bullet 1}, \ldots, n_{\bullet J}` from the original :math:`I \times J` contingency table.

2. Under the assumption of no association between drugs and AEs (i.e., independence of rows and columns), compute cell probabilities

.. math::

    p_{ij} = \left(\frac{n_{i\bullet}}{n_{\bullet \bullet}}\right)\left(\frac{n_{\bullet j}}{n_{\bullet \bullet}}\right),

where :math:`i = 1, \ldots, I` and :math:`j = 1, \ldots, J`.

3. Generate 10,000 :math:`I \times J` contingency tables with the above specified marginals and cell probabilities :math:`\{p_{ij}\}` through multinomial distribution

.. math::

    (n_{11}, n_{12}, \ldots, n_{IJ}) \sim \text{Multinomial}(n_{\bullet \bullet}, p),

where :math:`p = (p_{11}, p_{12}, \ldots, p_{IJ})^T`.

4. For the :math:`r`-th simulated table, :math:`r = 1, \ldots, 10000`, compute :math:`e_{ij}` for all the cells in the table, and obtain

.. math::

    m_{j,r} = \max_{1 \leq i \leq I} e_{ij} \times \mathbf{1}\{n_{ij}>5\}

for :math:`j = 1, \ldots, J`. For each drug :math:`j`, this will provide :math:`m_{j,1}, m_{j,2}, \ldots, m_{j,10000}`.

5. For each drug :math:`j`, obtain the cutoff value :math:`c_{univ,j}^+` as the 95-th quantile by ordering :math:`m_{j,1}, m_{j,2}, \ldots, m_{j,10000}` from smallest to largest.
