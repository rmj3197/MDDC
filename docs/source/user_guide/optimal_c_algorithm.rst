.. _optimalc_alg:

Data-driven Approach to Determine Optimal Coefficient Value for MDDC Boxplot Method for Controlling FDR
=========================================================================================================

This algorithm describes a method to determine the value of `c` in the cutoff formula :math:`Q_3 + c \times IQR` for controlling the False Discovery Rate (FDR) using the MDDC Boxplot method.

**Steps:**

1. For a given contingency table of dimension :math:`I \times J` calculate the :math:`n_{\cdot \cdot}`, and the :math:`\undertilde{p} = p_{11}, p_{12}, \ldots p_{IJ}`.

2. Generate a large number of :math:`I\times J$` tables :math:`r=1,\ldots,R` under the assumption of independence from multinomial distribution using the :math:`n_{\cdot \cdot}` and :math:`\undertilde{p}` determined in Step 1.

3. Compute the standardized Pearson residuals.

4. Compute the upper limits of the boxplot statistic with :math:`c = 1.5`, and calculate the FDR.

5. If :math:`FDR < 0.05`, stop. Otherwise, if :math:`FDR > 0.05`, use a grid search to find the optimal `c` such that :math:`FDR \leq 0.05`.