.. _optimalc_alg:
Data-driven Approach to Determine Optimal Coefficient Value for MDDC Boxplot Method for Controlling FDR
=========================================================================================================

This algorithm describes a method to determine the value of `c` in the cutoff formula 
:math:`Q_3 + c \times IQR` for controlling the False Discovery Rate (FDR) using the MDDC Boxplot method.

**Steps:**

1. Generate a large number of :math:`I × J` tables under the assumption of independence (:math:`\lambda_{ij} = 1`).
   
2. Compute the standardized Pearson residuals.

3. Compute the upper limits with :math:`c = 1.5`, and calculate the FDR.

4. If :math:`FDR < 0.05`, stop. Otherwise, if :math:`FDR > 0.05`, use a grid search to find the optimal `c` such that :math:`FDR \leq 0.05`.
