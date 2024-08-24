.. _mddc_algorithm:

Modified Detecting Deviating Cells (MDDC) Algorithm
====================================================

The Modified Detecting Deviating Cells (MDDC) algorithm is described as follows:

1. **Standardized Pearson Residual Calculation**

   For each cell in the contingency table, compute the standardized Pearson residual:
   
   .. math::

      e_{ij} = \frac{n_{ij} - \frac{n_{i\bullet} n_{\bullet j}}{n_{\bullet \bullet}}}{\sqrt{\frac{n_{i\bullet} n_{\bullet j}}{n_{\bullet \bullet}} \left(1 - \frac{n_{i\bullet}}{n_{\bullet \bullet}}\right) \left(1 - \frac{n_{\bullet j}}{n_{\bullet \bullet}}\right)}}

2. **Separating Residuals and Determining Cutoff Values**

   Separate the set of residuals into two groups:
   
      - :math:`\{e^+_{ij}\}` for cells with :math:`n_{ij} > 0`
      - :math:`\{e^0_{ij}\}` for cells with :math:`n_{ij} = 0`
   
   The boxplot statistics are used as cutoff values for detecting the first set of outlying cells:
   
   .. math::
   
      c_{univ,j^*}^0 = Q_1(\{e^0_{ij^*}\}) - 1.5 \times IQR(\{e^0_{ij^*}\})

      c_{univ,j^*}^+ = Q_3(\{e^+_{ij^*}\}) + 1.5 \times IQR(\{e^+_{ij^*}\})

   Define a matrix :math:`\mathbf{U}` with entries:
   
   .. math::
   
      u_{ij} = 
      \begin{cases}
      e^+_{ij} & \text{if } |e^+_{ij}| \leq c^+_{univ,j} \\
      \text{NA} & \text{if } |e^+_{ij}| > c^+_{univ,j} \\
      e^0_{ij} & \text{if } e^0_{ij} \geq c^0_{univ,j} \\
      \text{NA} & \text{if } e^0_{ij} < c^0_{univ,j}
      \end{cases}
   
   Cells with :math:`e^+_{ij} > c^+_{univ}` are labeled as signals.

3. **Correlation and Connection Determination**

   For any two AE rows :math:`i \neq k`, compute their Pearson correlation:
   
   .. math::
   
      cor_{ik} = Corr(\tilde{u}_{i}, \tilde{u}_{k})

   where :math:`\tilde{u}_{i} = (u_{i1}, u_{i2}, \ldots, u_{iJ})` and :math:`\tilde{u}_{k} = (u_{k1}, u_{k2}, \ldots, u_{kJ})`. Let :math:`c_{corr}` be the correlation threshold. For the :math:`i`-th AE, if :math:`|cor_{ik}| \geq c_{corr}` for :math:`k \neq i`, then AE :math:`k` is called a "connected" AE to AE :math:`i`.

4. **Prediction and Weighted Mean Calculation**

   Obtain predicted values for each cell based on the connected AEs. Suppose there are :math:`m` connected AEs for AE :math:`i`, with correlations :math:`cor_{ik_1}, cor_{ik_2}, \ldots, cor_{ik_m}`. For each connected AE :math:`k`, fit a simple linear regression with intercept :math:`\alpha_{ik}` and slope :math:`\beta_{ik}` using :math:`\tilde{u}_{i}` as the response variable and :math:`\tilde{u}_{k}` as the explanatory variable. The fitted value for AE :math:`i` based on AE :math:`k` is:

   .. math::
   
      \hat{u}_{ikj} = \alpha_{ik} + \beta_{ik} u_{kj}

   The fitted value for AE :math:`i` based on all the connected AEs is obtained as a weighted mean:

   .. math::
   
      \hat{u}_{ij} = \sum_{k=1}^{m} w_{ik} \hat{u}_{ikj}

   where :math:`w_{ik} = \frac{|cor_{ik}|}{\sum_{l=k_1}^{k_m} |cor_{il}|}`.

5. **Final Residual Calculation and P-value**

   Compute:

   .. math::
   
      r_{ij} = \frac{(e_{ij} - \hat{u}_{ij}) - A_j}{\sqrt{B_j}}

   where:

   .. math::
   
      A_j = \frac{1}{I_j} \sum_{i \in I_j} (e_{ij} - \hat{u}_{ij})

      B_j = \frac{1}{I_j} \sum_{i \in I_j} \left[(e_{ij} - \hat{u}_{ij}) - A_j\right]^2

   The computation is over all :math:`i` within drug :math:`j` where neither :math:`e_{ij}` nor :math:`\hat{u}_{ij}` is NA. 
   Calculate the upper tail probability of :math:`r_{ij}` in the standard normal distribution, which is used as the p-value for each cell. Obtain adjusted p-values via the Benjamini-Hochberg procedure to control the false discovery rate. The second set of signals are the cells with adjusted p-values less than 0.05.
