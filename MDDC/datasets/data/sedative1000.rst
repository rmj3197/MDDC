.. _sedative1000:

FDA dataset for sedatives with 1000 adverse events
==================================================

Description
-----------

A 1001 by 11 data matrix of a contingency table processed from the FDA Adverse Event Reporting System (FAERS) database. This dataset covers a specific period from Q1 2021 to Q4 2023.

Format
------

A data matrix with 1001 rows and 11 columns.

Details
-------

A 1001 by 11 data matrix of a contingency table from the FDA Adverse Event Reporting System (FAERS) database, covering a specified period from Q1 2021 to Q4 2023.

The 1000 rows correspond to the adverse events (AEs) with the highest overall frequency (row marginals) reported during the period, and 1 row for Other AEs. The reported AEs, "Off label use" and "Drug ineffective", have been excluded.

The dataset includes the following 10 columns: Clonazepam, Dexmedetomidine, Diazepam, Diphenhydramine, Doxepin, Lorazepam, Midazolam, Mirtazapine, Nitrazepam, Temazepam, and an Other column.

The marginal totals for each column are as follows:  
- Clonazepam: 110,453  
- Dexmedetomidine: 4,262  
- Diazepam: 74,859  
- Diphenhydramine: 13,465  
- Doxepin: 11,795  
- Lorazepam: 101,969  
- Midazolam: 26,264  
- Mirtazapine: 54,273  
- Nitrazepam: 3,473  
- Temazepam: 20,523  
- Other: 77,487,518

Also refer to the supplementary material of:

Ding, Y., Markatou, M., & Ball, R. (2020). An evaluation of statistical approaches to postmarketing surveillance. Statistics in Medicine, 39(7), 845-874.

for the data generation process.

Data Source
------------

The quarterly files can be found at `FDA FAERS QDE <https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html>`_.