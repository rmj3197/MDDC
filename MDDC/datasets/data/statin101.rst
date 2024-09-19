.. _statin101:

FDA statin dataset with 101 adverse events
==========================================

Description
-----------

A 102 by 5 data matrix of a contingency table processed from the FDA Adverse Event Reporting System (FAERS) database from Q1 2021 to Q4 2023.

Format
------

A data matrix with 102 rows and 5 columns.

Details
-------

A 102 by 5 data matrix of a contingency table from the FDA Adverse Event Reporting System (FAERS) database, covering Q1 2021 to Q4 2023.

The 101 rows correspond to the adverse events (AEs) with the highest overall frequency (row marginals) reported during the period. 
The reported AEs, "Off label use" and "Drug ineffective", have been excluded.

The 5 columns include 4 statin medications and an "other" column. Marginal totals for each drug: 101,462 for Atorvastatin, 
9,203 for Fluvastatin, 130,994 for Rosuvastatin, 87,841 for Simvastatin, and 5,739,383 for Other drugs.

Also refer to the supplementary material of:

Ding, Y., Markatou, M., & Ball, R. (2020). An evaluation of statistical approaches to postmarketing surveillance. Statistics in Medicine, 39(7), 845-874.

for the data generation process.

Data Source
------------

The quarterly files can be found at `FDA FAERS QDE <https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html>`_.


