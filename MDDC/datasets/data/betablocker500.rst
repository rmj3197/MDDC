.. _betablocker500:

FDA dataset for beta blockers with 500 adverse events
=====================================================

Description
-----------

A 501 by 9 data matrix of a contingency table processed from the FDA Adverse Event Reporting System (FAERS) database. 
This dataset covers a specific period from Q1 2021 to Q4 2023.

Format
------

A data matrix with 501 rows and 9 columns.

Details
-------

A 501 by 9 data matrix of a contingency table from the FDA Adverse Event Reporting System (FAERS) database, covering the period from Q1 2021 to Q4 2023.
The 500 rows correspond to the Adverse Events (AEs) with the highest overall frequency (row marginals) reported during the period, and 1 row for Other AEs.
The reported AEs - "Off label use" and "Drug ineffective" have been excluded.
The dataset includes the following 9 columns: Acebutolol, Atenolol, Bisoprolol, Carvedilol, Metoprolol, Nadolol, Propranolol, Timolol, and Other.

The marginal totals for each column are as follows:

- Acebutolol: 62,164
- Atenolol: 36,619
- Bisoprolol: 134,297
- Carvedilol: 35,922
- Metoprolol: 88,387
- Nadolol: 11,191
- Propranolol: 56,444
- Timolol: 16,077
- Other: 76,926,859

Also refer to the supplementary material of:

Ding, Y., Markatou, M., & Ball, R. (2020). An evaluation of statistical approaches to postmarketing surveillance. Statistics in Medicine, 39(7), 845-874.

for the data generation process.

Data Source
------------

The quarterly files can be found at `FDA FAERS QDE <https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html>`_.


