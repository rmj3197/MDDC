.. _statin49_cluster_idx:

`statin49`: FDA statin dataset with 49 adverse events with Cluster Index
==========================================================================

Description
-----------

A 50 by 7 data matrix of a contingency table processed from FDA Adverse Event Reporting System (FAERS) database from the third quarter of 2014 to the fourth quarter of 2020. This dataset also contains the cluster index of the various adverse events. 

Format
------

A data matrix with 50 rows and 7 columns.

Details
-------

A 50 by 7 data matrix of a contingency table processed from FDA Adverse Event Reporting System (FAERS) database from the third quarter of 2014 to the fourth quarter of 2020.

The 49 rows represent 49 important adverse events associated with the statin class, with the final row aggregating the remaining 5,990 adverse events. The 49 AEs are classified into three clusters:

1. AEs associated with signs and symptoms of muscle injury,
2. AEs associated with laboratory tests for muscle injury, and
3. AEs associated with kidney injury and its laboratory diagnosis and treatment.

The 7 columns include the six statin medications and an aggregate column for ``other drugs`` that also had occurrences of the 6,039 adverse events. The marginal totals for each drug column in the statin49 table are as follows: 197,390 for Atorvastatin, 5,742 for Fluvastatin, 3,230 for Lovastatin, 22,486 for Pravastatin, 122,450 for Rosuvastatin, 85,445 for Simvastatin, and 63,539,867 for Other drugs.

Data Source
------------

https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-latest-quarterly-data-files

