-- =============================================================================
-- BigQuery SQL: Sepsis-3 Cohort Creation (sepsis_cohort.sql)
-- =============================================================================
-- Target Schema: `mimic-research-490610.sepsis_cohort`
-- Source Schema: `physionet-data.mimiciv_3_1_icu`, `physionet-data.mimiciv_3_1_hosp`
--
-- This script builds the Sepsis-3 cohort from raw tables, bypassing the derived
-- dataset. Sepsis-3 is defined as a suspected infection (consecutive antibiotic
-- and culture collection) combined with an acute increase in SOFA score >= 2.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS `mimic-research-490610.sepsis_cohort`
OPTIONS(location="us");

-- ---------------------------------------------------------------------------
-- CREATE THE FINAL COHORT TABLE USING MULTIPLE CTEs
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE `mimic-research-490610.sepsis_cohort.sepsis_cohort` AS

-- Step 1: Define Antibiotic Administrations (by GSN or common names)
WITH abx_events AS (
  SELECT 
    hadm_id,
    subject_id,
    starttime AS abx_time
  FROM `physionet-data.mimiciv_3_1_hosp.prescriptions`
  WHERE 
    gsn IN ('002542','002543','007371','008873','008877','008879','008880','008935','008941',
      '008942','008943','008944','008983','008984','008990','008991','008992','008995','008996',
      '008998','009043','009046','009065','009066','009136','009137','009162','009164','009165',
      '009171','009182','009189','009213','009214','009218','009219','009221','009226','009227',
      '009235','009242','009263','009273','009284','009298','009299','009310','009322','009323',
      '009326','009327','009339','009346','009351','009354','009362','009394','009395','009396',
      '009509','009510','009511','009544','009585','009591','009592','009630','013023','013645',
      '013723','013724','013725','014182','014500','015979','016368','016373','016408','016931',
      '016932','016949','018636','018637','018766','019283','021187','021205','021735','021871',
      '023372','023989','024095','024194','024668','025080','026721','027252','027465','027470',
      '029325','029927','029928','037042','039551','039806','040819','041798','043350','043879',
      '044143','045131','045132','046771','047797','048077','048262','048266','048292','049835',
      '050442','050443','051932','052050','060365','066295','067471')
     OR LOWER(drug) LIKE '%penicillin%' OR LOWER(drug) LIKE '%amoxicillin%' OR LOWER(drug) LIKE '%ampicillin%' OR LOWER(drug) LIKE '%piperacillin%' OR LOWER(drug) LIKE '%cef%' OR LOWER(drug) LIKE '%ciprofloxacin%' OR LOWER(drug) LIKE '%levofloxacin%' OR LOWER(drug) LIKE '%vancomycin%' OR LOWER(drug) LIKE '%meropenem%' OR LOWER(drug) LIKE '%imipenem%' OR LOWER(drug) LIKE '%gentamicin%' OR LOWER(drug) LIKE '%tobramycin%' OR LOWER(drug) LIKE '%azithromycin%' OR LOWER(drug) LIKE '%erythromycin%' OR LOWER(drug) LIKE '%clarithromycin%' OR LOWER(drug) LIKE '%clindamycin%' OR LOWER(drug) LIKE '%linezolid%' OR LOWER(drug) LIKE '%metronidazole%'
),

-- Step 2: Define Microbiology Culture Events
culture_events AS (
  SELECT 
    hadm_id,
    subject_id,
    COALESCE(charttime, CAST(chartdate AS DATETIME)) AS culture_time
  FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents`
),

-- Step 3: Find suspected infection onset based on consecutive culture and antibiotic times
suspicion_infection AS (
  SELECT 
    a.hadm_id,
    a.subject_id,
    CASE 
      WHEN a.abx_time <= c.culture_time THEN a.abx_time
      ELSE c.culture_time
    END AS suspected_infection_time
  FROM abx_events a
  INNER JOIN culture_events c
    ON a.hadm_id = c.hadm_id
  WHERE 
    (c.culture_time >= a.abx_time AND DATETIME_DIFF(c.culture_time, a.abx_time, HOUR) <= 24)
    OR
    (a.abx_time >= c.culture_time AND DATETIME_DIFF(a.abx_time, c.culture_time, HOUR) <= 72)
),
first_suspicion AS (
  SELECT 
    hadm_id,
    MIN(suspected_infection_time) AS suspected_infection_time
  FROM suspicion_infection
  GROUP BY hadm_id
),

-- Step 4: Define ICU stay base population
icu_base AS (
  SELECT 
    i.subject_id, 
    i.hadm_id, 
    i.stay_id, 
    i.intime, 
    i.outtime, 
    i.los,
    (EXTRACT(YEAR FROM i.intime) - p.anchor_year + p.anchor_age) AS age,
    p.gender
  FROM `physionet-data.mimiciv_3_1_icu.icustays` AS i
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` AS p
    ON p.subject_id = i.subject_id
  WHERE (EXTRACT(YEAR FROM i.intime) - p.anchor_year + p.anchor_age) >= 18
    AND i.los >= 1.0
),

-- Step 5: Extract CNS GCS Score Component
sofa_cns AS (
  SELECT 
    stay_id,
    MIN(valuenum) AS min_gcs
  FROM `physionet-data.mimiciv_3_1_icu.chartevents`
  WHERE itemid = 220739 AND valuenum IS NOT NULL AND valuenum > 0
  GROUP BY stay_id
),

-- Step 6: Extract Coagulation Platelets Component
sofa_coag AS (
  SELECT 
    la.stay_id,
    MIN(l.valuenum) AS min_platelets
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
  INNER JOIN icu_base la ON la.hadm_id = l.hadm_id
  WHERE l.itemid = 51265 AND l.valuenum IS NOT NULL AND l.valuenum > 0
  GROUP BY la.stay_id
),

-- Step 7: Extract Liver Bilirubin Component
sofa_liver AS (
  SELECT 
    la.stay_id,
    MAX(l.valuenum) AS max_bilirubin
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
  INNER JOIN icu_base la ON la.hadm_id = l.hadm_id
  WHERE l.itemid = 50885 AND l.valuenum IS NOT NULL AND l.valuenum > 0
  GROUP BY la.stay_id
),

-- Step 8: Extract Renal Creatinine Component
sofa_renal AS (
  SELECT 
    la.stay_id,
    MAX(l.valuenum) AS max_creatinine
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
  INNER JOIN icu_base la ON la.hadm_id = l.hadm_id
  WHERE l.itemid = 50912 AND l.valuenum IS NOT NULL AND l.valuenum > 0
  GROUP BY la.stay_id
),

-- Step 9: Extract Cardiovascular MAP and Vasopressors Component
sofa_map AS (
  SELECT 
    stay_id,
    MIN(valuenum) AS min_map
  FROM `physionet-data.mimiciv_3_1_icu.chartevents`
  WHERE itemid IN (220052, 220181, 225312) AND valuenum IS NOT NULL AND valuenum > 0
  GROUP BY stay_id
),
sofa_vaso AS (
  SELECT DISTINCT 
    stay_id,
    1 AS vaso_flag
  FROM `physionet-data.mimiciv_3_1_icu.inputevents`
  WHERE itemid IN (221906, 221289, 221662, 221653) AND rate > 0
),

-- Step 10: Extract Respiration PaO2 and FiO2 Component
sofa_pao2 AS (
  SELECT 
    la.stay_id,
    MIN(l.valuenum) AS min_pao2
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
  INNER JOIN icu_base la ON la.hadm_id = l.hadm_id
  WHERE l.itemid = 50821 AND l.valuenum IS NOT NULL AND l.valuenum > 0
  GROUP BY la.stay_id
),
sofa_fio2 AS (
  SELECT 
    stay_id,
    MAX(valuenum) AS max_fio2
  FROM `physionet-data.mimiciv_3_1_icu.chartevents`
  WHERE itemid = 223835 AND valuenum IS NOT NULL AND valuenum > 0
  GROUP BY stay_id
),

-- Step 11: Compute Combined SOFA Scores
sofa_combined AS (
  SELECT 
    i.stay_id,
    -- CNS Score
    CASE 
      WHEN c.min_gcs < 6 THEN 4
      WHEN c.min_gcs <= 9 THEN 3
      WHEN c.min_gcs <= 12 THEN 2
      WHEN c.min_gcs <= 14 THEN 1
      ELSE 0
    END AS sofa_cns_score,
    -- Coagulation Score
    CASE 
      WHEN pl.min_platelets < 20 THEN 4
      WHEN pl.min_platelets < 50 THEN 3
      WHEN pl.min_platelets < 100 THEN 2
      WHEN pl.min_platelets < 150 THEN 1
      ELSE 0
    END AS sofa_coag_score,
    -- Liver Score
    CASE 
      WHEN li.max_bilirubin >= 12.0 THEN 4
      WHEN li.max_bilirubin >= 6.0 THEN 3
      WHEN li.max_bilirubin >= 2.0 THEN 2
      WHEN li.max_bilirubin >= 1.2 THEN 1
      ELSE 0
    END AS sofa_liver_score,
    -- Cardiovascular Score
    CASE 
      WHEN v.vaso_flag = 1 THEN 2 -- Minimum score for vasopressor administration
      WHEN m.min_map < 70 THEN 1
      ELSE 0
    END AS sofa_cv_score,
    -- Renal Score
    CASE 
      WHEN re.max_creatinine >= 5.0 THEN 4
      WHEN re.max_creatinine >= 3.5 THEN 3
      WHEN re.max_creatinine >= 2.0 THEN 2
      WHEN re.max_creatinine >= 1.2 THEN 1
      ELSE 0
    END AS sofa_renal_score,
    -- Respiration Score (estimated PaO2/FiO2 ratio)
    CASE 
      WHEN f.max_fio2 IS NOT NULL AND pa.min_pao2 IS NOT NULL THEN
        CASE 
          WHEN (pa.min_pao2 / (f.max_fio2 / 100.0)) < 100 THEN 4
          WHEN (pa.min_pao2 / (f.max_fio2 / 100.0)) < 200 THEN 3
          WHEN (pa.min_pao2 / (f.max_fio2 / 100.0)) < 300 THEN 2
          WHEN (pa.min_pao2 / (f.max_fio2 / 100.0)) < 400 THEN 1
          ELSE 0
        END
      ELSE 0
    END AS sofa_resp_score
  FROM icu_base i
  LEFT JOIN sofa_cns c ON c.stay_id = i.stay_id
  LEFT JOIN sofa_coag pl ON pl.stay_id = i.stay_id
  LEFT JOIN sofa_liver li ON li.stay_id = i.stay_id
  LEFT JOIN sofa_renal re ON re.stay_id = i.stay_id
  LEFT JOIN sofa_map m ON m.stay_id = i.stay_id
  LEFT JOIN sofa_vaso v ON v.stay_id = i.stay_id
  LEFT JOIN sofa_pao2 pa ON pa.stay_id = i.stay_id
  LEFT JOIN sofa_fio2 f ON f.stay_id = i.stay_id
),

-- Step 12: Assemble final cohort with sepsis flags
cohort_labeled AS (
  SELECT 
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime,
    i.outtime,
    i.los,
    i.age,
    i.gender,
    sp.suspected_infection_time,
    -- Sepsis-3 = Suspected Infection AND Max SOFA Score >= 2
    CASE 
      WHEN sp.suspected_infection_time IS NOT NULL 
           AND (COALESCE(sc.sofa_cns_score, 0) + 
                COALESCE(sc.sofa_coag_score, 0) + 
                COALESCE(sc.sofa_liver_score, 0) + 
                COALESCE(sc.sofa_cv_score, 0) + 
                COALESCE(sc.sofa_renal_score, 0) + 
                COALESCE(sc.sofa_resp_score, 0)) >= 2 THEN 1
      ELSE 0
    END AS sepsis_label,
    -- Onset time is defined as suspected_infection_time if patient has Sepsis-3
    CASE 
      WHEN sp.suspected_infection_time IS NOT NULL 
           AND (COALESCE(sc.sofa_cns_score, 0) + 
                COALESCE(sc.sofa_coag_score, 0) + 
                COALESCE(sc.sofa_liver_score, 0) + 
                COALESCE(sc.sofa_cv_score, 0) + 
                COALESCE(sc.sofa_renal_score, 0) + 
                COALESCE(sc.sofa_resp_score, 0)) >= 2 THEN sp.suspected_infection_time
      ELSE NULL
    END AS sepsis_onset_time
  FROM icu_base i
  LEFT JOIN first_suspicion sp ON sp.hadm_id = i.hadm_id
  LEFT JOIN sofa_combined sc ON sc.stay_id = i.stay_id
)

-- Step 13: Filter and apply exclusions (sepsis onset after 6 hours, or never-sepsis controls)
SELECT 
  subject_id,
  hadm_id,
  stay_id,
  intime,
  outtime,
  los,
  age,
  gender,
  sepsis_label,
  sepsis_onset_time
FROM cohort_labeled
WHERE 
  -- Sepsis Cases: Must develop sepsis after 6 hours from ICU admission
  (sepsis_label = 1 AND DATETIME_DIFF(sepsis_onset_time, intime, HOUR) > 6)
  OR
  -- Controls: Never develop sepsis during stay
  (sepsis_label = 0);
