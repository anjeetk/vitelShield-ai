-- =============================================================================
-- BigQuery SQL: Sepsis Feature Extraction & Aggregation (sepsis_features.sql)
-- =============================================================================
-- Target Schema: `mimic-research-490610.sepsis_cohort`
-- Source Schema: `physionet-data.mimiciv_3_1_icu`, `physionet-data.mimiciv_3_1_hosp`
--
-- This script creates both:
--   1. Sequential hourly features (6 rows per stay) for deep learning models:
--      `final_sepsis_seq_features_6h`, `final_sepsis_seq_features_12h`, etc.
--   2. Aggregated tabular features (1 row per stay) for traditional ML models:
--      `final_sepsis_features_6h`, `final_sepsis_features_12h`, etc.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- STEP 1: EXTRACT BASELINE STATIC DEMOGRAPHICS (Weight, Height)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE `mimic-research-490610.sepsis_cohort.sepsis_static_vars` AS
SELECT
  stay_id,
  AVG(CASE WHEN itemid IN (224639, 226512) THEN valuenum END) AS weight,
  AVG(CASE WHEN itemid = 226730 THEN valuenum END) AS height
FROM `physionet-data.mimiciv_3_1_icu.chartevents`
WHERE valuenum IS NOT NULL AND valuenum > 0
  AND itemid IN (224639, 226512, 226730)
GROUP BY stay_id;


-- ===========================================================================
-- HORIZON 1: 6-HOUR PREDICTION TABLES
-- ===========================================================================

-- 1.1. Sequential Hourly Data (6 rows per stay)
CREATE OR REPLACE TABLE `mimic-research-490610.sepsis_cohort.final_sepsis_seq_features_6h` AS
WITH targets AS (
  SELECT 
    stay_id,
    subject_id,
    hadm_id,
    intime,
    age,
    gender,
    sepsis_label,
    sepsis_onset_time,
    CASE 
      WHEN sepsis_label = 1 THEN DATETIME_SUB(sepsis_onset_time, INTERVAL 6 HOUR)
      ELSE DATETIME_ADD(intime, INTERVAL 24 HOUR)
    END AS T_target
  FROM `mimic-research-490610.sepsis_cohort.sepsis_cohort`
),
hour_grid AS (
  SELECT 1 AS hr UNION ALL
  SELECT 2 AS hr UNION ALL
  SELECT 3 AS hr UNION ALL
  SELECT 4 AS hr UNION ALL
  SELECT 5 AS hr UNION ALL
  SELECT 6 AS hr
),
stay_hours AS (
  SELECT 
    t.*,
    hg.hr,
    DATETIME_ADD(DATETIME_SUB(t.T_target, INTERVAL 6 HOUR), INTERVAL (hg.hr - 1) HOUR) AS hr_start,
    DATETIME_ADD(DATETIME_SUB(t.T_target, INTERVAL 6 HOUR), INTERVAL hg.hr HOUR) AS hr_end
  FROM targets t
  CROSS JOIN hour_grid hg
),
vitals_hourly AS (
  SELECT 
    sh.stay_id,
    sh.hr,
    AVG(CASE WHEN ch.itemid = 220045 THEN ch.valuenum END) AS heart_rate,
    AVG(CASE WHEN ch.itemid IN (220179, 220050) THEN ch.valuenum END) AS sbp,
    AVG(CASE WHEN ch.itemid IN (220180, 220051) THEN ch.valuenum END) AS dbp,
    AVG(CASE WHEN ch.itemid IN (220181, 220052, 225312) THEN ch.valuenum END) AS map,
    AVG(CASE WHEN ch.itemid = 220210 THEN ch.valuenum END) AS respiratory_rate,
    AVG(CASE WHEN ch.itemid = 223762 THEN ch.valuenum END) AS temperature,
    AVG(CASE WHEN ch.itemid = 220277 THEN ch.valuenum END) AS spo2
  FROM stay_hours sh
  INNER JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ch
    ON ch.stay_id = sh.stay_id
  WHERE ch.itemid IN (220045, 220179, 220050, 220180, 220051, 220181, 220052, 225312, 220210, 223762, 220277)
    AND ch.valuenum IS NOT NULL
    AND ch.charttime >= sh.hr_start AND ch.charttime < sh.hr_end
  GROUP BY sh.stay_id, sh.hr
),
labs_hourly AS (
  SELECT 
    sh.stay_id,
    sh.hr,
    AVG(CASE WHEN l.itemid IN (50813, 225668) THEN l.valuenum END) AS lactate,
    AVG(CASE WHEN l.itemid IN (50931, 225664, 50809) THEN l.valuenum END) AS glucose,
    AVG(CASE WHEN l.itemid IN (51301, 220546) THEN l.valuenum END) AS wbc,
    AVG(CASE WHEN l.itemid IN (51265, 227457) THEN l.valuenum END) AS platelets,
    AVG(CASE WHEN l.itemid IN (50912, 220615) THEN l.valuenum END) AS creatinine,
    AVG(CASE WHEN l.itemid IN (51006, 225624) THEN l.valuenum END) AS bun,
    AVG(CASE WHEN l.itemid IN (50983, 220645, 50824) THEN l.valuenum END) AS sodium,
    AVG(CASE WHEN l.itemid IN (50971, 227442, 50822) THEN l.valuenum END) AS potassium,
    AVG(CASE WHEN l.itemid IN (50902, 220602, 50806) THEN l.valuenum END) AS chloride,
    AVG(CASE WHEN l.itemid IN (50804, 50882, 227443) THEN l.valuenum END) AS bicarbonate,
    AVG(CASE WHEN l.itemid IN (50885, 225690) THEN l.valuenum END) AS bilirubin,
    AVG(CASE WHEN l.itemid IN (51222, 220228) THEN l.valuenum END) AS hemoglobin,
    AVG(CASE WHEN l.itemid IN (51221, 220545) THEN l.valuenum END) AS hematocrit,
    AVG(CASE WHEN l.itemid IN (50820, 223830) THEN l.valuenum END) AS ph,
    AVG(CASE WHEN l.itemid IN (50821, 220224) THEN l.valuenum END) AS pao2,
    AVG(CASE WHEN l.itemid IN (50818, 220235) THEN l.valuenum END) AS paco2,
    AVG(CASE WHEN l.itemid IN (50802, 224828) THEN l.valuenum END) AS base_excess
  FROM stay_hours sh
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.labevents` l
    ON l.hadm_id = sh.hadm_id
  WHERE l.itemid IN (50813, 225668, 50931, 225664, 50809, 51301, 220546, 51265, 227457, 50912, 220615, 51006, 225624, 50983, 220645, 50824, 50971, 227442, 50822, 50902, 220602, 50806, 50804, 50882, 227443, 50885, 225690, 51222, 220228, 51221, 220545, 50820, 223830, 50821, 220224, 50818, 220235, 50802, 224828)
    AND l.valuenum IS NOT NULL
    AND l.charttime >= sh.hr_start AND l.charttime < sh.hr_end
  GROUP BY sh.stay_id, sh.hr
)
SELECT 
  sh.stay_id AS patientunitstayid,
  sh.hr AS hour_idx,
  sh.sepsis_label AS label,
  sh.age,
  sh.gender,
  s.weight,
  s.height,
  v.heart_rate, v.sbp, v.dbp, v.map, v.respiratory_rate, v.temperature, v.spo2,
  l.lactate, l.glucose, l.wbc, l.platelets, l.creatinine, l.bun, l.sodium, l.potassium, l.chloride, l.bicarbonate, l.bilirubin, l.hemoglobin, l.hematocrit, l.ph, l.pao2, l.paco2, l.base_excess
FROM stay_hours sh
LEFT JOIN `mimic-research-490610.sepsis_cohort.sepsis_static_vars` s
  ON s.stay_id = sh.stay_id
LEFT JOIN vitals_hourly v
  ON v.stay_id = sh.stay_id AND v.hr = sh.hr
LEFT JOIN labs_hourly l
  ON l.stay_id = sh.stay_id AND l.hr = sh.hr;

-- 1.2. Aggregated Tabular Data (1 row per stay)
CREATE OR REPLACE TABLE `mimic-research-490610.sepsis_cohort.final_sepsis_features_6h` AS
SELECT 
  patientunitstayid,
  ANY_VALUE(label) AS label,
  ANY_VALUE(age) AS age,
  ANY_VALUE(gender) AS gender,
  ANY_VALUE(weight) AS weight,
  ANY_VALUE(height) AS height,
  AVG(heart_rate) AS heart_rate_mean, STDDEV(heart_rate) AS heart_rate_std,
  AVG(sbp) AS sbp_mean, STDDEV(sbp) AS sbp_std,
  AVG(dbp) AS dbp_mean, STDDEV(dbp) AS dbp_std,
  AVG(map) AS map_mean, STDDEV(map) AS map_std,
  AVG(respiratory_rate) AS respiratory_rate_mean, STDDEV(respiratory_rate) AS respiratory_rate_std,
  AVG(temperature) AS temperature_mean, STDDEV(temperature) AS temperature_std,
  AVG(spo2) AS spo2_mean, STDDEV(spo2) AS spo2_std,
  AVG(lactate) AS lactate_mean, STDDEV(lactate) AS lactate_std,
  AVG(glucose) AS glucose_mean, STDDEV(glucose) AS glucose_std,
  AVG(wbc) AS wbc_mean, STDDEV(wbc) AS wbc_std,
  AVG(platelets) AS platelets_mean, STDDEV(platelets) AS platelets_std,
  AVG(creatinine) AS creatinine_mean, STDDEV(creatinine) AS creatinine_std,
  AVG(bun) AS bun_mean, STDDEV(bun) AS bun_std,
  AVG(sodium) AS sodium_mean, STDDEV(sodium) AS sodium_std,
  AVG(potassium) AS potassium_mean, STDDEV(potassium) AS potassium_std,
  AVG(chloride) AS chloride_mean, STDDEV(chloride) AS chloride_std,
  AVG(bicarbonate) AS bicarbonate_mean, STDDEV(bicarbonate) AS bicarbonate_std,
  AVG(bilirubin) AS bilirubin_mean, STDDEV(bilirubin) AS bilirubin_std,
  AVG(hemoglobin) AS hemoglobin_mean, STDDEV(hemoglobin) AS hemoglobin_std,
  AVG(hematocrit) AS hematocrit_mean, STDDEV(hematocrit) AS hematocrit_std,
  AVG(ph) AS ph_mean, STDDEV(ph) AS ph_std,
  AVG(pao2) AS pao2_mean, STDDEV(pao2) AS pao2_std,
  AVG(paco2) AS paco2_mean, STDDEV(paco2) AS paco2_std,
  AVG(base_excess) AS base_excess_mean, STDDEV(base_excess) AS base_excess_std
FROM `mimic-research-490610.sepsis_cohort.final_sepsis_seq_features_6h`
GROUP BY patientunitstayid;


-- ===========================================================================
-- HORIZON 2: 12-HOUR PREDICTION TABLES
-- ===========================================================================

-- 2.1. Sequential Hourly Data (6 rows per stay)
CREATE OR REPLACE TABLE `mimic-research-490610.sepsis_cohort.final_sepsis_seq_features_12h` AS
WITH targets AS (
  SELECT 
    stay_id,
    subject_id,
    hadm_id,
    intime,
    age,
    gender,
    sepsis_label,
    sepsis_onset_time,
    CASE 
      WHEN sepsis_label = 1 THEN DATETIME_SUB(sepsis_onset_time, INTERVAL 12 HOUR)
      ELSE DATETIME_ADD(intime, INTERVAL 18 HOUR)
    END AS T_target
  FROM `mimic-research-490610.sepsis_cohort.sepsis_cohort`
),
hour_grid AS (
  SELECT 1 AS hr UNION ALL
  SELECT 2 AS hr UNION ALL
  SELECT 3 AS hr UNION ALL
  SELECT 4 AS hr UNION ALL
  SELECT 5 AS hr UNION ALL
  SELECT 6 AS hr
),
stay_hours AS (
  SELECT 
    t.*,
    hg.hr,
    DATETIME_ADD(DATETIME_SUB(t.T_target, INTERVAL 6 HOUR), INTERVAL (hg.hr - 1) HOUR) AS hr_start,
    DATETIME_ADD(DATETIME_SUB(t.T_target, INTERVAL 6 HOUR), INTERVAL hg.hr HOUR) AS hr_end
  FROM targets t
  CROSS JOIN hour_grid hg
),
vitals_hourly AS (
  SELECT 
    sh.stay_id,
    sh.hr,
    AVG(CASE WHEN ch.itemid = 220045 THEN ch.valuenum END) AS heart_rate,
    AVG(CASE WHEN ch.itemid IN (220179, 220050) THEN ch.valuenum END) AS sbp,
    AVG(CASE WHEN ch.itemid IN (220180, 220051) THEN ch.valuenum END) AS dbp,
    AVG(CASE WHEN ch.itemid IN (220181, 220052, 225312) THEN ch.valuenum END) AS map,
    AVG(CASE WHEN ch.itemid = 220210 THEN ch.valuenum END) AS respiratory_rate,
    AVG(CASE WHEN ch.itemid = 223762 THEN ch.valuenum END) AS temperature,
    AVG(CASE WHEN ch.itemid = 220277 THEN ch.valuenum END) AS spo2
  FROM stay_hours sh
  INNER JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ch
    ON ch.stay_id = sh.stay_id
  WHERE ch.itemid IN (220045, 220179, 220050, 220180, 220051, 220181, 220052, 225312, 220210, 223762, 220277)
    AND ch.valuenum IS NOT NULL
    AND ch.charttime >= sh.hr_start AND ch.charttime < sh.hr_end
  GROUP BY sh.stay_id, sh.hr
),
labs_hourly AS (
  SELECT 
    sh.stay_id,
    sh.hr,
    AVG(CASE WHEN l.itemid IN (50813, 225668) THEN l.valuenum END) AS lactate,
    AVG(CASE WHEN l.itemid IN (50931, 225664, 50809) THEN l.valuenum END) AS glucose,
    AVG(CASE WHEN l.itemid IN (51301, 220546) THEN l.valuenum END) AS wbc,
    AVG(CASE WHEN l.itemid IN (51265, 227457) THEN l.valuenum END) AS platelets,
    AVG(CASE WHEN l.itemid IN (50912, 220615) THEN l.valuenum END) AS creatinine,
    AVG(CASE WHEN l.itemid IN (51006, 225624) THEN l.valuenum END) AS bun,
    AVG(CASE WHEN l.itemid IN (50983, 220645, 50824) THEN l.valuenum END) AS sodium,
    AVG(CASE WHEN l.itemid IN (50971, 227442, 50822) THEN l.valuenum END) AS potassium,
    AVG(CASE WHEN l.itemid IN (50902, 220602, 50806) THEN l.valuenum END) AS chloride,
    AVG(CASE WHEN l.itemid IN (50804, 50882, 227443) THEN l.valuenum END) AS bicarbonate,
    AVG(CASE WHEN l.itemid IN (50885, 225690) THEN l.valuenum END) AS bilirubin,
    AVG(CASE WHEN l.itemid IN (51222, 220228) THEN l.valuenum END) AS hemoglobin,
    AVG(CASE WHEN l.itemid IN (51221, 220545) THEN l.valuenum END) AS hematocrit,
    AVG(CASE WHEN l.itemid IN (50820, 223830) THEN l.valuenum END) AS ph,
    AVG(CASE WHEN l.itemid IN (50821, 220224) THEN l.valuenum END) AS pao2,
    AVG(CASE WHEN l.itemid IN (50818, 220235) THEN l.valuenum END) AS paco2,
    AVG(CASE WHEN l.itemid IN (50802, 224828) THEN l.valuenum END) AS base_excess
  FROM stay_hours sh
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.labevents` l
    ON l.hadm_id = sh.hadm_id
  WHERE l.itemid IN (50813, 225668, 50931, 225664, 50809, 51301, 220546, 51265, 227457, 50912, 220615, 51006, 225624, 50983, 220645, 50824, 50971, 227442, 50822, 50902, 220602, 50806, 50804, 50882, 227443, 50885, 225690, 51222, 220228, 51221, 220545, 50820, 223830, 50821, 220224, 50818, 220235, 50802, 224828)
    AND l.valuenum IS NOT NULL
    AND l.charttime >= sh.hr_start AND l.charttime < sh.hr_end
  GROUP BY sh.stay_id, sh.hr
)
SELECT 
  sh.stay_id AS patientunitstayid,
  sh.hr AS hour_idx,
  sh.sepsis_label AS label,
  sh.age,
  sh.gender,
  s.weight,
  s.height,
  v.heart_rate, v.sbp, v.dbp, v.map, v.respiratory_rate, v.temperature, v.spo2,
  l.lactate, l.glucose, l.wbc, l.platelets, l.creatinine, l.bun, l.sodium, l.potassium, l.chloride, l.bicarbonate, l.bilirubin, l.hemoglobin, l.hematocrit, l.ph, l.pao2, l.paco2, l.base_excess
FROM stay_hours sh
LEFT JOIN `mimic-research-490610.sepsis_cohort.sepsis_static_vars` s
  ON s.stay_id = sh.stay_id
LEFT JOIN vitals_hourly v
  ON v.stay_id = sh.stay_id AND v.hr = sh.hr
LEFT JOIN labs_hourly l
  ON l.stay_id = sh.stay_id AND l.hr = sh.hr;

-- 2.2. Aggregated Tabular Data (1 row per stay)
CREATE OR REPLACE TABLE `mimic-research-490610.sepsis_cohort.final_sepsis_features_12h` AS
SELECT 
  patientunitstayid,
  ANY_VALUE(label) AS label,
  ANY_VALUE(age) AS age,
  ANY_VALUE(gender) AS gender,
  ANY_VALUE(weight) AS weight,
  ANY_VALUE(height) AS height,
  AVG(heart_rate) AS heart_rate_mean, STDDEV(heart_rate) AS heart_rate_std,
  AVG(sbp) AS sbp_mean, STDDEV(sbp) AS sbp_std,
  AVG(dbp) AS dbp_mean, STDDEV(dbp) AS dbp_std,
  AVG(map) AS map_mean, STDDEV(map) AS map_std,
  AVG(respiratory_rate) AS respiratory_rate_mean, STDDEV(respiratory_rate) AS respiratory_rate_std,
  AVG(temperature) AS temperature_mean, STDDEV(temperature) AS temperature_std,
  AVG(spo2) AS spo2_mean, STDDEV(spo2) AS spo2_std,
  AVG(lactate) AS lactate_mean, STDDEV(lactate) AS lactate_std,
  AVG(glucose) AS glucose_mean, STDDEV(glucose) AS glucose_std,
  AVG(wbc) AS wbc_mean, STDDEV(wbc) AS wbc_std,
  AVG(platelets) AS platelets_mean, STDDEV(platelets) AS platelets_std,
  AVG(creatinine) AS creatinine_mean, STDDEV(creatinine) AS creatinine_std,
  AVG(bun) AS bun_mean, STDDEV(bun) AS bun_std,
  AVG(sodium) AS sodium_mean, STDDEV(sodium) AS sodium_std,
  AVG(potassium) AS potassium_mean, STDDEV(potassium) AS potassium_std,
  AVG(chloride) AS chloride_mean, STDDEV(chloride) AS chloride_std,
  AVG(bicarbonate) AS bicarbonate_mean, STDDEV(bicarbonate) AS bicarbonate_std,
  AVG(bilirubin) AS bilirubin_mean, STDDEV(bilirubin) AS bilirubin_std,
  AVG(hemoglobin) AS hemoglobin_mean, STDDEV(hemoglobin) AS hemoglobin_std,
  AVG(hematocrit) AS hematocrit_mean, STDDEV(hematocrit) AS hematocrit_std,
  AVG(ph) AS ph_mean, STDDEV(ph) AS ph_std,
  AVG(pao2) AS pao2_mean, STDDEV(pao2) AS pao2_std,
  AVG(paco2) AS paco2_mean, STDDEV(paco2) AS paco2_std,
  AVG(base_excess) AS base_excess_mean, STDDEV(base_excess) AS base_excess_std
FROM `mimic-research-490610.sepsis_cohort.final_sepsis_seq_features_12h`
GROUP BY patientunitstayid;


-- ===========================================================================
-- HORIZON 3: 18-HOUR PREDICTION TABLES
-- ===========================================================================

-- 3.1. Sequential Hourly Data (6 rows per stay)
CREATE OR REPLACE TABLE `mimic-research-490610.sepsis_cohort.final_sepsis_seq_features_18h` AS
WITH targets AS (
  SELECT 
    stay_id,
    subject_id,
    hadm_id,
    intime,
    age,
    gender,
    sepsis_label,
    sepsis_onset_time,
    CASE 
      WHEN sepsis_label = 1 THEN DATETIME_SUB(sepsis_onset_time, INTERVAL 18 HOUR)
      ELSE DATETIME_ADD(intime, INTERVAL 12 HOUR)
    END AS T_target
  FROM `mimic-research-490610.sepsis_cohort.sepsis_cohort`
),
hour_grid AS (
  SELECT 1 AS hr UNION ALL
  SELECT 2 AS hr UNION ALL
  SELECT 3 AS hr UNION ALL
  SELECT 4 AS hr UNION ALL
  SELECT 5 AS hr UNION ALL
  SELECT 6 AS hr
),
stay_hours AS (
  SELECT 
    t.*,
    hg.hr,
    DATETIME_ADD(DATETIME_SUB(t.T_target, INTERVAL 6 HOUR), INTERVAL (hg.hr - 1) HOUR) AS hr_start,
    DATETIME_ADD(DATETIME_SUB(t.T_target, INTERVAL 6 HOUR), INTERVAL hg.hr HOUR) AS hr_end
  FROM targets t
  CROSS JOIN hour_grid hg
),
vitals_hourly AS (
  SELECT 
    sh.stay_id,
    sh.hr,
    AVG(CASE WHEN ch.itemid = 220045 THEN ch.valuenum END) AS heart_rate,
    AVG(CASE WHEN ch.itemid IN (220179, 220050) THEN ch.valuenum END) AS sbp,
    AVG(CASE WHEN ch.itemid IN (220180, 220051) THEN ch.valuenum END) AS dbp,
    AVG(CASE WHEN ch.itemid IN (220181, 220052, 225312) THEN ch.valuenum END) AS map,
    AVG(CASE WHEN ch.itemid = 220210 THEN ch.valuenum END) AS respiratory_rate,
    AVG(CASE WHEN ch.itemid = 223762 THEN ch.valuenum END) AS temperature,
    AVG(CASE WHEN ch.itemid = 220277 THEN ch.valuenum END) AS spo2
  FROM stay_hours sh
  INNER JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ch
    ON ch.stay_id = sh.stay_id
  WHERE ch.itemid IN (220045, 220179, 220050, 220180, 220051, 220181, 220052, 225312, 220210, 223762, 220277)
    AND ch.valuenum IS NOT NULL
    AND ch.charttime >= sh.hr_start AND ch.charttime < sh.hr_end
  GROUP BY sh.stay_id, sh.hr
),
labs_hourly AS (
  SELECT 
    sh.stay_id,
    sh.hr,
    AVG(CASE WHEN l.itemid IN (50813, 225668) THEN l.valuenum END) AS lactate,
    AVG(CASE WHEN l.itemid IN (50931, 225664, 50809) THEN l.valuenum END) AS glucose,
    AVG(CASE WHEN l.itemid IN (51301, 220546) THEN l.valuenum END) AS wbc,
    AVG(CASE WHEN l.itemid IN (51265, 227457) THEN l.valuenum END) AS platelets,
    AVG(CASE WHEN l.itemid IN (50912, 220615) THEN l.valuenum END) AS creatinine,
    AVG(CASE WHEN l.itemid IN (51006, 225624) THEN l.valuenum END) AS bun,
    AVG(CASE WHEN l.itemid IN (50983, 220645, 50824) THEN l.valuenum END) AS sodium,
    AVG(CASE WHEN l.itemid IN (50971, 227442, 50822) THEN l.valuenum END) AS potassium,
    AVG(CASE WHEN l.itemid IN (50902, 220602, 50806) THEN l.valuenum END) AS chloride,
    AVG(CASE WHEN l.itemid IN (50804, 50882, 227443) THEN l.valuenum END) AS bicarbonate,
    AVG(CASE WHEN l.itemid IN (50885, 225690) THEN l.valuenum END) AS bilirubin,
    AVG(CASE WHEN l.itemid IN (51222, 220228) THEN l.valuenum END) AS hemoglobin,
    AVG(CASE WHEN l.itemid IN (51221, 220545) THEN l.valuenum END) AS hematocrit,
    AVG(CASE WHEN l.itemid IN (50820, 223830) THEN l.valuenum END) AS ph,
    AVG(CASE WHEN l.itemid IN (50821, 220224) THEN l.valuenum END) AS pao2,
    AVG(CASE WHEN l.itemid IN (50818, 220235) THEN l.valuenum END) AS paco2,
    AVG(CASE WHEN l.itemid IN (50802, 224828) THEN l.valuenum END) AS base_excess
  FROM stay_hours sh
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.labevents` l
    ON l.hadm_id = sh.hadm_id
  WHERE l.itemid IN (50813, 225668, 50931, 225664, 50809, 51301, 220546, 51265, 227457, 50912, 220615, 51006, 225624, 50983, 220645, 50824, 50971, 227442, 50822, 50902, 220602, 50806, 50804, 50882, 227443, 50885, 225690, 51222, 220228, 51221, 220545, 50820, 223830, 50821, 220224, 50818, 220235, 50802, 224828)
    AND l.valuenum IS NOT NULL
    AND l.charttime >= sh.hr_start AND l.charttime < sh.hr_end
  GROUP BY sh.stay_id, sh.hr
)
SELECT 
  sh.stay_id AS patientunitstayid,
  sh.hr AS hour_idx,
  sh.sepsis_label AS label,
  sh.age,
  sh.gender,
  s.weight,
  s.height,
  v.heart_rate, v.sbp, v.dbp, v.map, v.respiratory_rate, v.temperature, v.spo2,
  l.lactate, l.glucose, l.wbc, l.platelets, l.creatinine, l.bun, l.sodium, l.potassium, l.chloride, l.bicarbonate, l.bilirubin, l.hemoglobin, l.hematocrit, l.ph, l.pao2, l.paco2, l.base_excess
FROM stay_hours sh
LEFT JOIN `mimic-research-490610.sepsis_cohort.sepsis_static_vars` s
  ON s.stay_id = sh.stay_id
LEFT JOIN vitals_hourly v
  ON v.stay_id = sh.stay_id AND v.hr = sh.hr
LEFT JOIN labs_hourly l
  ON l.stay_id = sh.stay_id AND l.hr = sh.hr;

-- 3.2. Aggregated Tabular Data (1 row per stay)
CREATE OR REPLACE TABLE `mimic-research-490610.sepsis_cohort.final_sepsis_features_18h` AS
SELECT 
  patientunitstayid,
  ANY_VALUE(label) AS label,
  ANY_VALUE(age) AS age,
  ANY_VALUE(gender) AS gender,
  ANY_VALUE(weight) AS weight,
  ANY_VALUE(height) AS height,
  AVG(heart_rate) AS heart_rate_mean, STDDEV(heart_rate) AS heart_rate_std,
  AVG(sbp) AS sbp_mean, STDDEV(sbp) AS sbp_std,
  AVG(dbp) AS dbp_mean, STDDEV(dbp) AS dbp_std,
  AVG(map) AS map_mean, STDDEV(map) AS map_std,
  AVG(respiratory_rate) AS respiratory_rate_mean, STDDEV(respiratory_rate) AS respiratory_rate_std,
  AVG(temperature) AS temperature_mean, STDDEV(temperature) AS temperature_std,
  AVG(spo2) AS spo2_mean, STDDEV(spo2) AS spo2_std,
  AVG(lactate) AS lactate_mean, STDDEV(lactate) AS lactate_std,
  AVG(glucose) AS glucose_mean, STDDEV(glucose) AS glucose_std,
  AVG(wbc) AS wbc_mean, STDDEV(wbc) AS wbc_std,
  AVG(platelets) AS platelets_mean, STDDEV(platelets) AS platelets_std,
  AVG(creatinine) AS creatinine_mean, STDDEV(creatinine) AS creatinine_std,
  AVG(bun) AS bun_mean, STDDEV(bun) AS bun_std,
  AVG(sodium) AS sodium_mean, STDDEV(sodium) AS sodium_std,
  AVG(potassium) AS potassium_mean, STDDEV(potassium) AS potassium_std,
  AVG(chloride) AS chloride_mean, STDDEV(chloride) AS chloride_std,
  AVG(bicarbonate) AS bicarbonate_mean, STDDEV(bicarbonate) AS bicarbonate_std,
  AVG(bilirubin) AS bilirubin_mean, STDDEV(bilirubin) AS bilirubin_std,
  AVG(hemoglobin) AS hemoglobin_mean, STDDEV(hemoglobin) AS hemoglobin_std,
  AVG(hematocrit) AS hematocrit_mean, STDDEV(hematocrit) AS hematocrit_std,
  AVG(ph) AS ph_mean, STDDEV(ph) AS ph_std,
  AVG(pao2) AS pao2_mean, STDDEV(pao2) AS pao2_std,
  AVG(paco2) AS paco2_mean, STDDEV(paco2) AS paco2_std,
  AVG(base_excess) AS base_excess_mean, STDDEV(base_excess) AS base_excess_std
FROM `mimic-research-490610.sepsis_cohort.final_sepsis_seq_features_18h`
GROUP BY patientunitstayid;


-- ===========================================================================
-- HORIZON 4: 24-HOUR PREDICTION TABLES
-- ===========================================================================

-- 4.1. Sequential Hourly Data (6 rows per stay)
CREATE OR REPLACE TABLE `mimic-research-490610.sepsis_cohort.final_sepsis_seq_features_24h` AS
WITH targets AS (
  SELECT 
    stay_id,
    subject_id,
    hadm_id,
    intime,
    age,
    gender,
    sepsis_label,
    sepsis_onset_time,
    CASE 
      WHEN sepsis_label = 1 THEN DATETIME_SUB(sepsis_onset_time, INTERVAL 24 HOUR)
      ELSE DATETIME_ADD(intime, INTERVAL 6 HOUR)
    END AS T_target
  FROM `mimic-research-490610.sepsis_cohort.sepsis_cohort`
),
hour_grid AS (
  SELECT 1 AS hr UNION ALL
  SELECT 2 AS hr UNION ALL
  SELECT 3 AS hr UNION ALL
  SELECT 4 AS hr UNION ALL
  SELECT 5 AS hr UNION ALL
  SELECT 6 AS hr
),
stay_hours AS (
  SELECT 
    t.*,
    hg.hr,
    DATETIME_ADD(DATETIME_SUB(t.T_target, INTERVAL 6 HOUR), INTERVAL (hg.hr - 1) HOUR) AS hr_start,
    DATETIME_ADD(DATETIME_SUB(t.T_target, INTERVAL 6 HOUR), INTERVAL hg.hr HOUR) AS hr_end
  FROM targets t
  CROSS JOIN hour_grid hg
),
vitals_hourly AS (
  SELECT 
    sh.stay_id,
    sh.hr,
    AVG(CASE WHEN ch.itemid = 220045 THEN ch.valuenum END) AS heart_rate,
    AVG(CASE WHEN ch.itemid IN (220179, 220050) THEN ch.valuenum END) AS sbp,
    AVG(CASE WHEN ch.itemid IN (220180, 220051) THEN ch.valuenum END) AS dbp,
    AVG(CASE WHEN ch.itemid IN (220181, 220052, 225312) THEN ch.valuenum END) AS map,
    AVG(CASE WHEN ch.itemid = 220210 THEN ch.valuenum END) AS respiratory_rate,
    AVG(CASE WHEN ch.itemid = 223762 THEN ch.valuenum END) AS temperature,
    AVG(CASE WHEN ch.itemid = 220277 THEN ch.valuenum END) AS spo2
  FROM stay_hours sh
  INNER JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ch
    ON ch.stay_id = sh.stay_id
  WHERE ch.itemid IN (220045, 220179, 220050, 220180, 220051, 220181, 220052, 225312, 220210, 223762, 220277)
    AND ch.valuenum IS NOT NULL
    AND ch.charttime >= sh.hr_start AND ch.charttime < sh.hr_end
  GROUP BY sh.stay_id, sh.hr
),
labs_hourly AS (
  SELECT 
    sh.stay_id,
    sh.hr,
    AVG(CASE WHEN l.itemid IN (50813, 225668) THEN l.valuenum END) AS lactate,
    AVG(CASE WHEN l.itemid IN (50931, 225664, 50809) THEN l.valuenum END) AS glucose,
    AVG(CASE WHEN l.itemid IN (51301, 220546) THEN l.valuenum END) AS wbc,
    AVG(CASE WHEN l.itemid IN (51265, 227457) THEN l.valuenum END) AS platelets,
    AVG(CASE WHEN l.itemid IN (50912, 220615) THEN l.valuenum END) AS creatinine,
    AVG(CASE WHEN l.itemid IN (51006, 225624) THEN l.valuenum END) AS bun,
    AVG(CASE WHEN l.itemid IN (50983, 220645, 50824) THEN l.valuenum END) AS sodium,
    AVG(CASE WHEN l.itemid IN (50971, 227442, 50822) THEN l.valuenum END) AS potassium,
    AVG(CASE WHEN l.itemid IN (50902, 220602, 50806) THEN l.valuenum END) AS chloride,
    AVG(CASE WHEN l.itemid IN (50804, 50882, 227443) THEN l.valuenum END) AS bicarbonate,
    AVG(CASE WHEN l.itemid IN (50885, 225690) THEN l.valuenum END) AS bilirubin,
    AVG(CASE WHEN l.itemid IN (51222, 220228) THEN l.valuenum END) AS hemoglobin,
    AVG(CASE WHEN l.itemid IN (51221, 220545) THEN l.valuenum END) AS hematocrit,
    AVG(CASE WHEN l.itemid IN (50820, 223830) THEN l.valuenum END) AS ph,
    AVG(CASE WHEN l.itemid IN (50821, 220224) THEN l.valuenum END) AS pao2,
    AVG(CASE WHEN l.itemid IN (50818, 220235) THEN l.valuenum END) AS paco2,
    AVG(CASE WHEN l.itemid IN (50802, 224828) THEN l.valuenum END) AS base_excess
  FROM stay_hours sh
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.labevents` l
    ON l.hadm_id = sh.hadm_id
  WHERE l.itemid IN (50813, 225668, 50931, 225664, 50809, 51301, 220546, 51265, 227457, 50912, 220615, 51006, 225624, 50983, 220645, 50824, 50971, 227442, 50822, 50902, 220602, 50806, 50804, 50882, 227443, 50885, 225690, 51222, 220228, 51221, 220545, 50820, 223830, 50821, 220224, 50818, 220235, 50802, 224828)
    AND l.valuenum IS NOT NULL
    AND l.charttime >= sh.hr_start AND l.charttime < sh.hr_end
  GROUP BY sh.stay_id, sh.hr
)
SELECT 
  sh.stay_id AS patientunitstayid,
  sh.hr AS hour_idx,
  sh.sepsis_label AS label,
  sh.age,
  sh.gender,
  s.weight,
  s.height,
  v.heart_rate, v.sbp, v.dbp, v.map, v.respiratory_rate, v.temperature, v.spo2,
  l.lactate, l.glucose, l.wbc, l.platelets, l.creatinine, l.bun, l.sodium, l.potassium, l.chloride, l.bicarbonate, l.bilirubin, l.hemoglobin, l.hematocrit, l.ph, l.pao2, l.paco2, l.base_excess
FROM stay_hours sh
LEFT JOIN `mimic-research-490610.sepsis_cohort.sepsis_static_vars` s
  ON s.stay_id = sh.stay_id
LEFT JOIN vitals_hourly v
  ON v.stay_id = sh.stay_id AND v.hr = sh.hr
LEFT JOIN labs_hourly l
  ON l.stay_id = sh.stay_id AND l.hr = sh.hr;

-- 4.2. Aggregated Tabular Data (1 row per stay)
CREATE OR REPLACE TABLE `mimic-research-490610.sepsis_cohort.final_sepsis_features_24h` AS
SELECT 
  patientunitstayid,
  ANY_VALUE(label) AS label,
  ANY_VALUE(age) AS age,
  ANY_VALUE(gender) AS gender,
  ANY_VALUE(weight) AS weight,
  ANY_VALUE(height) AS height,
  AVG(heart_rate) AS heart_rate_mean, STDDEV(heart_rate) AS heart_rate_std,
  AVG(sbp) AS sbp_mean, STDDEV(sbp) AS sbp_std,
  AVG(dbp) AS dbp_mean, STDDEV(dbp) AS dbp_std,
  AVG(map) AS map_mean, STDDEV(map) AS map_std,
  AVG(respiratory_rate) AS respiratory_rate_mean, STDDEV(respiratory_rate) AS respiratory_rate_std,
  AVG(temperature) AS temperature_mean, STDDEV(temperature) AS temperature_std,
  AVG(spo2) AS spo2_mean, STDDEV(spo2) AS spo2_std,
  AVG(lactate) AS lactate_mean, STDDEV(lactate) AS lactate_std,
  AVG(glucose) AS glucose_mean, STDDEV(glucose) AS glucose_std,
  AVG(wbc) AS wbc_mean, STDDEV(wbc) AS wbc_std,
  AVG(platelets) AS platelets_mean, STDDEV(platelets) AS platelets_std,
  AVG(creatinine) AS creatinine_mean, STDDEV(creatinine) AS creatinine_std,
  AVG(bun) AS bun_mean, STDDEV(bun) AS bun_std,
  AVG(sodium) AS sodium_mean, STDDEV(sodium) AS sodium_std,
  AVG(potassium) AS potassium_mean, STDDEV(potassium) AS potassium_std,
  AVG(chloride) AS chloride_mean, STDDEV(chloride) AS chloride_std,
  AVG(bicarbonate) AS bicarbonate_mean, STDDEV(bicarbonate) AS bicarbonate_std,
  AVG(bilirubin) AS bilirubin_mean, STDDEV(bilirubin) AS bilirubin_std,
  AVG(hemoglobin) AS hemoglobin_mean, STDDEV(hemoglobin) AS hemoglobin_std,
  AVG(hematocrit) AS hematocrit_mean, STDDEV(hematocrit) AS hematocrit_std,
  AVG(ph) AS ph_mean, STDDEV(ph) AS ph_std,
  AVG(pao2) AS pao2_mean, STDDEV(pao2) AS pao2_std,
  AVG(paco2) AS paco2_mean, STDDEV(paco2) AS paco2_std,
  AVG(base_excess) AS base_excess_mean, STDDEV(base_excess) AS base_excess_std
FROM `mimic-research-490610.sepsis_cohort.final_sepsis_seq_features_24h`
GROUP BY patientunitstayid;
