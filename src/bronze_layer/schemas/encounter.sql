CREATE OR REPLACE TABLE bronze.encounter (
    encounter_key INT,
    patient_key INT,
    provider_key INT,
    facility_key INT,
    payer_key INT,
    date_key INT,
    drg_code STRING,
    total_charge DOUBLE,
    total_allowed DOUBLE,
    total_paid DOUBLE,
    total_adjustment DOUBLE,
    load_date TIMESTAMP,
    file_name STRING
) USING DELTA;
