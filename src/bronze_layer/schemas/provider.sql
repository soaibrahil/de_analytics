CREATE OR REPLACE TABLE bronze.provider (
    provider_key INT,
    provider_id STRING,
    provider_name STRING,
    specialty STRING,
    department STRING,
    effective_from DATE,
    effective_to DATE,
    is_current BOOLEAN,
    load_date TIMESTAMP,
    file_name STRING
) USING DELTA;
