CREATE OR REPLACE TABLE bronze.payer (
    payer_key INT,
    payer_name STRING,
    payer_type STRING,
    load_date TIMESTAMP,
    file_name STRING
) USING DELTA;
