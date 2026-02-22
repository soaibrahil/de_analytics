CREATE OR REPLACE TABLE bronze.transaction (
    transaction_key INT,
    encounter_key INT,
    transaction_type STRING,
    amount DOUBLE,
    date_key INT,
    load_date TIMESTAMP,
    file_name STRING
) USING DELTA;
