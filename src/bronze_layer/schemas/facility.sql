CREATE OR REPLACE TABLE bronze.facility (
    facility_key INT,
    facility_name STRING,
    city STRING,
    state STRING,
    bed_count INT,
    load_date TIMESTAMP,
    file_name STRING
) USING DELTA;
