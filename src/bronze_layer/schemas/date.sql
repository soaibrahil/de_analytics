CREATE OR REPLACE TABLE bronze.date (
    date_key INT,
    date DATE,
    year INT,
    quarter INT,
    month INT,
    day INT,
    week INT,
    load_date TIMESTAMP,
    file_name STRING
) USING DELTA;
