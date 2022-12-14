CREATE DATABASE IF NOT EXISTS bank_database;

USE bank_database;

-- Drop previous tables

DROP TABLE IF EXISTS account;
DROP TABLE IF EXISTS card_train;
DROP TABLE IF EXISTS card_test;
DROP TABLE IF EXISTS client;
DROP TABLE IF EXISTS disposition;
DROP TABLE IF EXISTS district;
DROP TABLE IF EXISTS loan_train;
DROP TABLE IF EXISTS loan_test;
DROP TABLE IF EXISTS trans_train;
DROP TABLE IF EXISTS trans_test;

-- Create and Load Account Table

CREATE TABLE IF NOT EXISTS account (
    account_id INT NOT NULL,
    district_id INT NOT NULL,
    frequency VARCHAR(20) NOT NULL,
    creation_date INT NOT NULL
);

LOAD DATA LOCAL INFILE  
'../ficheiros_competicao/account.csv'
INTO TABLE account
FIELDS TERMINATED BY ';' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(account_id, district_id, frequency, creation_date);

-- Create and Load Card Train Table

CREATE TABLE IF NOT EXISTS card_train (
    card_id INT NOT NULL,
    disp_id INT NOT NULL,
    card_type VARCHAR(20) NOT NULL,
    issued INT NOT NULL
);

LOAD DATA LOCAL INFILE  
'../ficheiros_competicao/card_train.csv'
INTO TABLE card_train
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(card_id, disp_id, card_type, issued);

-- Create and Load Card Test Table

CREATE TABLE IF NOT EXISTS card_test (
    card_id INT NOT NULL,
    disp_id INT NOT NULL,
    card_type VARCHAR(20) NOT NULL,
    issued INT NOT NULL
);

LOAD DATA LOCAL INFILE  
'../ficheiros_competicao/card_test.csv'
INTO TABLE card_test
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(card_id, disp_id, card_type, issued);

-- Create and Load Client Table

CREATE TABLE IF NOT EXISTS client (
    client_id INT NOT NULL,
    birth_number INT NOT NULL,
    district_id INT NOT NULL
);

LOAD DATA LOCAL INFILE  
'../ficheiros_competicao/client.csv'
INTO TABLE client
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(client_id, birth_number, district_id);

-- Create and Load Disposition Table

CREATE TABLE IF NOT EXISTS disposition (
    disp_id INT NOT NULL,
    client_id INT NOT NULL,
    account_id INT NOT NULL,
    disp_type VARCHAR(20) NOT NULL
);

LOAD DATA LOCAL INFILE  
'../ficheiros_competicao/disp.csv'
INTO TABLE disposition
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(disp_id, client_id, account_id, disp_type);

-- Create and Load District Table

CREATE TABLE IF NOT EXISTS district (
    district_id INT NOT NULL,
    district_name VARCHAR(20) NOT NULL,
    region VARCHAR(20) NOT NULL,
    nr_inhabitants INT NOT NULL,
    nr_municip_inhabitants_499 INT NOT NULL,
    nr_municip_inhabitants_500_1999 INT NOT NULL,
    nr_municip_inhabitants_2000_9999 INT NOT NULL,
    nr_municip_inhabitants_10000 INT NOT NULL,
    nr_cities INT NOT NULL,
    ratio_urban_inhabitants FLOAT NOT NULL,
    average_salary FLOAT NOT NULL,
    unemployment_rate_95 VARCHAR(20) NOT NULL,
    unemployment_rate_96 FLOAT NOT NULL,
    nr_enterpreneurs_1000_inhabitants INT NOT NULL,
    nr_commited_crimes_95 VARCHAR(20) NOT NULL,
    nr_commited_crimes_96 INT NOT NULL
);

LOAD DATA LOCAL INFILE  
'../ficheiros_competicao/district.csv'
INTO TABLE district
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(district_id, district_name, region, nr_inhabitants, nr_municip_inhabitants_499,
nr_municip_inhabitants_500_1999, nr_municip_inhabitants_2000_9999, nr_municip_inhabitants_10000,
nr_cities, ratio_urban_inhabitants, average_salary, unemployment_rate_95, unemployment_rate_96,
nr_enterpreneurs_1000_inhabitants, nr_commited_crimes_95, nr_commited_crimes_96);

-- Create and Load Loan Train Table

CREATE TABLE IF NOT EXISTS loan_train (
    loan_id INT NOT NULL,
    account_id INT NOT NULL,
    granted_date INT NOT NULL,
    amount INT NOT NULL,
    duration INT NOT NULL,
    payments INT NOT NULL,
    loan_status INT NOT NULL
);

LOAD DATA LOCAL INFILE  
'../ficheiros_competicao/loan_train.csv'
INTO TABLE loan_train
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(loan_id, account_id, granted_date, amount, duration, payments, loan_status);

-- Create and Load Loan Test Table

CREATE TABLE IF NOT EXISTS loan_test (
    loan_id INT NOT NULL,
    account_id INT NOT NULL,
    granted_date INT NOT NULL,
    amount INT NOT NULL,
    duration INT NOT NULL,
    payments INT NOT NULL,
    loan_status DECIMAL NOT NULL
);

LOAD DATA LOCAL INFILE  
'../ficheiros_competicao/loan_test.csv'
INTO TABLE loan_test
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(loan_id, account_id, granted_date, amount, duration, payments, loan_status);

-- Create and Load Transaction Train Table

CREATE TABLE IF NOT EXISTS trans_train (
    trans_id INT,
    account_id INT,
    trans_date INT,
    trans_type VARCHAR(20),
    operation VARCHAR(50),
    amount FLOAT,
    balance FLOAT,
    k_symbol VARCHAR(50),
    bank VARCHAR(20),
    account INT
);

LOAD DATA LOCAL INFILE  
'../ficheiros_competicao/trans_train.csv'
INTO TABLE trans_train
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(trans_id, account_id, trans_date, trans_type, operation, amount, balance, k_symbol, bank, account);

-- Create and Load Transaction Test Table

CREATE TABLE trans_test (
    trans_id INT NOT NULL,
    account_id INT NOT NULL,
    trans_date INT NOT NULL,
    trans_type VARCHAR(20) NOT NULL,
    operation VARCHAR(50),
    amount INT NOT NULL,
    balance INT NOT NULL,
    k_symbol VARCHAR(50),
    bank VARCHAR(20),
    account INT
);

LOAD DATA LOCAL INFILE  
'../ficheiros_competicao/trans_test.csv'
INTO TABLE trans_test
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(trans_id, account_id, trans_date, trans_type, operation, amount, balance, k_symbol, bank, account);