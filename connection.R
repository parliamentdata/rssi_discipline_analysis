# Load required libraries
suppressWarnings({
  suppressPackageStartupMessages({
    library(DBI)
    library(RPostgres)
    library(dbplyr)
    library(config)
  })
})

# Define the configuration file path
config_path <- file.path("utility", "config.yml")

# Load the active configuration
is_old <- config::is_active("old")  # Correct usage, no file argument here
conn_args <- config::get("dataconnection", file = config_path)  # Specify the file here

# Establish the database connection
if (!is_old) {
  myconn <- dbConnect(
    RPostgres::Postgres(),
    dbname = conn_args$database,
    host = conn_args$server,
    port = conn_args$port,
    user = conn_args$uid,
    password = conn_args$pwd,
    bigint = "numeric"
  )
} else {
  myconn <- dbConnect(
    odbc::odbc(),
    Driver = conn_args$driver,
    Server = conn_args$server,
    UID = conn_args$uid,
    PWD = conn_args$pwd,
    Port = conn_args$port,
    Database = conn_args$database,
    bigint = "integer"
  )
}
