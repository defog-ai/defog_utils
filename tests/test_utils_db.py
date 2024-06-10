import unittest
from defog_utils.defog_utils.utils_db import (
    convert_data_type_postgres,
    fix_md,
    generate_aliases,
    get_table_names,
    mk_create_ddl,
    mk_create_table_ddl,
    mk_delete_ddl,
    parse_md,
)


class TestResolveDataTypePostgres(unittest.TestCase):
    def test_convert_data_type_postgres(self):
        self.assertEqual(convert_data_type_postgres("int"), "integer")
        self.assertEqual(convert_data_type_postgres("double"), "double precision")
        self.assertEqual(convert_data_type_postgres("varchar"), "text")
        self.assertEqual(convert_data_type_postgres("datetime"), "timestamp")
        self.assertEqual(convert_data_type_postgres("timestamp_ntz"), "timestamp")
        self.assertEqual(convert_data_type_postgres("timestamp_ltz"), "timestamp")
        self.assertEqual(convert_data_type_postgres("timestamp_tz"), "timestamp")
        self.assertEqual(convert_data_type_postgres("unknown"), "unknown")


class TestMkCreateTableDDL(unittest.TestCase):
    def test_mk_create_table_ddl(self):
        table_name = "table1"
        columns = [
            {
                "column_name": "col1",
                "data_type": "int",
                "column_description": "primary key",
            },
            {
                "column_name": "col2",
                "data_type": "character varying",
                "column_description": "not null",
            },
            {
                "column_name": "col3",
                "data_type": "text",
                "column_description": " hello world",
            },
        ]
        expected_output = (
            "CREATE TABLE table1 (\n"
            "  col1 integer, --primary key\n"
            "  col2 character varying, --not null\n"
            "  col3 text -- hello world\n"
            ");\n"
        )
        self.assertEqual(mk_create_table_ddl(table_name, columns), expected_output)


class TestMkCreateDDL(unittest.TestCase):
    def test_mk_create_ddl(self):
        md = {
            "table1": [
                {
                    "column_name": "col1",
                    "data_type": "int",
                    "column_description": "primary key",
                },
                {
                    "column_name": "col2",
                    "data_type": "text",
                    "column_description": "not null",
                },
                {"column_name": "col3", "data_type": "text", "column_description": ""},
            ],
            "table2": [
                {
                    "column_name": "col3",
                    "data_type": "int",
                    "column_description": "some column description",
                },
            ],
        }
        expected_output = (
            "CREATE TABLE table1 (\n"
            "  col1 integer, --primary key\n"
            "  col2 text, --not null\n"
            "  col3 text\n"
            ");\n"
            "CREATE TABLE table2 (\n"
            "  col3 integer --some column description\n"
            ");\n"
        )
        self.assertEqual(mk_create_ddl(md), expected_output)


class TestMkDeleteDDL(unittest.TestCase):
    def test_mk_delete_ddl(self):
        md = {
            "table1": [
                {
                    "column_name": "col1",
                    "data_type": "int",
                    "column_description": "primary key",
                },
                {
                    "column_name": "col2",
                    "data_type": "text",
                    "column_description": "not null",
                },
                {"column_name": "col3", "data_type": "text", "column_description": ""},
            ],
            "table2": [
                {
                    "column_name": "col1",
                    "data_type": "int",
                    "column_description": "primary key",
                },
                {
                    "column_name": "col2",
                    "data_type": "text",
                    "column_description": "not null",
                },
            ],
        }
        expected_output = (
            "DROP TABLE IF EXISTS table1 CASCADE;\n"
            "DROP TABLE IF EXISTS table2 CASCADE;\n"
        )
        self.assertEqual(mk_delete_ddl(md), expected_output)


class TestMkCreateDDLWithSchema(unittest.TestCase):
    def test_mk_create_ddl_with_schema(self):
        md = {
            "schema1": {
                "table1": [
                    {
                        "column_name": "col1",
                        "data_type": "int",
                        "column_description": "primary key",
                    },
                    {
                        "column_name": "col2",
                        "data_type": "text",
                        "column_description": "not null",
                    },
                    {
                        "column_name": "col3",
                        "data_type": "text",
                        "column_description": "",
                    },
                ]
            },
            "schema2": {
                "table2": [
                    {
                        "column_name": "col1",
                        "data_type": "int",
                        "column_description": "desc goes here",
                    },
                    {
                        "column_name": "col2",
                        "data_type": "text",
                        "column_description": "not null",
                    },
                ],
            },
        }
        expected_output = (
            "CREATE SCHEMA IF NOT EXISTS schema1;\n"
            "CREATE TABLE schema1.table1 (\n"
            "  col1 integer, --primary key\n"
            "  col2 text, --not null\n"
            "  col3 text\n"
            ");\n"
            "CREATE SCHEMA IF NOT EXISTS schema2;\n"
            "CREATE TABLE schema2.table2 (\n"
            "  col1 integer, --desc goes here\n"
            "  col2 text --not null\n"
            ");\n"
        )
        self.assertEqual(mk_create_ddl(md), expected_output)


class TestMkDeleteDDLWithSchema(unittest.TestCase):
    def test_mk_delete_ddl_tables(self):
        md = {
            "table1": [
                {
                    "column_name": "col1",
                    "data_type": "int",
                    "column_description": "primary key",
                },
                {
                    "column_name": "col2",
                    "data_type": "text",
                    "column_description": "not null",
                },
                {"column_name": "col3", "data_type": "text", "column_description": ""},
            ],
            "table2": [
                {
                    "column_name": "col1",
                    "data_type": "int",
                    "column_description": "primary key",
                },
                {
                    "column_name": "col2",
                    "data_type": "text",
                    "column_description": "not null",
                },
            ],
        }
        expected_output = (
            "DROP TABLE IF EXISTS table1 CASCADE;\n"
            "DROP TABLE IF EXISTS table2 CASCADE;\n"
        )
        self.assertEqual(mk_delete_ddl(md), expected_output)

    def test_mk_delete_ddl_schema(self):
        md = {
            "schema1": {
                "table1": [
                    {
                        "column_name": "col1",
                        "data_type": "int",
                        "column_description": "primary key",
                    },
                    {
                        "column_name": "col2",
                        "data_type": "text",
                        "column_description": "not null",
                    },
                    {
                        "column_name": "col3",
                        "data_type": "text",
                        "column_description": "",
                    },
                ],
                "table2": [
                    {
                        "column_name": "col1",
                        "data_type": "int",
                        "column_description": "desc goes here",
                    },
                    {
                        "column_name": "col2",
                        "data_type": "text",
                        "column_description": "not null",
                    },
                ],
            }
        }
        expected_output = "DROP SCHEMA IF EXISTS schema1 CASCADE;\n"
        self.assertEqual(mk_delete_ddl(md), expected_output)


class TestFixMd(unittest.TestCase):
    def test_fix_md(self):
        md = {
            "table1": [
                {
                    "column_name": "col1",
                    "data_type": "int",
                    "column_description": "primary key",
                },
                {
                    "column_name": "col2",
                    "data_type": "text",
                    "column_description": "not null",
                },
                # should ignore duplicate column with different case
                {
                    "column_name": "Col2",
                    "data_type": "text",
                    "column_description": "another description",
                },
                {"column_name": "col1", "data_type": "text", "column_description": ""},
            ],
            # should normalize table names by removing quotes internally and preserving just the outermost quotes
            '"db"."schema"."table2"': [
                {
                    "column_name": "col/1-",  # should remove / and -
                    "data_type": "double",
                    "column_description": "primary key",
                },
                {
                    "column_name": "col2_date",
                    "data_type": "Date/Time",  # should remove /
                    "column_description": "not null",
                },
                {
                    "column_name": "col3",
                    "data_type": "varchar?",  # should remove ? and convert to text
                    "column_description": "not null",
                },
                {
                    "column_name": "group",  # should add quotes for reserved keyword
                    "data_type": "varchar",  # should convert to text
                    "column_description": "another description",
                },
            ],
        }
        expected_output = {
            "table1": [
                {
                    "column_name": "col1",
                    "data_type": "integer",
                    "column_description": "primary key",
                },
                {
                    "column_name": "col2",
                    "data_type": "text",
                    "column_description": "not null",
                },
            ],
            '"db.schema.table2"': [
                {
                    "column_name": "col1",
                    "data_type": "double precision",
                    "column_description": "primary key",
                },
                {
                    "column_name": "col2_date",
                    "data_type": "timestamp",
                    "column_description": "not null",
                },
                {
                    "column_name": "col3",
                    "data_type": "text",
                    "column_description": "not null",
                },
                {
                    "column_name": '"group"',  # should add quotes for reserved keyword
                    "data_type": "text",
                    "column_description": "another description",
                },
            ],
        }
        self.assertEqual(fix_md(md), expected_output)


class TestParseMd(unittest.TestCase):
    def test_parse_md(self):
        md_str = (
            "CREATE TABLE schema1.table1 (\n"
            "  my col1 integer, --primary key\n"
            "  my col2 double precision, --not null\n"
            "  col3 text\n"
            ");\n"
            "CREATE TABLE table2 (\n"
            "  col1 integer, --primary key\n"
            "  col2 text\n"
            ");\n"
        )
        expected_output = {
            "schema1": {
                "table1": [
                    {
                        "column_name": "my col1",
                        "data_type": "integer",
                        "column_description": "primary key",
                    },
                    {
                        "column_name": "my col2",
                        "data_type": "double precision",
                        "column_description": "not null",
                    },
                    {
                        "column_name": "col3",
                        "data_type": "text",
                        "column_description": "",
                    },
                ]
            },
            "table2": [
                {
                    "column_name": "col1",
                    "data_type": "integer",
                    "column_description": "primary key",
                },
                {"column_name": "col2", "data_type": "text", "column_description": ""},
            ],
        }
        self.assertEqual(parse_md(md_str), expected_output)


class TestGetTableNames(unittest.TestCase):
    def test_get_table_names_1(self):
        md = "CREATE TABLE branches (\n  branch_id integer, --Unique identifier for each branch\n  branch_name character varying --Name of the branch\n);\nCREATE TABLE loans (\n  loan_id integer, --Unique ID for each loan\n  branch_id integer, --Foreign key linking loan to branch\n  loan_amount real, --Amount of the loan\n  loan_date date, --Date of the loan\n  loan_details character varying --Information about the loan\n);\nCREATE TABLE customers (\n  customer_id integer, --Unique ID for each customer\n  customer_name character varying, --Name of the customer\n  other_details character varying --Extra customer details\n);\nCREATE TABLE payments (\n  payment_id integer, --Unique ID for each payment\n  loan_id integer, --Foreign key referencing the loan\n  payment_amount real, --Amount of the payment\n  payment_date date, --Date of the payment\n  payment_details character varying --Information about the payment\n);\nCREATE TABLE loan_types (\n  loan_type_code character, --Unique code for loan type\n  loan_type_description character varying --Description of loan type\n);"
        result = get_table_names(md)
        expected_result = ["branches", "loans", "customers", "payments", "loan_types"]
        self.assertListEqual(result, expected_result)

    def test_get_table_names_2(self):
        md = "CREATE TABLE station (\n  station_code text, --Unique code for the train station. Eg S1, S2\n  station_name text, --Name of the train station\n  city text --City where the station is located (eg Munich)\n);\nCREATE TABLE fare (\n  fare_id integer, --Unique ID for the fare\n  one_way_cost integer, --Cost for a one-way ticket for this fare\n  round_trip_cost integer, --Cost for a round-trip ticket\n  fare_type text, --Type of fare such as 'standard', 'business', etc\n  restrictions text --Any restrictions or limitations for this fare\n);\nCREATE TABLE room (\n  room_code text, --Unique code for the room\n  model text, --Model of the room\n  manufacturer text, --Manufacturer of the room\n  fuel_source text, --One of Diesel, Electric, Hybrid\n  num_carriages bigint, --Number of carriages that the train has\n  year bigint, --Year built\n  horsepower bigint, --Horsepower of the train\n  capacity bigint, --Number of passengers that the train can hold\n  rpm bigint --train's engine revolutions per minute\n);\nCREATE TABLE booking (\n  booking_id integer, --Unique ID for the booking\n  room_code text, --Unique code for the room\n  check_in_time integer, --Check-in time in UNIX timestamp format\n  check_out_time integer, --Check-out time in UNIX timestamp format\n  stops integer, --Number of stops during the ride\n  distance integer, --Distance of the ride in km\n  total_time integer, --Total ride duration in minutes\n  operator_code text, --Code of company operating the train\n  from_station text, --Departure station code\n  to_station text --Arrival station code\n);\nCREATE TABLE ride_fare (\n  ride_id text, --Unique ID for the ride\n  fare_id text --Unique ID for the fare\n);\nCREATE TABLE stop (\n  ride_id integer, --Ride ID\n  stop_number integer, --Order of stop for ride\n  arrival_time integer, --Arrival time at stop in UNIX format\n  departure_time integer, --Departure time from stop in UNIX format\n  stop_duration integer, --Duration of stop in minutes\n  station_code text --Code for station where stop occurs\n);\nCREATE TABLE operator (\n  operator_code text, --Unique code for the train operator\n  name text --Name of the train operator\n);\nCREATE TABLE station_operator (\n  station_code text, --Station code\n  station_operator_code text --Station operator code\n);\nCREATE TABLE amenity (\n  amenity_id integer, --Unique ID for amenity\n  amenity_type text, --Eg. 'Lounge', 'WiFi', 'Cafe', 'Prayer Room'\n  station_code text --Station code where amenity is available\n);"
        result = get_table_names(md)
        expected_result = [
            "station",
            "fare",
            "room",
            "booking",
            "ride_fare",
            "stop",
            "operator",
            "station_operator",
            "amenity",
        ]
        self.assertListEqual(result, expected_result)

    def test_get_table_names_3(self):
        md = "CREATE TABLE teams (\n  games_team_id integer, --Unique identifier for each team\n  team_name text,\n  team_city text\n);\nCREATE TABLE players (\n  games_player_id integer, --Unique identifier for each player\n  player_name text,\n  games_team_id integer\n);\nCREATE TABLE games (\n  games_game_id integer, --Unique identifier for each game\n  home_team_id integer,\n  away_team_id integer,\n  game_date text, --format: yyyy-mm-dd\n  game_start_time integer, --timestamp in seconds since epoch\n  game_end_time bigint --Game timestamp when the game ended. Unix millis.\n);\nCREATE TABLE game_events (\n  games_event_id integer, --Unique identifier for each event\n  games_game_id integer,\n  event_type text,\n  event_time time without time zone,\n  event_details text\n);\nCREATE TABLE player_stats (\n  stat_id integer, --Unique identifier for the player stat record\n  games_player_id integer,\n  games_game_id integer,\n  stat_name text,\n  stat_value integer\n);\n"
        result = get_table_names(md)
        expected_result = ["teams", "players", "games", "game_events", "player_stats"]
        self.assertListEqual(result, expected_result)

    def test_get_table_names_4(self):
        md = "CREATE TABLE users (\n  id integer, --Unique identifier for each user\n  cust_code text, --Customer code assigned to each user\n  delivery_store integer, --ID of the delivery store associated with the user. This can be joined with id column from stores table\n  order_schedule smallint, --1 means Instore order, 2 means Pickup and Delivery order, 3 means Pickup order, 4 means Delivery order\n  join_date text, --Date when the user joined the service. in format YYYY-MM-DD\n  first_name text, --First name of the user\n  last_name text, --Last name of the user\n  address1 text, --First line of user's address\n  address2 text, --Second line of user's address\n  city text, --City where the user resides\n  state text, --State where the user resides\n  zipcode text, --Zipcode of the user's address\n  country text, --Country where the user resides\n  cust_map_pos text, --Map position of the user's location. This will contain latitude and longitude\n  email_id text, --Email address of the user\n  phone text, --Phone number of the user\n  country_code text, --Country code of the user's phone number\n  country_prefix_code text, --Country prefix code of the user's phone number\n  mobile text, --Mobile number of the user\n  status text, --Status of the user's account. This will contain value 'enable','disable'\n  last_login text --Date and time of the user's last login in format YYYY-MM-DD HH:MM:SS\n);\nCREATE TABLE stores (\n  id integer, --Unique identifier for each store\n  create_date text, --Date when the store was created in format 'YYYY-MM-DD'\n  store_name text, --Name of the store\n  short_name text, --Short name of the store\n  store_add1 text, --Address line 1 of the store\n  store_add2 text, --Address line 2 of the store\n  store_city text, --City where the store is located\n  store_state text, --State where the store is located\n  store_zip text, --Zip code of the store\n  store_email text, --Email address of the store\n  store_main text --Whether the store is the main store or not. If this column contains value 'yes' then it is considered as main store.\n);\nCREATE TABLE food_orders (\n  order_id integer, --Unique identifier for the item associated with an order.\n  order_invoice text, --Invoice number for the order.\n  cust_order_no text, --Order number associated with the customer.\n  order_date date, --Date when the order was placed.\n  customer_id integer, --Identifier for the customer who placed the order. This can be joined with id from the users table.\n  item_code text, --Identifier of the item/food.\n  order_service text, --Type of service ordered.\n  order_price text, --Price of the item.\n  order_qty text, --Quantity of the items.\n  product_discount_type text, --Type of discount applied to the product. This will include 'Percentage', 'Flat' or NULL values\n  product_discount_value numeric, --Value of the discount applied to the product.\n  actual_amount numeric, --Actual amount of the item\n  item_total_amount decimal --This column stores the total amount for the item, including any discounts or taxes applied. It is calculated by adding the price of all items and subtracting any discounts, then adding any applicable taxes.\n);\nCREATE TABLE machine_list (\n  id integer, --Unique identifier for each machine\n  store_id integer, --Identifier for the store where the machine is located. This can be joined with id from the stores table.\n  machine_name text, --Name of the machine\n  machine_type integer, --Type of the machine. This can be joined with id from the machine_type table.\n  machine_cost_per_cycle numeric, --Cost per cycle of the machine\n  time bigint, --Time is used for dryer in minutes or in case of washer it will 0\n  machine_status text --Current status of the machine. This includes following values: ['start', 'stop']\n);\nCREATE TABLE machine_type (\n  id integer, --Unique identifier for each machine type\n  name text --Name of the machine type. This includes following values: ['Washer' 'Dryer']\n);\nCREATE TABLE machine_processing (\n  id integer, --Unique identifier for each record\n  machine_id integer, --Identifier for the machine used for processing. This can be joined with id from the machine_list table.\n  order_id bigint, --Identifier for the order being processed. This can be joined with order_id column from table food_orders\n  start_time time, --Time when the processing started 'HH:MM:SS'\n  created_date date, --Date when the record was created in format YYYY-MM-DD\n  end_time time, --Time when the processing ended in format HH:MM:SS\n  updated_date date, --Date when the record was last updated in format YYYY-MM-DD\n  no_of_turns integer, --Number of turns taken by the machine for processing\n  machine_status text, --Status of the machine during processing. This includes following values: ['start', 'stop']\n  dryer_used_time time, --Time for which the dryer was used in format HH:MM:SS\n  cost_per_cycle text, --Cost incurred per processing cycle in format of serialize array() like 'a:3:{i:0;d:654;i:1;d:654;i:2;d:654;}'\n  machine_used_cost numeric, --Total cost incurred for using the machine\n  cash_payment smallint --Indicates if the payment was made in cash or not. This includes following values: [0, 1]\n);"
        result = get_table_names(md)
        expected_result = [
            "users",
            "stores",
            "food_orders",
            "machine_list",
            "machine_type",
            "machine_processing",
        ]
        self.assertListEqual(result, expected_result)

    def test_get_table_names_5(self):
        md = "CREATE TABLE Loan_Claims (\n  LoanCLM_Claim_ID integer, --Unique identifier for each claim\n  LoanCLM_Account_ID integer, --Foreign key referencing the account associated with the claim\n  LoanCLM_Date_Claim_Made date, --Date when the claim was made\n  LoanCLM_Date_Claim_Settled date, --Date when the claim was settled\n  LoanCLM_Amount_Claimed integer, --Amount claimed for the claim\n  LoanCLM_Amount_Settled integer --Amount settled for the claim\n);\nCREATE TABLE Loan_Payments (\n  LoanPay_Payment_ID integer, --Unique identifier for each payment\n  LoanPay_Settlement_ID integer, --Foreign key referencing the settlement associated with the payment\n  LoanPay_Payment_Method_Code character varying, --Code representing the payment method used\n  LoanPay_Date_Payment_Made date, --Date when the payment was made\n  LoanPay_Amount_Payment integer --Amount of the payment made\n);\nCREATE TABLE Customers (\n  Customer_ID integer, --Unique identifier for each customer\n  Customer_Details character varying --Details of the customer\n);\nCREATE TABLE Loan_Settlements (\n  LoanSTM_Settlement_ID integer, --Unique identifier for each settlement\n  LoanSTM_Claim_ID integer, --Foreign key referencing the claim associated with the settlement\n  LoanSTM_Date_Claim_Made date, --Date when the claim was made\n  LoanSTM_Date_Claim_Settled date, --Date when the claim was settled\n  LoanSTM_Amount_Claimed integer, --Amount claimed for the settlement\n  LoanSTM_Amount_Settled integer --Amount settled for the settlement\n  LoanSTM_Customer_Account_ID integer --Foreign key referencing the customer account associated with the settlement\n);\nCREATE TABLE Customer_Accounts (\n  CustACC_Account_ID integer, --Unique identifier for each account\n  CustACC_Customer_ID integer, --Foreign key referencing the customer associated with the account\n  CustACC_Account_Type_Code character, --Code representing the type of account\n  CustACC_Start_Date date, --Date when the account starts\n  CustACC_End_Date date --Date when the account ends\n);"
        result = get_table_names(md)
        expected_result = [
            "Loan_Claims",
            "Loan_Payments",
            "Customers",
            "Loan_Settlements",
            "Customer_Accounts",
        ]
        self.assertListEqual(result, expected_result)

    def test_get_table_names_no_tables(self):
        md = """
        CREATE VIEW view1 AS
        SELECT * FROM table1;
        """
        result = get_table_names(md)
        expected_result = []
        self.assertEqual(result, expected_result)

    def test_get_table_names_empty_string(self):
        md = ""
        result = get_table_names(md)
        expected_result = []
        self.assertEqual(result, expected_result)


class TestGenerateAliases(unittest.TestCase):
    def test_generate_aliases_1(self):
        table_names = [
            "Loan_Claims",
            "Loan_Payments",
            "Customers",
            "Loan_Settlements",
            "Customer_Accounts",
        ]
        result = generate_aliases(table_names)
        print(result)
        expected_result = """-- Loan_Claims AS lc
-- Loan_Payments AS lp
-- Customers AS c
-- Loan_Settlements AS ls
-- Customer_Accounts AS ca
"""
        self.assertEqual(result, expected_result)

    def test_generate_aliases_2(self):
        table_names = [
            "station",
            "fare",
            "room",
            "booking",
            "ride_fare",
            "stop",
            "operator",
            "station_operator",
            "amenity",
        ]
        result = generate_aliases(table_names)
        print(result)
        expected_result = """-- station AS s
-- fare AS f
-- room AS r
-- booking AS b
-- ride_fare AS rf
-- stop AS st
-- operator AS o
-- station_operator AS so
-- amenity AS a
"""
        self.assertEqual(result, expected_result)

    def test_generate_aliases_3(self):
        table_names = [
            "users",
            "stores",
            "food_orders",
            "machine_list",
            "machine_type",
            "machine_processing",
        ]
        result = generate_aliases(table_names)
        print(result)
        expected_result = """-- users AS u
-- stores AS s
-- food_orders AS fo
-- machine_list AS ml
-- machine_type AS mt
-- machine_processing AS mp
"""
        self.assertEqual(result, expected_result)

    def test_generate_aliases_4(self):
        table_names = [
            "teams1",
            "teams2",
            "teamPlayers",
            "teamGames",
            "gameEvents",
            "gamePlayer_stats",
        ]
        result = generate_aliases(table_names)
        print(result)
        expected_result = """-- teams1 AS t1
-- teams2 AS t2
-- teamPlayers AS tp
-- teamGames AS tg
-- gameEvents AS ge
-- gamePlayer_stats AS gs
"""
        self.assertEqual(result, expected_result)

    def test_generate_aliases_with_reserved_keywords(self):
        table_names = [
            "information_systems",
            "digital_options",
            "digital_ops_logs",
            "digital_ops_details",
        ]
        result = generate_aliases(table_names)
        print(result)
        expected_result = """-- information_systems AS inf
-- digital_options AS di
-- digital_ops_logs AS dol
-- digital_ops_details AS dod
"""
        self.assertEqual(result, expected_result)

    def test_generate_aliases_with_dots_and_underscores(self):
        table_names = ["db.schema.table1", "db.schema.table2", "db.schema.table3"]
        result = generate_aliases(table_names)
        print(result)
        expected_result = "-- db.schema.table1 AS t1\n-- db.schema.table2 AS t2\n-- db.schema.table3 AS t3\n"
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
