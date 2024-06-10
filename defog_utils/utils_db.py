# File for all db-related operations confined to psycopg2/sqlalchemy
import re
from typing import Any, Dict, List, Optional, Tuple

import psycopg2


like_pattern = r"LIKE[\s\S]*'"
escape_percent = r"LIKE '%'"

TEST_DB_NAME = "test"

creds_local_pg = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "postgres",
}

reserved_keywords = [
    "abs",
    "all",
    "and",
    "any",
    "avg",
    "as",
    "at",
    "asc",
    "bit",
    "by",
    "day",
    "dec",
    "do",
    "div",
    "end",
    "for",
    "go",
    "in",
    "is",
    "not",
    "or",
    "to",
]


def convert_data_type_postgres(dtype: str) -> str:
    """
    Convert the data type to be used in SQL queries to be postgres compatible
    """
    # remove any question marks from dtype and convert to lowercase
    dtype = re.sub(r"[\/\?]", "", dtype.lower())
    if dtype in {"int", "tinyint", "integer"}:
        return "integer"
    elif dtype == "double":
        return "double precision"
    elif dtype in {"varchar", "user-defined", "enum", "longtext", "string"}:
        return "text"
    elif dtype.startswith("number"):
        return "numeric"
    # if regex match dtype starting with datetime or timestamp, return timestamp
    elif dtype.startswith("datetime") or dtype.startswith("timestamp"):
        return "timestamp"
    elif dtype == "array":
        return "text[]"
    elif "byte" in dtype:
        return "text"
    else:
        return dtype


def normalize_table_name(table_name: str) -> str:
    """
    Normalize table name to be used in SQL queries
    """
    table_name = table_name.replace('"', "").replace("/", "")
    # check if table name has spaces or special characters
    if " " in table_name or "." in table_name:
        table_name = f'"{table_name}"'
    return table_name


def clean_column_name(column_name: str) -> str:
    """
    Normalize column name to be used in SQL queries
    """
    # remove /- from column name using re.sub
    column_name = re.sub(r"[\/\-\(\)]", "", column_name)
    if column_name.lower() == "group":
        column_name = '"group"'
    return column_name


def setup_test_db(creds: Dict[str, str], test_db: str = TEST_DB_NAME):
    """
    Create a test database given the credentials
    This is for testing custom metadata where the tables are not from defog_data/defog_data_private
    """
    db_name = creds.get("db_name", test_db)
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user=creds["user"],
            password=creds["password"],
            host=creds["host"],
            port=creds["port"],
        )
        cur = conn.cursor()
        conn.autocommit = True  # Disabling autocommit mode to ensure that CREATE DATABASE is not executed within a transaction else it will fail
        cur.execute(f"CREATE DATABASE {db_name}")
        print(f"Database {db_name} created")
    except psycopg2.errors.DuplicateDatabase:
        print("Database already exists")
    except Exception as e:
        print(e)
    finally:
        if "cur" in locals() or "cur" in globals():
            cur.close()
        if "conn" in locals() or "conn" in globals():
            conn.close()


def delete_test_db(creds: Dict[str, str], test_db: str = TEST_DB_NAME):
    """
    Delete a test database given the credentials
    This is for testing custom metadata where the tables are not from defog_data/defog_data_private
    """
    db_name = creds.get("db_name", test_db)
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user=creds["user"],
            password=creds["password"],
            host=creds["host"],
            port=creds["port"],
        )
        cur = conn.cursor()
        conn.autocommit = True  # Disabling autocommit mode to ensure that DROP DATABASE is not executed within a transaction else it will fail
        cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
        print(f"Database {db_name} deleted")
    except Exception as e:
        print(e)
    finally:
        if "cur" in locals() or "cur" in globals():
            cur.close()
        if "conn" in locals() or "conn" in globals():
            conn.close()


########################################
### Metadata Related Functions Below ###
########################################


def mk_create_table_ddl(table_name: str, columns: List[Dict[str, str]]) -> str:
    """
    Return a DDL statement for creating a table from a list of columns
    `columns` is a list of dictionaries with the following keys:
    - column_name: str
    - data_type: str
    - column_description: str
    """
    md_create = ""
    md_create += f"CREATE TABLE {table_name} (\n"
    for i, column in enumerate(columns):
        col_name = column["column_name"]
        # if column name has spaces and hasn't been wrapped in double quotes, wrap it in double quotes
        if " " in col_name and not col_name.startswith('"'):
            col_name = f'"{col_name}"'
        dtype = convert_data_type_postgres(column["data_type"])
        col_desc = column.get("column_description", "").replace("\n", " ")
        if col_desc:
            col_desc = f" --{col_desc}"
        if i < len(columns) - 1:
            md_create += f"  {col_name} {dtype},{col_desc}\n"
        else:
            # avoid the trailing comma for the last line
            md_create += f"  {col_name} {dtype}{col_desc}\n"
    md_create += ");\n"
    return md_create


def mk_create_ddl(md: Dict[str, List[Dict[str, str]]]) -> str:
    """
    Return a DDL statement for creating tables from a metadata dictionary
    `md` can have either a dictionary of schemas or a dictionary of tables.
    The former (with schemas) would look like this:
    {'schema1':
        {'table1': [
            {'column_name': 'col1', 'data_type': 'int', 'column_description': 'primary key'},
            {'column_name': 'col2', 'data_type': 'text', 'column_description': 'not null'},
            {'column_name': 'col3', 'data_type': 'text', 'column_description': ''},
        ],
        'table2': [
        ...
        ]},
    'schema2': ...}
    Schema is optional, and if not provided, the dictionary will be treated as
    a single schema of the form:
    {'table1': [
        {'column_name': 'col1', 'data_type': 'int', 'column_description': 'primary key'},
        {'column_name': 'col2', 'data_type': 'text', 'column_description': 'not null'},
        {'column_name': 'col3', 'data_type': 'text', 'column_description': ''},
    ],
    'table2': [
    ...
    ]}
    """
    md_create = ""
    for schema_or_table, contents in md.items():
        is_schema = isinstance(contents, dict)
        if is_schema:
            schema = schema_or_table
            tables = contents
            schema_ddl = f"CREATE SCHEMA IF NOT EXISTS {schema};\n"
            md_create += schema_ddl
            for table_name, table_dict in tables.items():
                schema_table_name = f"{schema}.{table_name}"
                md_create += mk_create_table_ddl(schema_table_name, table_dict)
        else:
            table_name = schema_or_table
            table_dict = contents
            md_create += mk_create_table_ddl(table_name, table_dict)
    return md_create


def mk_delete_ddl(md: Dict[str, Any]) -> str:
    """
    Return a DDL statement for deleting tables from a metadata dictionary
    `md` has the same structure as in `mk_create_ddl`
    This is for purging our temporary tables after creating them and testing the sql query
    """
    is_schema = False
    for _, contents in md.items():
        # check if the contents is a dictionary of tables or a list of tables
        is_schema = isinstance(contents, Dict)
        break
        
    if is_schema:
        md_delete = ""
        for schema, tables in md.items():
            schema = normalize_table_name(schema)
            md_delete += f"DROP SCHEMA IF EXISTS {schema} CASCADE;\n"
    else:
        md_delete = ""
        for table, _ in md.items():
            table = normalize_table_name(table)
            md_delete += f"DROP TABLE IF EXISTS {table} CASCADE;\n"
    return md_delete


def escape_percent(match):
    """Subroutine for escaping '%' in LIKE clauses in SQL queries."""
    # Extract the matched group
    group = match.group(0)
    # Replace '%' with '%%' within the matched group
    escaped_group = group.replace("%", "%%")
    # Return the escaped group
    return escaped_group


def fix_md(md: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Given a metadata dictionary, fix the following issues with the metadata:
    - Remove duplicate column names
    - Normalize table names
    - Convert data types to be used in SQL queries
    """
    md_new = {}
    for table, columns in md.items():
        column_names = []
        columns_new = []
        # ignore non-list columns
        if not isinstance(columns, list):
            continue
        for column in columns:
            column["data_type"] = convert_data_type_postgres(column["data_type"])
            # clean column name
            column_name = clean_column_name(column["column_name"])
            column["column_name"] = column_name
            if column_name.lower() in column_names:
                # if column name already exists, skip it
                continue
            else:
                column_names.append(column_name.lower())
                columns_new.append(column)
        table = normalize_table_name(table)
        md_new[table] = columns_new
    return md_new


def test_valid_md_sql(sql: str, md: dict, creds: Dict = None, conn = None, verbose: bool = False):
    """
    Test custom metadata and a sql query
    This will perform the following steps:
    1. Delete the tables in the metadata (to ensure that similarly named tables from previous tests are not used)
    2. Create the tables in the metadata. If any errors occur with the metadata, we return early.
    3. Run the sql query
    4. Delete the tables created
    If provided with the variable `conn`, this reuses the same database connection
    to avoid creating a new connection for each query. Otherwise it will connect
    via psycopg2 using the credentials provided (note that creds should set db_name)
    This will not manage `conn` in any way (eg closing `conn`) - it is left to 
    the caller to manage the connection.
    Returns tuple of (sql_valid, md_valid, err_message)
    """
    try:
        local_conn = False
        if conn is not None and conn.closed == 0:
            cur = conn.cursor()
        else:
            conn = psycopg2.connect(
                dbname=creds["db_name"],
                user=creds["user"],
                password=creds["password"],
                host=creds["host"],
                port=creds["port"],
            )
            local_conn = True
            cur = conn.cursor()
        delete_ddl = mk_delete_ddl(md)
        cur.execute(delete_ddl)
        if verbose:
            print(f"Deleted tables with: {delete_ddl}")
        create_ddl = mk_create_ddl(md)
        cur.execute(create_ddl)
        if verbose:
            print(f"Created tables with: {create_ddl}")
    except Exception as e:
        if "cur" in locals() or "cur" in globals():
            cur.close()
        if local_conn:
            conn.close()
        return False, False, e
    try:
        cur.execute(sql)
        results = cur.fetchall()
        if verbose:
            for row in results:
                print(row)
        delete_ddl = mk_delete_ddl(md)
        cur.execute(delete_ddl)
        if verbose:
            print(f"Deleted tables with: {delete_ddl}")
        cur.close()
        if local_conn:
            conn.close()
        return True, True, None
    except Exception as e:
        if "cur" in locals() or "cur" in globals():
            cur.close()
        if local_conn:
            conn.close()
        return False, True, e


def test_valid_md(
    sql: str, md: dict, creds: dict, verbose: bool = False, idx: str = ""
) -> Tuple[bool, Optional[Exception]]:
    """
    Test a sql query with custom metadata
    This will perform the following steps:
    1. Delete the test database if it exists
    2. Create the test database
    2. Create the tables in the metadata (to ensure that similarly named tables from previous tests are not used)
    3. Run the sql query
    4. Delete the test database
    We return the results of the query, and any errors that occur
    Set idx for parallel processing to avoid conflicts
    """
    try:
        # delete the test database if it exists
        delete_test_db(creds, TEST_DB_NAME + idx)
        # create database if non-existent
        setup_test_db(creds, TEST_DB_NAME + idx)
        conn = psycopg2.connect(
            dbname=TEST_DB_NAME + idx,
            user=creds["user"],
            password=creds["password"],
            host=creds["host"],
            port=creds["port"],
        )
        cur = conn.cursor()
        create_ddl = mk_create_ddl(md)
        cur.execute(create_ddl)
        if verbose:
            print(create_ddl)
        cur.execute(sql)
        results = cur.fetchall()
        if verbose:
            for row in results:
                print(row)
        valid, err = True, None
    except Exception as e:
        valid, err = False, e
    finally:
        if "cur" in locals() or "cur" in globals():
            cur.close()
        if "conn" in locals() or "conn" in globals():
            conn.close()
        delete_test_db(creds, TEST_DB_NAME + idx)
    return valid, err


def test_md(pruned_md: Dict, creds: Dict, verbose: bool = False, idx: str = ""):
    """
    Given a metadata dictionary and credentials, test if the tables can be created and deleted.
    Mostly to check for invalid formats/types in the metadata.
    Set index for parallel processing to avoid conflicts
    """
    try:
        # delete the test database if it exists
        delete_test_db(creds, TEST_DB_NAME + idx)
        # create database if non-existent
        setup_test_db(creds, TEST_DB_NAME + idx)
        conn = psycopg2.connect(
            dbname=TEST_DB_NAME + idx,
            user=creds["user"],
            password=creds["password"],
            host=creds["host"],
            port=creds["port"],
        )
        cur = conn.cursor()
        create_ddl = mk_create_ddl(pruned_md)
        cur.execute(create_ddl)
        return True
    except Exception as e:
        if verbose:
            print(e)
            # print stacktrace
            import traceback

            traceback.print_exc()
        if "cur" in locals() or "cur" in globals():
            cur.close()
        if "conn" in locals() or "conn" in globals():
            conn.close()
        delete_test_db(creds, TEST_DB_NAME + idx)
        return False


def parse_md(md_str: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Given a string of metadata formatted as a series of
    CREATE TABLE statements, return a dictionary of the metadata.
    If the table name contains a `.`, then the table is assumed to be in a schema,
    and we will return a dictionary of the form:
    {'schema1':
        {'table1': [
            {'column_name': 'col1', 'data_type': 'int', 'column_description': 'primary key'},
            {'column_name': 'col2', 'data_type': 'text', 'column_description': 'not null'},
            {'column_name': 'col3', 'data_type': 'text', 'column_description': ''},
        ],
        'table2': [
        ...
        ]},
    'schema2': ...}
    Otherwise, we will return a dictionary with the tables as the keys:
    {'table1': [
        {'column_name': 'col1', 'data_type': 'int', 'column_description': 'primary key'},
        {'column_name': 'col2', 'data_type': 'text', 'column_description': 'not null'},
        {'column_name': 'col3', 'data_type': 'text', 'column_description': ''},
    ],
    'table2': [
    ...
    ]}
    """
    md = {}
    if "CREATE TABLE" not in md_str:
        return md
    # split the md_str into individual tables on the ); delimiter
    for table_md_str in md_str.split(");"):
        if "CREATE TABLE" not in table_md_str:
            continue
        # split the table_md_str into the header and the columns
        header, columns_str = table_md_str.split("(", 1)
        table_name = header.split("CREATE TABLE", 1)[1].strip()
        if "." in table_name:
            schema, table_name = table_name.split(".", 1)
        else:
            schema = None
        columns = []
        for column_str in columns_str.split("\n"):
            column_str = column_str.strip()
            if not column_str:
                continue
            # split the column_str into the column name/type and the description
            column_str_split = re.split(r",?\s*--", column_str, 1)
            if len(column_str_split) == 1:
                # if no -- is found, then there is no description
                column_name_type = column_str.split(",", 1)[0]
                column_desc = ""
            else:
                # if -- is found, then the second part is the description
                column_name_type, column_desc = column_str_split
            if " " not in column_name_type:
                # if no space is found, then there is no data type, and we skip that invalid line
                continue
            # split the column_name_type into the column name and the data type
            # some data types have spaces between them, so we handle these cases separately,
            # and we assume the rest of the data types to have no spaces
            if "double precision" in column_name_type:
                column_name = column_name_type.split("double precision", 1)[0]
                column_type = "double precision"
            elif "character varying" in column_name_type:
                column_name = column_name_type.split("character varying", 1)[0]
                column_type = "character varying"
            elif "timestamp without time zone" in column_name_type:
                column_name = column_name_type.split("timestamp without time zone", 1)[
                    0
                ]
                column_type = "timestamp without time zone"
            elif "timestamp with time zone" in column_name_type:
                column_name = column_name_type.split("timestamp with time zone", 1)[0]
                column_type = "timestamp with time zone"
            elif "time with time zone" in column_name_type:
                column_name = column_name_type.split("time with time zone", 1)[0]
                column_type = "time with time zone"
            elif "time without time zone" in column_name_type:
                column_name = column_name_type.split("time without time zone", 1)[0]
                column_type = "time without time zone"
            else:
                column_name, column_type = column_name_type.rsplit(" ", 1)
            column_dict = {
                "column_name": column_name.strip(),
                "data_type": column_type.strip(),
                "column_description": column_desc.strip(),
            }
            columns.append(column_dict)
        if schema:
            if schema not in md:
                md[schema] = {}
            md[schema][table_name] = columns
        else:
            md[table_name] = columns
    return md


def get_table_names(md: str) -> List[str]:
    """
    Given a string of metadata formatted as a series of
    CREATE TABLE statements, return a list of table names in the same order as 
    they appear in the metadata.
    """
    table_names = []
    if "CREATE TABLE" not in md:
        return table_names
    for table_md_str in md.split(");"):
        if "CREATE TABLE " not in table_md_str:
            continue
        header = table_md_str.split("(", 1)[0]
        table_name = header.split("CREATE TABLE ", 1)[1].strip()
        table_names.append(table_name)
    return table_names


def generate_aliases_dict(
    table_names: List, reserved_keywords: List[str] = reserved_keywords
) -> Dict[str, str]:
    """
    Generate aliases for table names as a dictionary mapping of table names to aliases
    Aliases should always be in lower case
    """
    aliases = {}
    for original_table_name in table_names:
        if "." in original_table_name:
            table_name = original_table_name.rsplit(".", 1)[-1]
        else:
            table_name = original_table_name
        if "_" in table_name:
            # get the first letter of each subword delimited by "_"
            alias = "".join([word[0] for word in table_name.split("_")]).lower()
        else:
            # if camelCase, get the first letter of each subword
            # otherwise defaults to just getting the 1st letter of the table_name
            temp_table_name = table_name[0].upper() + table_name[1:]
            alias = "".join(
                [char for char in temp_table_name if char.isupper()]
            ).lower()
            # append ending numbers to alias if table_name ends with digits
            m = re.match(r".*(\d+)$", table_name)
            if m:
                alias += m.group(1)
        if alias in aliases.values() or alias in reserved_keywords:
            alias = table_name[:2].lower()
        if alias in aliases.values() or alias in reserved_keywords:
            alias = table_name[:3].lower()
        num = 2
        while alias in aliases.values() or alias in reserved_keywords:
            alias = table_name[0].lower() + str(num)
            num += 1

        aliases[original_table_name] = alias
    return aliases


def mk_alias_str(table_aliases: Dict[str, str]) -> str:
    """
    Given a dictionary of table names to aliases, return a string of aliases in the form:
    -- table1 AS t1
    -- table2 AS t2
    """
    aliases_str = ""
    for table_name, alias in table_aliases.items():
        aliases_str += f"-- {table_name} AS {alias}\n"
    return aliases_str


def generate_aliases(
    table_names: List, reserved_keywords: List[str] = reserved_keywords
) -> str:
    """
    Generate aliases for table names in a comment str form, eg
    -- table1 AS t1
    -- table2 AS t2
    """
    aliases = generate_aliases_dict(table_names, reserved_keywords)
    return mk_alias_str(aliases)
