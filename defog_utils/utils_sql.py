from dataclasses import dataclass
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sqlglot import errors, exp, parse, parse_one
import sqlparse

from .utils_feature import Features
from .utils_instructions import idk_re_pattern


@dataclass
class SchemaFeatures(Features):
    """
    Dataclass for tracking features extracted from a schema.
    """

    _prefix: str = "schema"

    num_tables: int = 0
    num_columns: int = 0
    num_comments: int = 0
    has_date: bool = False
    # true if a text/vachar column contains date-like strings
    has_date_text: bool = False
    # true if an integer column contains date-like integers (eg year, month, day)
    has_date_int: bool = False
    has_schema: bool = False  # true if CREATE TABLE schema.table ...
    has_catalog: bool = False  # true if CREATE TABLE db.schema.table ...
    quoted_table_names: bool = False
    quoted_column_names: bool = False
    join_hints: bool = False
    invalid_ddl: bool = False


@dataclass
class SqlFeatures(Features):
    """
    Dataclass for tracking features extracted from a SQL query.
    Can be converted into a dict using `.to_dict()` and passed to a DataFrame constructor.
    """

    _prefix: str = "sql"

    num_columns: int = 0
    num_tables: int = 0
    table_alias: bool = False
    joins: int = 0
    join_same: bool = False
    join_left: bool = False
    has_null: bool = False
    distinct: bool = False
    cte: int = 0
    union: bool = False
    case_condition: bool = False
    has_in: bool = False
    additive: bool = False
    ratio: bool = False
    round: bool = False
    order_by: bool = False
    limit: bool = False
    group_by: bool = False
    having: bool = False
    agg_count: bool = False
    agg_count_distinct: bool = False
    agg_sum: bool = False
    agg_avg: bool = False
    agg_min: bool = False
    agg_max: bool = False
    agg_var: bool = False
    agg_percentile: bool = False
    window_over: bool = False
    lag: bool = False
    rank: bool = False
    has_date: bool = False
    has_date_text: bool = False
    has_date_int: bool = False
    date_trunc: bool = False
    date_part: bool = False
    strftime: bool = False
    current_date_time: bool = False
    interval: bool = False
    date_time_type_conversion: bool = False
    date_time_format: bool = False
    generate_timeseries: bool = False
    string_concat: bool = False
    string_exact_match: bool = False
    string_case_insensitive_match: bool = False
    string_like_match: bool = False
    string_ilike_match: bool = False
    string_substring: bool = False
    string_regex: bool = False
    sorry: bool = False


# internal constants used for feature extraction
date_pattern = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"
time_pattern = r"^(0\d|1\d|2[0-3]):([0-5]\d):([0-5]\d)"
date_or_time_pattern = f"({date_pattern}|{time_pattern})"
date_column_pattern = r"(date|timestamp)(\s|$)"
variance_expressions = [
    exp.VariancePop,
    exp.Variance,
    exp.StddevPop,
    exp.StddevSamp,
]
current_date_time_expressions = [
    exp.CurrentDate,
    exp.CurrentDatetime,
    exp.CurrentTime,
    exp.CurrentTimestamp,
]
date_time_types = ["DATE", "TIMESTAMP"]
int_types = ["INT", "INTEGER", "BIGINT", "SMALLINT", "UINT", "UBIGINT"]
comparison_expressions = [
    # binary op with 2 children
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
    # ternary op with 3 children
    exp.Between,
]

### file paths for tracking certain outputs
# these 2 files record columns that are date-like but not of type date or timestamp
# it helps us understand what other date-like columns are present in our data
# and whether our regex rules capture them correctly. there is some inevitable
# race conditions but we only need the approximate count so this is acceptable.
deviant_columns_file = "deviant_columns.txt"
matched_columns_file = "matched_columns.txt"


def get_schema_features(schema_raw: str) -> Tuple[SchemaFeatures, Dict[str, str]]:
    """
    Extracts features from a SQL DDL string describing the schema by making a single
      pass through the parsed SQL abstract syntax tree (AST).
    Args:
        schema_raw: SQL schema string.
    Returns the SchemaFeatures object, and a dictionary parsed information
        about date columns in the schema. This is to avoid duplicate parsing.
    """
    features = SchemaFeatures()
    # preprocess the schema to avoid a few edge cases. these are inconsequential
    # for the purposes of feature extraction
    schema_raw = re.sub("```", "", schema_raw)
    split_ddl = schema_raw.split("Here is a list of joinable columns", 1)
    if len(split_ddl) >= 2 and split_ddl[1].strip():
        features.join_hints = True
    schema_processed = re.sub(
        "character varying", "varchar", split_ddl[0], flags=re.IGNORECASE
    )
    return_dict = {
        "date_type_date_time": set(),
        "date_type_int": set(),
        "date_type_text": set(),
    }
    # We use parse here instead of parse_one because each CREATE TABLE statement
    # is a separate root in the AST. `create` contains the root node for each
    # CREATE TABLE statement.
    try:
        parsed_ddl = parse(schema_processed)
        for create in parsed_ddl:
            schema = create.this
            table = schema.this
            if isinstance(table, exp.Table):
                features.num_tables += 1
                if table.db:
                    features.has_schema = True
                if table.catalog:
                    features.has_catalog = True
                if table.this.quoted:
                    features.quoted_table_names = True
            for column_def in schema.expressions:
                if isinstance(column_def, exp.ColumnDef):
                    features.num_columns += 1
                    if isinstance(column_def.kind, exp.DataType):
                        column_type = str(column_def.kind)
                        if column_type in date_time_types:
                            features.has_date = True
                            return_dict["date_type_date_time"].add(
                                column_def.name.lower()
                            )
                        elif has_date_in_name(column_def.name):
                            if column_type in int_types:
                                features.has_date_int = True
                                return_dict["date_type_int"].add(
                                    column_def.name.lower()
                                )
                                with open(matched_columns_file, "a") as f:
                                    f.write(column_def.name + " " + column_type + "\n")
                            elif is_text_type(column_type):
                                features.has_date_text = True
                                return_dict["date_type_text"].add(
                                    column_def.name.lower()
                                )
                                with open(matched_columns_file, "a") as f:
                                    f.write(column_def.name + " " + column_type + "\n")
                            else:
                                deviant = (
                                    f"column {column_def.name} is of type {column_type}"
                                )
                                with open(deviant_columns_file, "a") as f:
                                    f.write(deviant + "\n")

                    if column_def.this.quoted:
                        features.quoted_column_names = True
                    if column_def.comments:
                        features.num_comments += 1
    except errors.ParseError as e:
        features.invalid_ddl = True
        # fallback to a simpler feature extraction method
        for table in schema_processed.split(");"):
            if not table.strip():
                continue
            table_header, columns = table.strip().split("(", 1)
            table_name = table_header.split("CREATE TABLE ", 1)[1]
            features.num_tables += 1
            # count number of . in table name
            table_dots = table_name.count(".")
            if table_dots >= 1:
                features.has_schema = True
            if table_dots == 2:
                features.has_catalog = True
            if '"' in table_name:
                features.quoted_table_names = True
            for column in columns.strip().split("\n"):
                if column.strip():
                    features.num_columns += 1
                    if "--" in column:
                        features.num_comments += 1
                    column_name_type = column.split(",", 1)[0].split("--", 1)[0].strip()
                    column_name = column_name_type.split(" ", 1)[0].lower()
                    column_type = column_name_type.split(" ", 1)[-1].lower()
                    if re.search(date_column_pattern, column_name_type, re.IGNORECASE):
                        features.has_date = True
                        return_dict["date_type_date_time"].add(column_name)
                    elif has_date_in_name(column_name):
                        if column_type in int_types:
                            features.has_date_int = True
                            return_dict["date_type_int"].add(column_name)
                            with open(matched_columns_file, "a") as f:
                                f.write(column + "\n")
                        elif is_text_type(column_type):
                            features.has_date_text = True
                            return_dict["date_type_text"].add(column_name)
                            with open(matched_columns_file, "a") as f:
                                f.write(column + "\n")
                        else:
                            deviant = f"column {column_name} is of type {column_type}"
                            with open(deviant_columns_file, "a") as f:
                                f.write(deviant + "\n")
                    if '"' in column_name_type:
                        features.quoted_column_names = True
    return features, return_dict


def get_sql_features(
    sql: str,
    md_cols: Optional[Set[str]] = None,
    md_tables: Optional[Set[str]] = None,
    extra_column_info: Optional[Dict[str, str]] = None,
    dialect: str = "postgres",
) -> SqlFeatures:
    """
    Extracts features from a SQL query string by making a single pass through the parsed SQL abstract syntax tree (AST).
    Args:
        sql: SQL query string.
        md_cols: Set of column names from the database schema. Used to filter out derived column names if provided.
        md_tables: Set of table names from the database schema. Used to filter out derived table names if provided.
        extra_column_info: Dictionary containing additional information about date columns in the schema. This is
            precomputed when getting schema features and helps us avoid duplicate parsing of the schema. It should
            contain the following keys:
                - date_type_date_time: Set of column names that are of type DATE or TIMESTAMP.
                - date_type_int: Set of column names that are of type INT, INTEGER, BIGINT, SMALLINT, UINT, or UBIGINT
                    and contain date-like integers (eg year, month, day).
                - date_type_text: Set of column names that are of type TEXT, VARCHAR, or CHAR and contain date-like strings.
    Returns the SqlFeatures object.
    Known issue: there are 2 similarly named columns from different tables used, we will only count it once for simplicity.
    """
    # dictionary for tracking features
    features = SqlFeatures()
    if is_sorry_sql(sql):
        features.sorry = True
        return features
    # extract non-parseable features that would result in a ParseError
    # find and replace ~* or ~ with LIKE/ILIKE
    # note: REGEXP_LIKE and REGEXP_ILIKE require putting the preceding and following expressions
    # in brackets which entail some parsing, which we avoid doing so here for simplicity
    if " ~* '" in sql:
        sql = sql.replace(" ~* '", " ILIKE '")
    if " ~ '" in sql:
        sql = sql.replace(" ~ '", " LIKE '")
    sql = re.sub("character varying", "varchar", sql, flags=re.IGNORECASE)
    parsed = parse_one(sql, dialect)
    # internal state for computing various summarized/derived quantities
    columns_in_sql = set()
    tables_in_sql = set()
    # get respective sets from extra_column_info for easier reference below
    if extra_column_info is None:
        date_cols, date_int_cols, date_text_cols = set(), set(), set()
    else:
        date_cols = extra_column_info.get("date_type_date_time", set())
        date_int_cols = extra_column_info.get("date_type_int", set())
        date_text_cols = extra_column_info.get("date_type_text", set())
    # make 1 single pass through the full SQL abstract syntax tree (AST)
    for node in parsed.walk():
        if isinstance(node, exp.Column):
            column_name_lower = node.name.lower()
            columns_in_sql.add(column_name_lower)
            if date_cols and column_name_lower in date_cols:
                features.has_date = True
            if date_int_cols and column_name_lower in date_int_cols:
                features.has_date_int = True
            if date_text_cols and column_name_lower in date_text_cols:
                features.has_date_text = True
        elif isinstance(node, exp.Table):
            tables_in_sql.add(node.name.lower())
            if node.alias:
                features.table_alias = True
        elif isinstance(node, exp.Join):
            features.joins += 1
            # from_node is sibling node of the join node
            from_node = node.parent.find(exp.From)
            join_table = node.find(exp.Table)
            from_table = from_node.find(exp.Table)
            # test if join_table/from_table exists first since joins can be on expressions
            # like generate_series(...)
            if join_table and from_table and from_table.name == join_table.name:
                features.join_same = True
            if node.side == "LEFT":
                features.join_left = True
        elif isinstance(node, exp.Null):
            features.has_null = True
        elif isinstance(node, exp.Distinct):
            features.distinct = True
        elif isinstance(node, exp.CTE):
            features.cte += 1
        elif isinstance(node, exp.Union):
            features.union = True
        elif isinstance(node, exp.Case):
            features.case_condition = True
        elif isinstance(node, exp.In):
            features.has_in = True
        elif isinstance(node, exp.Add) or isinstance(node, exp.Sub):
            features.additive = True
        elif isinstance(node, exp.Div):
            features.ratio = True
        elif isinstance(node, exp.Round):
            features.round = True
        elif isinstance(node, exp.Order):
            features.order_by = True
        elif isinstance(node, exp.Limit):
            features.limit = True
        elif isinstance(node, exp.Group):
            features.group_by = True
        elif isinstance(node, exp.Having):
            features.having = True
        elif isinstance(node, exp.Count):
            features.agg_count = True
            for child in node.flatten():
                if isinstance(child, exp.Distinct):
                    features.agg_count_distinct = True
                    break
        elif isinstance(node, exp.Sum):
            features.agg_sum = True
        elif isinstance(node, exp.Avg):
            features.agg_avg = True
        elif isinstance(node, exp.Min):
            features.agg_min = True
        elif isinstance(node, exp.Max):
            features.agg_max = True
        elif type(node) in variance_expressions:
            features.agg_var = True
        elif isinstance(node, exp.PercentileCont) or isinstance(
            node, exp.PercentileDisc
        ):
            features.agg_percentile = True
        elif isinstance(node, exp.Window):
            features.window_over = True
        elif isinstance(node, exp.Lag):
            features.lag = True
        elif isinstance(node, exp.RowNumber):
            features.rank = True
        elif isinstance(node, exp.DateTrunc) or isinstance(node, exp.TimestampTrunc):
            features.date_trunc = True
        elif isinstance(node, exp.StrToTime):
            features.date_time_type_conversion = True
        elif isinstance(node, exp.Extract):
            features.date_part = True
        elif type(node) in current_date_time_expressions:
            features.current_date_time = True
        elif isinstance(node, exp.Interval):
            features.interval = True
        elif isinstance(node, exp.Date):
            features.date_time_type_conversion = True
        elif isinstance(node, exp.Cast):
            for c in node.flatten():
                if isinstance(c, exp.DataType) and str(c) in date_time_types:
                    features.date_time_type_conversion = True
                    break
        elif isinstance(node, exp.ToChar) or isinstance(node, exp.TimeToStr):
            features.date_time_format = True
            if isinstance(node, exp.TimeToStr):
                features.strftime = True
        elif isinstance(node, exp.GenerateSeries):
            features.generate_timeseries = True
        elif isinstance(node, exp.DPipe):
            features.string_concat = True
        # we first detect if a matching expression is present, then check if either side of the comparison is a string
        elif type(node) in comparison_expressions:
            has_string = False
            has_upper_lower = False
            has_date = False
            for child in node.flatten():
                if isinstance(child, exp.Literal) and child.is_string:
                    if is_date_or_time_str(child.name):
                        has_date = True
                    else:
                        has_string = True
                elif isinstance(child, exp.Upper) or isinstance(child, exp.Lower):
                    has_upper_lower = True
            if has_string:
                if has_upper_lower:
                    features.string_case_insensitive_match = True
                else:
                    features.string_exact_match = True
            if has_date:
                features.has_date = True
        elif isinstance(node, exp.Like):
            features.string_like_match = True
        elif isinstance(node, exp.ILike):
            features.string_ilike_match = True
        elif isinstance(node, exp.Substring):
            features.string_substring = True
        elif isinstance(node, exp.RegexpILike) or isinstance(node, exp.RegexpLike):
            features.string_regex = True

        # other non-defined expressions in sqlglot
        elif isinstance(node, exp.Anonymous):
            node_name = node.name.lower()
            if (
                node_name == "rank"
                or node_name == "dense_rank"
                or node_name == "percent_rank"
            ):
                features.rank = True
            elif node_name == "date_part":
                features.date_part = True
            elif node_name == "strftime":
                features.strftime = True
            elif node_name == "now":
                features.current_date_time = True
            elif node_name == "to_date" or node_name == "to_timestamp":
                features.date_time_type_conversion = True
        # other non-defined non-Anonymous expressions in sqlglot
        else:
            if "'now'" in str(node).lower():
                features.current_date_time = True

    if md_cols and md_tables:
        md_cols = set([c.lower() for c in md_cols])
        md_tables = set([t.lower() for t in md_tables])
        columns_in_sql = columns_in_sql.intersection(md_cols)
        tables_in_sql = tables_in_sql.intersection(md_tables)
    features.num_columns = len(columns_in_sql)
    features.num_tables = len(tables_in_sql)
    return features


def is_date_or_time_str(s: str) -> bool:
    m = re.match(date_or_time_pattern, s)
    return bool(m)


def has_date_in_name(s: str) -> bool:
    return bool(re.search(r"(year|quarter|month|week|day)", s))


def is_text_type(s) -> bool:
    return bool(re.search(r"(text|varchar\(?|char)", str(s), re.IGNORECASE))


def is_sorry_sql(query):
    """
    Detects if the query contains specific phrases indicating an inability to answer
    due to a lack of data, as opposed to normal SQL queries.

    Returns True if the special phrase is detected, False otherwise.
    """
    return bool(re.search(idk_re_pattern, query))


def add_space_padding(text: str) -> str:
    # Add space before opening parenthesis if not preceded by a space
    text = re.sub(r"(?<!\s)\(", r" (", text)
    # Add space after opening parenthesis if not followed by a space
    text = re.sub(r"\((?!\s)", r"( ", text)
    # Add space before closing parenthesis if not preceded by a space
    text = re.sub(r"(?<!\s)\)", r" )", text)
    # Add space after closing parenthesis if not followed by a space
    text = re.sub(r"\)(?!\s)", r") ", text)
    # Remove any double spaces introduced by the substitutions
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def normalize_sql(sql: str, pad_brackets: bool = False) -> str:
    """
    Normalize SQL query string by converting all keywords to uppercase and
    stripping whitespace.
    We can pad spaces before and after brackets to avoid glueing characters/part
    of words into a single token.
    """
    # remove ; if present first
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()
    sql = sqlparse.format(
        sql, keyword_case="upper", strip_whitespace=True, strip_comments=True
    )
    # add back ;
    if not sql.endswith(";"):
        sql += ";"
    if pad_brackets:
        sql = add_space_padding(sql)
    sql = re.sub(r" cast\(", " CAST(", sql)
    sql = re.sub(r" case when ", " CASE WHEN ", sql)
    sql = re.sub(r" then ", " THEN ", sql)
    sql = re.sub(r" else ", " ELSE ", sql)
    sql = re.sub(r" end ", " END ", sql)
    sql = re.sub(r" as ", " AS ", sql)
    sql = re.sub(r"::float", "::FLOAT", sql)
    sql = re.sub(r"::date", "::DATE", sql)
    sql = re.sub(r"::timestamp", "::TIMESTAMP", sql)
    sql = re.sub(r" float", " FLOAT", sql)
    sql = re.sub(r" date\)", " DATE)", sql)
    sql = re.sub(r" date_part\(", " DATE_PART(", sql)
    sql = re.sub(r" date_trunc\(", " DATE_TRUNC(", sql)
    sql = re.sub(r" timestamp\)", " TIMESTAMP)", sql)
    sql = re.sub(r"to_timestamp\(", "TO_TIMESTAMP(", sql)
    sql = re.sub(r"count\(", "COUNT(", sql)
    sql = re.sub(r"sum\(", "SUM(", sql)
    sql = re.sub(r"avg\(", "AVG(", sql)
    sql = re.sub(r"min\(", "MIN(", sql)
    sql = re.sub(r"max\(", "MAX(", sql)
    sql = re.sub(r"distinct\(", "DISTINCT(", sql)
    sql = re.sub(r"nullif\(", "NULLIF(", sql)
    sql = re.sub(r"extract\(", "EXTRACT(", sql)
    return sql


def fix_comma(cols: List[str]) -> List[str]:
    """
    Given list of strings containing column info, fix commas so that each string
    (except the last) ends with a comma. If string has a sql comment "--", then
    check for comma before comment and add if necessary.
    """
    fixed_cols = []
    for col in cols:
        # check if string has a comment
        if "--" in col:
            # check if comma is just before comment
            if not re.search(r",\s*--", col):
                # use re.sub to replace (any whitespace)-- with , --
                col = re.sub(r"\s*--", ", --", col)
        # check if string ends with comma (optionally with additional spaces)
        elif not re.search(r",\s*$", col):
            # end with comma if not present
            col = re.sub(r"\s*$", ",", col)
        fixed_cols.append(col)
    # for the last col, we want to remove the comma
    last_col = fixed_cols[-1]
    if "--" in last_col:
        # check if comma is after a word/closing brace, followed by spaces before -- and remove if present

        pre_comment, after_comment = last_col.split("--", 1)
        # check if pre_comment ends with a comma with optional spaces
        if re.search(r",\s*$", pre_comment):
            pre_comment = re.sub(r",\s*$", "", pre_comment)
            # remove any trailing spaces in pre_comment
            pre_comment = pre_comment.rstrip()
            last_col = pre_comment + " --" + after_comment
    # if last_col ends with a comma with optional spaces, remove it
    elif re.search(r",\s*$", last_col):
        last_col = re.sub(r",\s*$", "", last_col)
        # remove any trailing spaces in last_col
        last_col = last_col.rstrip()
    fixed_cols[-1] = last_col
    return fixed_cols


def shuffle_table_metadata(md_str: str, seed: Optional[int] = None) -> str:
    """
    Shuffles the column names within each table.
    Shuffles the table order.
    Seed is used to ensure reproducibility.
    If iterating through rows of a dataframe, the seed can be None and we assume that
    the seed has been set externally and do not reset it.
    """
    if seed is not None:
        np.random.seed(seed)
    parts = md_str.split("Here is a list of joinable columns:", 1)
    if len(parts) == 1:
        logging.debug(f"md_str does not contain join statements")
        join_statements = ""
    elif len(parts) == 2:
        md_str, join_statements = parts
        join_statements = join_statements.strip()
        if (join_statements != "") and ("can be joined with" not in join_statements):
            logging.info(
                f"join_statements does not contain 'join':\n\"{join_statements}\"\ndropping this join statement."
            )
            join_statements = ""
    schema_str_list = []
    while "CREATE SCHEMA" in md_str:
        logging.debug(
            f"md_str contains CREATE SCHEMA statements\nmd_str: {md_str}\nshuffling columns within tables only"
        )
        # use regex to find and extract line with CREATE SCHEMA
        schema_match = re.search(r"CREATE SCHEMA.*;", md_str)
        if schema_match:
            schema_line = schema_match.group(0)
            schema_str_list.append(schema_line)
            md_str = md_str.replace(schema_line, "")
        
    md_table_list = md_str.split(");")
    md_table_shuffled_list = []
    for md_table in md_table_list:
        md_table = md_table.strip()
        if md_table in {"", "```"}:
            continue
        if "CREATE TABLE" not in md_table:
            continue
        row_split = md_table.split("\n")
        header = row_split[0]
        cols = row_split[1:]
        shuffled_cols = cols.copy()
        if shuffled_cols == []:
            logging.info(
                f"md_str has a table with no columns\nmd_str: {md_table}\nheader: {header}\nskipping"
            )
            continue
        np.random.shuffle(shuffled_cols)
        shuffled_cols = fix_comma(shuffled_cols)
        shuffled_cols_str = "\n".join(shuffled_cols)
        md_table_shuffled = f"{header}\n{shuffled_cols_str}\n);"
        md_table_shuffled_list.append(md_table_shuffled)
    # shuffle md_table_shuffled_list
    np.random.shuffle(md_table_shuffled_list)
    md_table_shuffled_str = "\n".join(md_table_shuffled_list)
    # add back join statements if present in data
    if join_statements != "":
        md_table_shuffled_str += (
            "\nHere is a list of joinable columns:\n" + join_statements
        )
    if schema_str_list != []:
        md_table_shuffled_str = "\n".join(schema_str_list) + "\n" + md_table_shuffled_str
    return md_table_shuffled_str


def replace_alias(
    sql: str, new_alias_map: Dict[str, str], dialect: str = "postgres"
) -> str:
    """
    Replaces the table aliases in the SQL query with the new aliases provided in 
    the new_alias_map.
    `new_alias_map` is a dict of table_name -> new_alias.
    Note that aliases are always in lowercase, and will be converted to lowercase
    if necessary.
    """
    parsed = parse_one(sql, dialect=dialect)
    existing_alias_map = {}
    # save and update the existing table aliases
    for node in parsed.walk():
        if isinstance(node, exp.Table):
            table_name = node.name
            # save the existing alias if present
            if node.alias:
                node_alias = node.alias.lower()
                existing_alias_map[node_alias] = table_name
                # set the alias to the new alias if it exists in the new_alias_map
                if table_name in new_alias_map:
                    node.set("alias", new_alias_map[table_name])
            else:
                node.set("alias", new_alias_map.get(table_name, table_name))
    # go through each column, and if it has a table alias, replace it with the new alias
    for node in parsed.walk():
        if isinstance(node, exp.Column):
            if node.table:
                node_table = node.table.lower()
                # if in existing alias map, set the table to the new alias
                if node_table in existing_alias_map:
                    original_table_name = existing_alias_map[node_table]
                    if original_table_name in new_alias_map:
                        node.set("table", new_alias_map[original_table_name])
                # else if in new alias map, set the table to the new alias
                elif node_table in new_alias_map:
                    node.set("table", new_alias_map[node_table])
    return parsed.sql(dialect, normalize_functions="upper", comments=False)
