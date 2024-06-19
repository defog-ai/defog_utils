import unittest
from defog_utils.defog_utils.utils_sql import (
    add_space_padding,
    fix_comma,
    get_schema_features,
    get_sql_features,
    is_date_or_time_str,
    is_sorry_sql,
    replace_alias,
    shuffle_table_metadata,
)


class TestGetSqlFeatures(unittest.TestCase):
    def setUp(self):
        # Mock metadata
        self.md_cols = {
            "sbtickersymbol",
            "sbdpdate",
            "sbdpLow",
            "sbdpHigh",
            "sbDpTickerId",
        }
        self.md_tables = {"sbdailyprice", "sbticker"}
        self.empty_extra_column_info = {
            "date_type_date_time": set(),
            "date_type_int": set(),
            "date_type_text": set(),
        }

    def test_table_alias(self):
        sql = "SELECT t1.column1 as c1 FROM table1 t1"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.table_alias)
        sql = "SELECT column1 as c1 FROM table1 t1"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.table_alias)
        sql = "SELECT table1.column1 as c1 FROM table1"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.table_alias)
        sql = "SELECT column1 as c1 FROM table1"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.table_alias)

    def test_join_same(self):
        sql = "SELECT * FROM table1 a JOIN table1 b ON a.id < b.id"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.join_same)
        sql = "SELECT * FROM table1 a JOIN table2 b ON a.id < b.id"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.join_same)

    def test_join_left(self):
        sql = "SELECT * FROM table1 t1 LEFT JOIN table2 t2 ON t1.id = t2.id"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.join_left)
        sql = "SELECT * FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.join_left)
    
    def test_addition(self):
        sql = "SELECT column1 + column2 FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.addition)

    def test_subtraction(self):
        sql = "SELECT column1 - column2 FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.subtraction)

    def test_cte(self):
        sql = "WITH cte1 AS (SELECT * FROM table), cte2 AS (SELECT * FROM cte1), cte3 AS (SELECT * FROM cte2 JOIN cte1 ON cte1.id = cte2.id) SELECT * FROM cte3"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertEqual(features.cte, 3)
        sql = "SELECT * FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertEqual(features.cte, 0)

    def test_order_by_limit(self):
        sql = "SELECT * FROM table ORDER BY column LIMIT 10"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.order_by and features.limit)
        sql = "SELECT * FROM table ORDER BY column"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.order_by)

    def test_group_by_having(self):
        sql = "SELECT column, COUNT(*) FROM table GROUP BY column HAVING COUNT(*) > 1"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.group_by and features.having)
        sql = "SELECT column, COUNT(*) FROM table GROUP BY column"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.group_by)
        self.assertFalse(features.having)

    def test_agg_min_max(self):
        sql = "SELECT MIN(column), MAX(column) FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.agg_min and features.agg_max)
        sql = "SELECT MIN(column) FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.agg_min)
        self.assertFalse(features.agg_max)
        sql = "SELECT MAX(column) FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.agg_min)
        self.assertTrue(features.agg_max)

    def test_window_over(self):
        sql = "SELECT COUNT(*) OVER () FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.window_over)

    def test_case_condition(self):
        sql = "SELECT CASE WHEN condition THEN result ELSE other END FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.case_condition)

    def test_ratio(self):
        sql = "SELECT column1 / column2 FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.ratio)

    def test_agg_count_distinct(self):
        sql = "SELECT COUNT(DISTINCT column) FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.agg_count_distinct)
        self.assertTrue(features.agg_count)
        self.assertTrue(features.distinct)

    def test_distinct(self):
        sql = "SELECT DISTINCT column1, column2 FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.distinct)
        self.assertFalse(features.agg_count_distinct)

    def test_has_date_columns(self):
        # no date columns present
        sql = "SELECT column1 FROM table WHERE column2 = column3"
        features = get_sql_features(
            sql, self.md_cols, self.md_tables, self.empty_extra_column_info
        )
        self.assertFalse(features.has_date)
        # date columns present
        extra_column_info = {
            "date_type_date_time": {"column2", "column10"},
            "date_type_int": set(),
            "date_type_text": {"column4"},
        }
        features = get_sql_features(
            sql, self.md_cols, self.md_tables, extra_column_info
        )
        self.assertTrue(features.has_date)
        self.assertFalse(features.has_date_text)
        self.assertFalse(features.has_date_int)

        sql = "SELECT column2 FROM table WHERE column1 < column3 AND column4 = '2023-01-01'"
        features = get_sql_features(
            sql, self.md_cols, self.md_tables, extra_column_info
        )
        self.assertTrue(features.has_date)
        self.assertTrue(features.has_date_text)
        self.assertFalse(features.has_date_int)
        self.assertTrue(features.date_literal)
    
    def test_date_literal(self):
        sql = "SELECT column1 FROM table WHERE column2 = '2023-01-01'"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.date_literal)
        self.assertFalse(features.string_case_insensitive_match)
        self.assertFalse(features.string_exact_match)
        sql = "SELECT column1 FROM table WHERE column2 < '2023-01-01 12:34:56'"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.date_literal)
        self.assertFalse(features.string_case_insensitive_match)
        self.assertFalse(features.string_exact_match)
        sql = "SELECT column1 FROM table WHERE column2 >= '12:34:56'"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.date_literal)
        self.assertFalse(features.string_case_insensitive_match)
        self.assertFalse(features.string_exact_match)

    def test_date_trunc(self):
        sql = "SELECT DATE_TRUNC('day', column) FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.date_trunc)
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.date_trunc)

    def test_strftime(self):
        sql = "SELECT STRFTIME('%Y-%m-%d', column) FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.strftime)
        self.assertFalse(features.date_time_format)
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.strftime)
        self.assertFalse(features.date_time_format)

    def test_date_part(self):
        sql = "SELECT EXTRACT(YEAR FROM column) FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.date_part)
        sql = "SELECT DATE_PART('year', column) FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.date_part)

    def test_date_comparison(self):
        sql_left = "SELECT * FROM table WHERE column1 > '2023-01-01'"
        sql_right = "SELECT * FROM table WHERE '2023-01-01' < column1"
        extra_column_info = {
            "date_type_date_time": {"column1"},
            "date_type_int": set(),
            "date_type_text": set(),
        }
        for sql in [sql_left, sql_right]:
            features_with_empty_col_info = \
                get_sql_features(sql, self.md_cols, self.md_tables, self.empty_extra_column_info)
            self.assertFalse(features_with_empty_col_info.date_comparison)
            features_with_date_col_info = \
                get_sql_features(sql, self.md_cols, self.md_tables, extra_column_info)
            self.assertTrue(features_with_date_col_info.date_comparison)

    def test_date_sub_date(self):
        sql = "SELECT column1 - column2 FROM table"
        extra_column_info_both = {
            "date_type_date_time": {"column1", "column2"},
            "date_type_int": set(),
            "date_type_text": set(),
        }
        extra_column_info_1 = {
            "date_type_date_time": {"column1"},
            "date_type_int": set(),
            "date_type_text": set(),
        }
        extra_column_info_2 = {
            "date_type_date_time": {"column2"},
            "date_type_int": set(),
            "date_type_text": set(),
        }
        # date - date
        features = get_sql_features(sql, self.md_cols, self.md_tables, extra_column_info_both)
        self.assertTrue(features.date_sub_date)
        self.assertFalse(features.date_sub)
        # x - date or date - x
        for ec in [extra_column_info_1, extra_column_info_2]:
            features = get_sql_features(sql, self.md_cols, self.md_tables, ec)
            self.assertFalse(features.date_sub_date)
            self.assertTrue(features.date_sub)
        # x - x
        features = get_sql_features(sql, self.md_cols, self.md_tables, self.empty_extra_column_info)
        self.assertFalse(features.date_sub_date)
        self.assertFalse(features.date_sub)

    def test_current_date_time(self):
        sql = "SELECT col_date - CURRENT_DATE FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.current_date_time)
        sql = "SELECT CURRENT_TIMESTAMP - col_date FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.current_date_time)
        sql = "SELECT NOW() - ts FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.current_date_time)
        sql = "WITH RECURSIVE week_series AS (SELECT DATE('2023-01-01') AS week UNION ALL SELECT DATE(week, '+7 days') FROM week_series WHERE week < DATE('now', 'weekday 0', '-7 days')), order_item_counts AS (SELECT o.order_status, strftime('%Y-%W', o.date_order_placed) AS week, o.order_id, COUNT(oi.order_item_id) AS order_item_count FROM orders AS o JOIN order_items AS oi ON o.order_id = oi.order_id WHERE o.date_order_placed >= '2023-01-01' GROUP BY o.order_status, week, o.order_id), avg_item_counts AS (SELECT o.order_status, o.week, ROUND(AVG(o.order_item_count), 3) AS avg_item_count FROM order_item_counts AS o GROUP BY o.order_status, o.week), status_week_range AS (SELECT s.order_status, w.week FROM (SELECT DISTINCT order_status FROM orders) AS s CROSS JOIN week_series AS w) SELECT s.order_status, s.week, COALESCE(a.avg_item_count, 0) AS avg_item_count FROM status_week_range AS s LEFT JOIN avg_item_counts AS a ON s.order_status = a.order_status AND s.week = a.week ORDER BY s.order_status, s.week;"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.current_date_time)
        sql = "SELECT * FROM col_now"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.current_date_time)

    def test_interval(self):
        sql = "SELECT column + INTERVAL '1 day' FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.interval)

    def test_date_time_type_conversion(self):
        sql = "SELECT CAST(column AS TIMESTAMP) FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.date_time_type_conversion)
        sql = "SELECT CAST(column AS DATE) FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.date_time_type_conversion)
        sql = "SELECT TO_DATE(column, 'YYYY-MM-DD') FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.date_time_type_conversion)
        sql = "SELECT TO_TIMESTAMP(column, 'YYYY-MM-DD') FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.date_time_type_conversion)
        sql = "SELECT column::TIMESTAMP FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.date_time_type_conversion)
        sql = "SELECT column::DATE FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.date_time_type_conversion)
        sql = "SELECT DATE(column) FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.date_time_type_conversion)

    def test_date_time_format(self):
        sql = "SELECT TO_CHAR(column, 'YYYY-MM-DD') FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.date_time_format)
        self.assertTrue(features.strftime)
        sql = "SELECT TO_DATE(column, 'YYYY-MM-DD') FROM table"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertFalse(features.date_time_format)
        self.assertFalse(features.strftime)

    def test_generate_timeseries(self):
        sql = "SELECT generate_series(1, 10)"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.generate_timeseries)
        sql = "SELECT generate_series('2023-01-01'::DATE, '2023-01-10'::DATE, '1 day')"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.generate_timeseries)
        sql = "SELECT generate_series('2023-01-01'::TIMESTAMP, '2023-01-10'::TIMESTAMP, '1 day')"
        features = get_sql_features(sql, self.md_cols, self.md_tables, dialect="postgres")
        self.assertTrue(features.generate_timeseries)

    def test_string_concat(self):
        sql = "SELECT name || ' ' || description FROM table1"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.string_concat)

    def test_string_exact_match(self):
        sql = "SELECT * FROM table1 WHERE name = 'Exact Match'"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertTrue(features.string_exact_match)
        self.assertFalse(features.string_case_insensitive_match)
        self.assertFalse(features.string_like_match)
        self.assertFalse(features.string_ilike_match)
        self.assertFalse(features.string_substring)
        self.assertFalse(features.string_regex)

        # Test case to ensure it does not falsely trigger
        sql = "SELECT * FROM table1 WHERE date = '2023-01-01'"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.string_exact_match)
        self.assertFalse(features.string_case_insensitive_match)
        self.assertFalse(features.string_like_match)
        self.assertFalse(features.string_ilike_match)
        self.assertFalse(features.string_substring)
        self.assertFalse(features.string_regex)

    def test_string_case_insensitive_match(self):
        sql = "SELECT * FROM table1 WHERE LOWER(name) = 'value'"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.string_exact_match)
        self.assertTrue(features.string_case_insensitive_match)
        self.assertFalse(features.string_like_match)
        self.assertFalse(features.string_ilike_match)
        self.assertFalse(features.string_substring)
        self.assertFalse(features.string_regex)

    def test_string_like_match(self):
        sql = "SELECT * FROM table1 WHERE name LIKE '%like%'"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.string_exact_match)
        self.assertFalse(features.string_case_insensitive_match)
        self.assertTrue(features.string_like_match)
        self.assertFalse(features.string_ilike_match)
        self.assertFalse(features.string_substring)
        self.assertFalse(features.string_regex)
        sql = "SELECT * FROM table1 WHERE name ~ 'john';"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.string_exact_match)
        self.assertFalse(features.string_case_insensitive_match)
        self.assertTrue(features.string_like_match)
        self.assertFalse(features.string_ilike_match)
        self.assertFalse(features.string_substring)
        self.assertFalse(features.string_regex)

    def test_string_ilike_match(self):
        sql = "SELECT * FROM table1 WHERE name ILIKE '%ilike%'"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.string_exact_match)
        self.assertFalse(features.string_case_insensitive_match)
        self.assertFalse(features.string_like_match)
        self.assertTrue(features.string_ilike_match)
        self.assertFalse(features.string_substring)
        self.assertFalse(features.string_regex)
        sql = "SELECT * FROM table1 WHERE name ~* 'john';"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.string_exact_match)
        self.assertFalse(features.string_case_insensitive_match)
        self.assertFalse(features.string_like_match)
        self.assertTrue(features.string_ilike_match)
        self.assertFalse(features.string_substring)
        self.assertFalse(features.string_regex)

    def test_string_substring(self):
        sql = "SELECT SUBSTRING(name FROM 1 FOR 3) FROM table1"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        self.assertFalse(features.string_exact_match)
        self.assertFalse(features.string_case_insensitive_match)
        self.assertFalse(features.string_like_match)
        self.assertFalse(features.string_ilike_match)
        self.assertTrue(features.string_substring)
        self.assertFalse(features.string_regex)

    def test_string_regex(self):
        sqls = [
            "SELECT * FROM table1 WHERE REGEXP_LIKE(name, 'regex')",
            "SELECT * FROM table1 WHERE REGEXP_LIKE(name, 'regex', 'i')",
        ]
        for sql in sqls:
            features = get_sql_features(sql, self.md_cols, self.md_tables)
            self.assertFalse(features.string_exact_match)
            self.assertFalse(features.string_case_insensitive_match)
            self.assertFalse(features.string_like_match)
            self.assertFalse(features.string_ilike_match)
            self.assertFalse(features.string_substring)
            self.assertTrue(features.string_regex)

    def test_no_md(self):
        sql = "SELECT column1, column2 FROM table1 JOIN table2 ON table1.id = table2.id"
        features = get_sql_features(sql, None, None)
        self.assertEqual(features.num_columns, 3)
        self.assertEqual(features.num_tables, 2)
        features = get_sql_features(sql, set(), set())
        self.assertEqual(features.num_columns, 3)
        self.assertEqual(features.num_tables, 2)

    def test_sorry(self):
        test_cases = [
            (
                "SELECT 'Alas, without the necessary data, I can''t provide an answer.' AS answer;",
                True,
            ),
            (
                "SELECT 'Answering that is beyond my capacity without the required data.' AS answer;",
                True,
            ),
            (
                "SELECT 'Apologies, I lack the necessary data to provide an answer.' AS answer;",
                True,
            ),
            (
                "SELECT 'I apologize, but I''m not equipped with the data to answer that question.' AS answer;",
                True,
            ),
            (
                "SELECT 'I must apologize, as I do not possess the needed data to answer this.' AS answer;",
                True,
            ),
            (
                "SELECT 'I must express my regret for not having the data to answer that.' AS answer;",
                True,
            ),
            (
                "SELECT 'I regret to inform you that I don''t have the data needed to answer.' AS answer;",
                True,
            ),
            (
                "SELECT 'I''m afraid I cannot answer that due to a lack of necessary data.' AS answer;",
                True,
            ),
            (
                "SELECT 'I''m sorry, but answering that is not feasible without the appropriate data.' AS answer;",
                True,
            ),
            (
                "SELECT 'I''m sorry, but I don''t possess the information needed to answer that question.' AS answer;",
                True,
            ),
            (
                "SELECT 'I''m sorry, but the data needed to respond to this is unavailable to me.' AS answer;",
                True,
            ),
            (
                "SELECT 'I''m unable to provide an answer as I lack the required information.' AS answer;",
                True,
            ),
            (
                "SELECT 'It''s unfortunate, but I don''t have the information required to answer.' AS answer;",
                True,
            ),
            (
                "SELECT 'It's unfortunate, but I don''t have the information required to answer.' AS answer;",
                True,
            ),
            (
                "SELECT 'My apologies, but answering that is not possible without the relevant data.' AS answer;",
                True,
            ),
            (
                "SELECT 'My inability to answer stems from a lack of the necessary data.' AS answer;",
                True,
            ),
            (
                "SELECT 'Regretfully, I don''t hold the data needed to provide a response.' AS answer;",
                True,
            ),
            (
                "SELECT 'Regrettably, I''m without the required data to respond to that.' AS answer;",
                True,
            ),
            (
                "SELECT 'Regrettably, the data required to respond is not within my reach.' AS answer;",
                True,
            ),
            ("SELECT 'Sorry, I do not have the data to answer that.' AS answer;", True),
            (
                "SELECT 'Sorry, the necessary data to respond to your query is not available to me.' AS answer;",
                True,
            ),
            (
                "SELECT 'Unfortunately, I am unable to answer due to insufficient data.' AS answer;",
                True,
            ),
            ("DELETE FROM sessions WHERE session_id = 'ABC123';", False),
            ("SELECT sorry FROM provide WHERE regret = 'data';", False),
            ("UPDATE users SET last_login = NOW() WHERE user_id = 456;", False),
        ]
        for i, (query, expected) in enumerate(test_cases, 1):
            result = is_sorry_sql(query)
            self.assertEqual(result, expected, f"Test case {i} failed: {query}")

    def test_from_compact(self):
        sql = "SELECT column1, column2 FROM table1 JOIN table2 ON table1.id = table2.id"
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        features_compact = features.compact()
        features_from_compact = features.from_compact(features_compact)
        self.assertEqual(features, features_from_compact)

    # Test cases for more complex SQL queries:
    def test_complex_sql_1(self):
        sql = """
WITH stock_stats AS (
    SELECT t.sbTickerSymbol, MIN(d.sbDpLow) AS min_price, MAX(d.sbDpHigh) AS max_price
    FROM sbDailyPrice d
    LEFT JOIN sbTicker t ON d.sbDpTickerId = t.sbTickerId WHERE d.sbDpDate BETWEEN '2023-04-01' AND '2023-04-03' AND t.sbTickerId IS NOT NULL GROUP BY t.sbTickerSymbol) SELECT DISTINCT sbTickerSymbol, max_price - min_price AS price_change FROM stock_stats ORDER BY price_change DESC LIMIT 3
"""
        features = get_sql_features(sql, self.md_cols, self.md_tables)
        features_compact = features.compact()
        expected_compact = "5,2,1,1,0,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
        self.assertEqual(features_compact, expected_compact)
        positive_features = features.positive_features()
        expected_positive = {
            "sql_num_columns": 5,
            "sql_num_tables": 2,
            "sql_table_alias": True,
            "sql_joins": 1,
            "sql_join_left": True,
            "sql_has_null": True,
            "sql_distinct": True,
            "sql_cte": 1,
            "sql_subtraction": True,
            "sql_order_by": True,
            "sql_limit": True,
            "sql_group_by": True,
            "sql_agg_min": True,
            "sql_agg_max": True,
            "sql_date_literal": True,
        }
        self.assertEqual(positive_features, expected_positive)

    def test_complex_sql_2(self):
        sql = """
WITH yearly_max_rpm AS (
    SELECT YEAR, MAX(rpm) AS max_rpm
    FROM train WHERE manufacturer = 'Mfr 1'
    GROUP BY YEAR
) SELECT y.year, COALESCE(ymr.max_rpm, 0) AS max_rpm
FROM (
    SELECT generate_series(MIN(YEAR), MAX(YEAR)) AS YEAR FROM train
    ) y
LEFT JOIN yearly_max_rpm ymr ON y.year = ymr.year ORDER BY y.year NULLS LAST;
"""
        features = get_sql_features(
            sql, {"year", "rpm", "manufacturer", "train_id"}, {"train"}
        )
        features_compact = features.compact()
        expected_compact = "3,1,1,1,0,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0"
        self.assertEqual(features_compact, expected_compact)
        positive_features = features.positive_features()
        expected_positive = {
            "sql_num_columns": 3,
            "sql_num_tables": 1,
            "sql_table_alias": True,
            "sql_joins": 1,
            "sql_join_left": True,
            "sql_cte": 1,
            "sql_order_by": True,
            "sql_group_by": True,
            "sql_agg_min": True,
            "sql_agg_max": True,
            "sql_generate_timeseries": True,
            "sql_string_exact_match": True,
        }
        self.assertEqual(positive_features, expected_positive)


class TestSchemaFeatures(unittest.TestCase):
    def setUp(self):
        self.empty_extra_column_info = {
            "date_type_date_time": set(),
            "date_type_int": set(),
            "date_type_text": set(),
        }

    def test_has_date(self):
        schema = "CREATE TABLE table1 (id INT, birth_date DATE);"
        features, extra_column_info = get_schema_features(schema)
        expected_extra_column_info = {
            "date_type_date_time": {"birth_date"},
            "date_type_int": set(),
            "date_type_text": set(),
        }
        self.assertTrue(features.has_date)
        self.assertDictEqual(extra_column_info, expected_extra_column_info)
        schema = "CREATE TABLE table1 (id INT, birth_date TIMESTAMP);"
        features, extra_column_info = get_schema_features(schema)
        expected_extra_column_info = {
            "date_type_date_time": {"birth_date"},
            "date_type_int": set(),
            "date_type_text": set(),
        }
        self.assertTrue(features.has_date)
        self.assertFalse(features.has_date_text)
        self.assertFalse(features.has_date_int)
        self.assertDictEqual(extra_column_info, expected_extra_column_info)
        schema = "CREATE TABLE table1 (login_date DATE, id BIGINT);"
        features, extra_column_info = get_schema_features(schema)
        expected_extra_column_info = {
            "date_type_date_time": {"login_date"},
            "date_type_int": set(),
            "date_type_text": set(),
        }
        self.assertTrue(features.has_date)
        self.assertFalse(features.has_date_text)
        self.assertFalse(features.has_date_int)
        self.assertDictEqual(extra_column_info, expected_extra_column_info)

    def test_has_date_text(self):
        schema = "CREATE TABLE table1 (user_id INT, birth_date TEXT, birth_year TEXT);"
        features, extra_column_info = get_schema_features(schema)
        expected_extra_column_info = {
            "date_type_date_time": set(),
            "date_type_int": set(),
            "date_type_text": {"birth_year"},
        }
        self.assertFalse(features.has_date)
        self.assertTrue(features.has_date_text)
        self.assertFalse(features.has_date_int)
        self.assertDictEqual(extra_column_info, expected_extra_column_info)
        schema = """CREATE TABLE table1 (
            id INT,
            birth_date VARCHAR(255) -- e.g. '2023-May-01'
        );
        CREATE TABLE table2 (
            id INT,
            event_year VARCHAR, -- e.g. '2023'
            event_month VARCHAR -- e.g. '1', '2', ..., '12'
        );"""
        features, extra_column_info = get_schema_features(schema)
        expected_extra_column_info = {
            "date_type_date_time": set(),
            "date_type_int": set(),
            "date_type_text": {"event_year", "event_month"},
        }
        self.assertFalse(features.has_date)
        self.assertTrue(features.has_date_text)
        self.assertFalse(features.has_date_int)
        self.assertDictEqual(extra_column_info, expected_extra_column_info)

    def test_has_date_int(self):
        schema = "CREATE TABLE reviews (id INT, review_year INT, review_month INT, review TEXT);"
        features, extra_column_info = get_schema_features(schema)
        expected_extra_column_info = {
            "date_type_date_time": set(),
            "date_type_int": {"review_year", "review_month"},
            "date_type_text": set(),
        }
        self.assertFalse(features.has_date)
        self.assertFalse(features.has_date_text)
        self.assertTrue(features.has_date_int)
        self.assertDictEqual(extra_column_info, expected_extra_column_info)
        schema = "CREATE TABLE table1 (id INT, day_of_week BIGINT);"
        features, date_cols = get_schema_features(schema)
        expected_extra_column_info = {
            "date_type_date_time": set(),
            "date_type_int": {"day_of_week"},
            "date_type_text": set(),
        }
        self.assertFalse(features.has_date)
        self.assertFalse(features.has_date_text)
        self.assertTrue(features.has_date_int)
        self.assertDictEqual(date_cols, expected_extra_column_info)

    def test_has_schema(self):
        schema = "CREATE TABLE table1 (id INT);"
        features, extra_column_info = get_schema_features(schema)
        self.assertFalse(features.has_schema)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)
        schema = "CREATE TABLE schema.table1 (id INT);"
        features, extra_column_info = get_schema_features(schema)
        self.assertTrue(features.has_schema)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)
        schema = "CREATE TABLE catalogue.schema.table1 (id INT);"
        features, extra_column_info = get_schema_features(schema)
        self.assertTrue(features.has_schema)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)

    def test_has_catalog(self):
        schema = "CREATE TABLE table1 (id INT);"
        features, extra_column_info = get_schema_features(schema)
        self.assertFalse(features.has_catalog)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)
        schema = "CREATE TABLE schema.table1 (id INT);"
        features, extra_column_info = get_schema_features(schema)
        self.assertFalse(features.has_catalog)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)
        schema = "CREATE TABLE catalogue.schema.table1 (id INT);"
        features, extra_column_info = get_schema_features(schema)
        self.assertTrue(features.has_catalog)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)

    def test_quoted_table_names(self):
        schema = """
        CREATE TABLE "table1" (id INT);
        CREATE TABLE table2 (id INT);
        """
        features, extra_column_info = get_schema_features(schema)
        self.assertTrue(features.quoted_table_names)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)
        schema = """
        CREATE TABLE table1 (id INT);
        CREATE TABLE table2 (id INT);
        """
        features, extra_column_info = get_schema_features(schema)
        self.assertFalse(features.quoted_table_names)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)

    def test_quoted_column_names(self):
        schema = """
        CREATE TABLE table1 ("id" INT, name VARCHAR(255));
        """
        features, extra_column_info = get_schema_features(schema)
        self.assertTrue(features.quoted_column_names)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)
        schema = """
        CREATE TABLE table1 (id INT, name VARCHAR(255));
        """
        features, extra_column_info = get_schema_features(schema)
        self.assertFalse(features.quoted_column_names)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)

    def test_join_hints(self):
        schema = """
        CREATE TABLE table1 (id INT, name VARCHAR(255));
        CREATE TABLE table2 (id INT, description TEXT);

        Here is a list of joinable columns:
        table1.id = table2.id
        """
        features, extra_column_info = get_schema_features(schema)
        self.assertTrue(features.join_hints)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)
        schema = """
        CREATE TABLE table1 (id INT, name VARCHAR(255));
        CREATE TABLE table2 (id INT, description TEXT);
        """
        features, extra_column_info = get_schema_features(schema)
        self.assertFalse(features.join_hints)
        self.assertDictEqual(extra_column_info, self.empty_extra_column_info)

    def test_invalid_ddl(self):
        schema = """
        CREATE TABLE table1 (
            id INT,
            name VARCHAR(255)
        );
        CREATE TABLE table2 (
            id INT, --id col
            description_date DATE -- description col
        );
        """
        features, extra_column_info = get_schema_features(schema)
        expected_extra_column_info = {
            "date_type_date_time": {"description_date"},
            "date_type_int": set(),
            "date_type_text": set(),
        }
        self.assertDictEqual(extra_column_info, expected_extra_column_info)
        self.assertFalse(features.invalid_ddl)
        schema = """
        CREATE TABLE table1 (
            id INT,
            some_value numeric(10, --2),Total value of asset holding
        );
        CREATE TABLE table2 (
            id INT, --id col
            description_date DATE -- description col
        );
        """
        features, extra_column_info = get_schema_features(schema)
        expected_extra_column_info = {
            "date_type_date_time": {"description_date"},
            "date_type_int": set(),
            "date_type_text": set(),
        }
        self.assertTrue(features.invalid_ddl)
        self.assertDictEqual(extra_column_info, expected_extra_column_info)
        print(features.positive_features())
        # note that we can't parse the column descriptions properly hence it's expected to be 0
        expected_positive = {
            "schema_num_tables": 2,
            "schema_num_columns": 4,
            "schema_num_comments": 3,
            "schema_has_date": True,
            "schema_invalid_ddl": True,
        }
        self.assertDictEqual(features.positive_features(), expected_positive)

    def test_from_compact(self):
        schema = """
        CREATE TABLE user (
            id INT,
            name VARCHAR(255), -- trailing comma on the last column is fine
        );
        CREATE TABLE schema.supervisor (
            id SERIAL, -- supervisor's unique id
            status BOOLEAN, age INT,
            name CHARACTER VARYING(255),
        );
        CREATE TABLE cat.schema.table2 (id INT, description TEXT, created_at TIMESTAMP, create_date DATE);
        """
        features, extra_column_info = get_schema_features(schema)
        features_compact = features.compact()
        features_from_compact = features.from_compact(features_compact)
        self.assertEqual(features, features_from_compact)
        expected_column_info = {
            "date_type_date_time": {"created_at", "create_date"},
            "date_type_int": set(),
            "date_type_text": set(),
        }
        self.assertDictEqual(extra_column_info, expected_column_info)

    def test_schema_1(self):
        schema = """
        ```
        CREATE TABLE user (
            id INT,
            name VARCHAR(255),
            birth_year INT, -- trailing comma on the last column is fine
        );
        CREATE TABLE schema.supervisor (
            id SERIAL, -- supervisor's unique id
            status BOOLEAN, age INT,
            name CHARACTER VARYING(255),
        );
        CREATE TABLE cat.schema.table2 (id INT, description TEXT, created_at TIMESTAMP, updated_at DATE);
        ```
        """
        features, extra_column_info = get_schema_features(schema)
        expected_column_info = {
            "date_type_date_time": {"created_at", "updated_at"},
            "date_type_int": {"birth_year"},
            "date_type_text": set(),
        }
        self.assertEqual(extra_column_info, expected_column_info)
        self.assertEqual(features.num_tables, 3)
        self.assertEqual(features.num_columns, 11)
        self.assertEqual(features.num_comments, 2)
        self.assertTrue(features.has_date)
        self.assertTrue(features.has_date_int)
        self.assertFalse(features.has_date_text)
        self.assertTrue(features.has_schema)
        self.assertTrue(features.has_catalog)
        self.assertFalse(features.quoted_table_names)
        self.assertFalse(features.quoted_column_names)
        self.assertFalse(features.invalid_ddl)
        features_compact = features.compact()
        expected_compact = "3,11,2,1,0,1,1,1,0,0,0,0"
        self.assertEqual(features_compact, expected_compact)
        positive_features = features.positive_features()
        expected_positive = {
            "schema_num_tables": 3,
            "schema_num_columns": 11,
            "schema_num_comments": 2,
            "schema_has_date": True,
            "schema_has_date_int": True,
            "schema_has_schema": True,
            "schema_has_catalog": True,
        }
        self.assertEqual(positive_features, expected_positive)

    def test_schema_2(self):
        schema = """CREATE TABLE biz_arts.visits (
  visit_id integer,
  class_id integer,
  class_start_date_time timestamp,
  id integer,
  client_id character varying,
  class_name character varying,
  service_name character varying,
  location_id integer
);
CREATE TABLE biz_arts.studio_sites (
  dashboard boolean,
  utc_start_time_booking_management smallint,
  trello_api_key character varying,
  number_of_forgivables integer,
  studio_label character varying,
  location_id integer,
  performance_report boolean,
  studio_utc_offset smallint,
  report_email character varying,
  booking_management boolean,
  timezone character varying,
  late_cancellation_charge boolean,
  utc_start_time_dashboard smallint,
  mindbody_token character varying,
  last_forgivable timestamp,
  late_cancel_product_id integer,
  country character varying,
  city character varying,
  welcome_text character varying,
  trello_token character varying,
  logo_url text,
  state_province character varying,
  performance_report_email character varying,
  utc_start_time_late_cancel smallint,
  studio_type character varying,
  region character varying,
  late_cancellation_report boolean,
  no_show_product_id integer
);
CREATE TABLE biz_arts.clients (
  location_id integer,
  photo_url text,
  phone_number character varying,
  last_name character varying,
  id integer,
  referred_by character varying,
  client_id character varying,
  creation_date timestamp,
  email character varying,
  notes text,
  first_name character varying,
  last_modified_date_time timestamp,
  forgiveables_remaining integer,
  unique_id integer,
  red_alert text,
  cc_type character varying,
  active boolean,
  visits_at_site integer,
  status character varying
);
CREATE TABLE biz_arts.transactions (
  expiration_date timestamp,
  transaction_code integer,
  location_id integer,
  contract_id integer,
  sale_date timestamp,
  product_id integer,
  unit_price numeric,
  active_date timestamp
);"""
        features, extra_column_info = get_schema_features(schema)
        print(features.positive_features())
        expected_column_info = {
            "date_type_date_time": {
                "active_date",
                "class_start_date_time",
                "creation_date",
                "expiration_date",
                "last_forgivable",
                "last_modified_date_time",
                "sale_date",
            },
            "date_type_int": set(),
            "date_type_text": set(),
        }
        self.assertEqual(extra_column_info, expected_column_info)
        self.assertEqual(features.num_tables, 4)
        self.assertEqual(features.num_columns, 63)
        self.assertEqual(features.num_comments, 0)
        self.assertTrue(features.has_date)
        self.assertFalse(features.has_date_int)
        self.assertFalse(features.has_date_text)
        self.assertTrue(features.has_schema)
        self.assertFalse(features.has_catalog)
        self.assertFalse(features.quoted_table_names)
        self.assertFalse(features.quoted_column_names)
        self.assertFalse(features.invalid_ddl)
        features_compact = features.compact()
        expected_compact = "4,63,0,1,0,0,1,0,0,0,0,0"
        self.assertEqual(features_compact, expected_compact)
        positive_features = features.positive_features()
        expected_positive = {
            "schema_num_tables": 4,
            "schema_num_columns": 63,
            "schema_has_date": True,
            "schema_has_schema": True,
        }
        self.assertEqual(positive_features, expected_positive)


class TestFixComma(unittest.TestCase):
    def test_fix_comma_1(self):
        cols = [
            "  CUSTOMER_EMAIL VARCHAR,",
            "  CUSTOMER_PHONE VARCHAR(200) --Phone number of the customer", # add comma
            "  value numeric(10,2),", # remove trailing comma
        ]
        expected = [
            "  CUSTOMER_EMAIL VARCHAR,",
            "  CUSTOMER_PHONE VARCHAR(200), --Phone number of the customer",
            "  value numeric(10,2)",
        ]
        result = fix_comma(cols)
        print(result)
        self.assertEqual(result, expected)


class TestShuffleTableMetadata(unittest.TestCase):
    def test_shuffle_table_metadata_seed_1(self):
        input_md_str = """CREATE SCHEMA IF NOT EXISTS TEST_DB;
CREATE TABLE TEST_DB.PUBLIC.CUSTOMERS (
  CUSTOMER_EMAIL VARCHAR,
  CUSTOMER_PHONE VARCHAR, --Phone number of the customer
  CUSTOMER_ID NUMERIC, --Unique identifier for each customer
  CUSTOMER_NAME VARCHAR --Name of the customer
);
CREATE TABLE physician (
  name character varying, --name of the physician
  position character varying,
  employeeid integer,
  ssn integer, --social security number of the physician
);
CREATE TABLE patient (
  pcp integer,
  phone character varying,
  ssn integer
);
```
Here is a list of joinable columns:

TEST_DB.PUBLIC.CUSTOMERS.CUSTOMER_ID can be joined with patient.ssn
"""
        expected_md_shuffled = """CREATE SCHEMA IF NOT EXISTS TEST_DB;
CREATE TABLE physician (
  position character varying,
  ssn integer, --social security number of the physician
  name character varying, --name of the physician
  employeeid integer
);
CREATE TABLE TEST_DB.PUBLIC.CUSTOMERS (
  CUSTOMER_PHONE VARCHAR, --Phone number of the customer
  CUSTOMER_NAME VARCHAR, --Name of the customer
  CUSTOMER_EMAIL VARCHAR,
  CUSTOMER_ID NUMERIC --Unique identifier for each customer
);
CREATE TABLE patient (
  pcp integer,
  phone character varying,
  ssn integer
);
Here is a list of joinable columns:
TEST_DB.PUBLIC.CUSTOMERS.CUSTOMER_ID can be joined with patient.ssn"""

        md_shuffled = shuffle_table_metadata(input_md_str, 42)
        self.maxDiff = None
        self.assertEqual(md_shuffled, expected_md_shuffled)
    
    def test_shuffle_table_metadata_seed_2(self):
        input_md_str = """CREATE TABLE branch_info (
  branch_open_date date, --Date branch opened
  value numeric(10,2),
  manager_name varchar(100) --Name of the branch manager
);
CREATE TABLE employee (
  employee_id integer,
  ytd_return numeric(5,2)
);"""
        expected_md_shuffled = """CREATE TABLE employee (
  employee_id integer,
  ytd_return numeric(5,2)
);
CREATE TABLE branch_info (
  manager_name varchar(100), --Name of the branch manager
  value numeric(10,2),
  branch_open_date date --Date branch opened
);"""
        md_shuffled = shuffle_table_metadata(input_md_str, 0)
        print(md_shuffled)
        self.assertEqual(md_shuffled, expected_md_shuffled)

    def test_shuffle_table_metadata_seed_3(self):
        input_md_str = "CREATE TABLE sales (\n  pl_sales_rep_id integer,\n  pl_site_id integer,\n  pl_id integer,\n  pl_sale_date timestamp,\n  pl_product_id integer,\n  pl_contract_id integer,\n  pl_active_date timestamp,\n  pl_expiration_date timestamp,\n  pl_sale_id integer,\n  pl_purchased_item_id integer,\n  pl_remaining_passes integer,\n  pl_unit_price numeric,\n  pl_total_amount numeric,\n  pl_payment_ref_id integer,\n  pl_quantity integer,\n  pl_sale_detail_id integer,\n  pl_actual_sale_id integer,\n  pl_actual_purchased_item_id integer,\n  pl_recipient_client_id integer,\n  pl_client_id character varying,\n  pl_dt_created date,\n  pl_dt_modified date\n);\nCREATE TABLE visits (\n  pl_class_id integer,\n  pl_site_id integer,\n  pl_class_start_date_time timestamp,\n  pl_visit_id integer,\n  pl_id integer,\n  pl_client_id character varying,\n  pl_class_name character varying,\n  pl_service_name character varying,\n  pl_dt_deleted date,\n  pl_dt_updated date\n);"
        expected_md_shuffled = """CREATE TABLE sales (
  pl_dt_created date,
  pl_remaining_passes integer,
  pl_quantity integer,
  pl_payment_ref_id integer,
  pl_site_id integer,
  pl_dt_modified date,
  pl_unit_price numeric,
  pl_client_id character varying,
  pl_sale_id integer,
  pl_active_date timestamp,
  pl_actual_sale_id integer,
  pl_product_id integer,
  pl_id integer,
  pl_contract_id integer,
  pl_actual_purchased_item_id integer,
  pl_purchased_item_id integer,
  pl_expiration_date timestamp,
  pl_recipient_client_id integer,
  pl_sale_date timestamp,
  pl_sales_rep_id integer,
  pl_sale_detail_id integer,
  pl_total_amount numeric
);
CREATE TABLE visits (
  pl_class_start_date_time timestamp,
  pl_id integer,
  pl_visit_id integer,
  pl_class_name character varying,
  pl_service_name character varying,
  pl_class_id integer,
  pl_dt_deleted date,
  pl_dt_updated date,
  pl_client_id character varying,
  pl_site_id integer
);"""
        md_shuffled = shuffle_table_metadata(input_md_str, 0)
        print(md_shuffled)
        self.assertEqual(md_shuffled, expected_md_shuffled)

    def test_shuffle_table_metadata_seed_4(self):
        input_md_str = "CREATE TABLE fin_acct_perf (\n  account_id integer, --Primary key, foreign key to fin_cust_acct table\n  snapshot_date date, --Date of performance snapshot\n  ytd_return numeric(5,2) --Year-to-date return percentage\n);"
        expected_md_shuffled = """CREATE TABLE fin_acct_perf (
  ytd_return numeric(5,2), --Year-to-date return percentage
  snapshot_date date, --Date of performance snapshot
  account_id integer --Primary key, foreign key to fin_cust_acct table
);"""
        md_shuffled = shuffle_table_metadata(input_md_str, 0)
        print(md_shuffled)
        self.assertEqual(md_shuffled, expected_md_shuffled)

class TestFunctions(unittest.TestCase):
    def test_is_date_or_time_str(self):
        s1 = "2022-03-19"
        s2 = "2022-03-19 12:34:56"
        s3 = "12:34:56"
        self.assertTrue(is_date_or_time_str(s1))
        self.assertTrue(is_date_or_time_str(s2))
        self.assertTrue(is_date_or_time_str(s3))

    def test_add_space_padding(self):
        sql1 = "SELECT SUM(a.amount), a.tx FROM amounts a WHERE EXTRACT('month' FROM a.tx) = 1 GROUP BY a.tx"
        expected = "SELECT SUM ( a.amount ) , a.tx FROM amounts a WHERE EXTRACT ( 'month' FROM a.tx ) = 1 GROUP BY a.tx"
        result = add_space_padding(sql1)
        self.assertEqual(result, expected)

        sql2 = "SELECT v.venuename, e.eventid, row_number () OVER (PARTITION BY v.venuename ORDER BY COUNT (s.salesid) DESC) AS rank FROM venue v JOIN event e ON v.venueid = e.venueid JOIN sales s ON e.eventid = s.eventid AND e.dateid = s.dateid GROUP BY v.venuename, e.eventid;"
        expected = "SELECT v.venuename, e.eventid, row_number ( ) OVER ( PARTITION BY v.venuename ORDER BY COUNT ( s.salesid ) DESC ) AS rank FROM venue v JOIN event e ON v.venueid = e.venueid JOIN sales s ON e.eventid = s.eventid AND e.dateid = s.dateid GROUP BY v.venuename, e.eventid;"
        result = add_space_padding(sql2)
        print(result)
        self.assertEqual(result, expected)


class TestReplaceAlias(unittest.TestCase):
    def test_replace_alias_change_existing_alias(self):
        # should replace users a with users u and info b with info i
        sql = "SELECT a.name, b.age FROM users a JOIN info b ON a.id = b.id"
        new_alias_map = {"users": "u", "info": "i"}
        expected = "SELECT u.name, i.age FROM users AS u JOIN info AS i ON u.id = i.id"
        self.assertEqual(replace_alias(sql, new_alias_map), expected)

    def test_replace_alias_no_alias(self):
        # should add alias to tables if not present
        sql = "SELECT name, age FROM users JOIN info ON users.id = info.id"
        new_alias_map = {"users": "u", "info": "i"}
        expected = "SELECT name, age FROM users AS u JOIN info AS i ON u.id = i.id"
        self.assertEqual(replace_alias(sql, new_alias_map), expected)

    def test_replace_alias_no_table_alias_have_column_table(self):
        # should replace alias in columns using new_alias_map and add alias to tables
        sql = "SELECT users.name, info.age FROM users JOIN info ON users.id = info.id"
        new_alias_map = {"users": "u", "info": "i"}
        expected = "SELECT u.name, i.age FROM users AS u JOIN info AS i ON u.id = i.id"
        self.assertEqual(replace_alias(sql, new_alias_map), expected)

    def test_replace_alias_no_change(self):
        sql = "SELECT a.name, b.age FROM users AS a JOIN info AS b ON a.id = b.id"
        new_alias_map = {"users": "a", "logs": "l"}
        expected = sql
        self.assertEqual(replace_alias(sql, new_alias_map), expected)

    # def test_replace_with_subquery(self):
    #     # failing case. uncomment if this is fixed
    #     sql = "WITH user_count AS (SELECT COUNT(u.id) AS cnt FROM users u) SELECT u.cnt FROM user_count u"
    #     new_alias_map = {"users": "us"}
    #     expected = "WITH user_count AS (SELECT COUNT(us.id) AS cnt FROM users AS us) SELECT u.cnt FROM user_count AS u"
    #     self.assertEqual(replace_alias(sql, new_alias_map), expected)

    def test_sql_1(self):
        sql = """WITH player_games AS (
  SELECT gm.gm_game_id AS game_id, pl.pl_player_id AS player_id, pl.pl_team_id AS team_id
  FROM games gm
  JOIN players pl ON gm.gm_home_team_id = pl.pl_team_id OR gm.gm_away_team_id = pl.pl_team_id
),
unique_games AS (
  SELECT DISTINCT gm_game_id FROM games
)
SELECT CAST(COUNT(DISTINCT pg.game_id) AS FLOAT) / NULLIF(COUNT(DISTINCT ug.gm_game_id), 0) AS fraction
FROM player_games pg
RIGHT JOIN unique_games ug ON pg.game_id = ug.gm_game_id;"""
        new_alias_map = {
            "teams": "t",
            "players": "p",
            "games": "g",
            "game_events": "ge",
            "player_stats": "ps",
        }
        expected = """WITH player_games AS (SELECT g.gm_game_id AS game_id, p.pl_player_id AS player_id, p.pl_team_id AS team_id FROM games AS g JOIN players AS p ON g.gm_home_team_id = p.pl_team_id OR g.gm_away_team_id = p.pl_team_id), unique_games AS (SELECT DISTINCT gm_game_id FROM games AS g) SELECT CAST(COUNT(DISTINCT pg.game_id) AS DOUBLE PRECISION) / NULLIF(COUNT(DISTINCT ug.gm_game_id), 0) AS fraction FROM player_games AS pg RIGHT JOIN unique_games AS ug ON pg.game_id = ug.gm_game_id"""
        result = replace_alias(sql, new_alias_map)
        print(result)
        self.assertEqual(result, expected)

    def test_sql_2(self):
        sql = "SELECT train.year, manufacturer, AVG(train.capacity) AS average_capacity FROM train WHERE train.manufacturer ILIKE '%Mfr 1%' GROUP BY train.year, train.manufacturer ORDER BY train.year;"
        new_alias_map = {"train": "t"}
        expected = "SELECT t.year, manufacturer, AVG(t.capacity) AS average_capacity FROM train AS t WHERE t.manufacturer ILIKE '%Mfr 1%' GROUP BY t.year, t.manufacturer ORDER BY t.year"
        result = replace_alias(sql, new_alias_map)
        print(result)
        self.assertEqual(result, expected)

    def test_sql_3(self):
        sql = "SELECT CAST((SELECT COUNT(aw.artwork_id) FROM artwork aw WHERE aw.year_created = 1888 AND aw.description IS NULL) AS FLOAT) / NULLIF((SELECT COUNT(at.artist_id) FROM artists AT WHERE at.nationality ilike '%French%'), 0) AS ratio;"
        new_alias_map = {'exhibit_artworks': 'ea', 'exhibitions': 'e', 'collaborations': 'c', 'artwork': 'a', 'artists': 'ar'}
        expected = "SELECT CAST((SELECT COUNT(a.artwork_id) FROM artwork AS a WHERE a.year_created = 1888 AND a.description IS NULL) AS DOUBLE PRECISION) / NULLIF((SELECT COUNT(ar.artist_id) FROM artists AS ar WHERE ar.nationality ILIKE '%French%'), 0) AS ratio"
        result = replace_alias(sql, new_alias_map)
        print(result)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
