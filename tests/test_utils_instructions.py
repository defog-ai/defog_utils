import unittest
from defog_utils.defog_utils.constants import idk_strings
from defog_utils.defog_utils.utils_instructions import (
    InstructionFeatures,
    get_instruction_features,
)


class TestGetInstructionFeatures(unittest.TestCase):
    def test_get_instruction_features_alias1(self):
        # these are samples from our actual generated data
        instructions_list = [
            '"Use abbreviated table aliases."',
            "Use short table aliases.",
            '"Use max 2-letter table name aliases. Use numbers in GROUP BY."',
            "Use ILIKE to match queries on text/varchar columns. Prefix columns with a short table alias.",
            "Translate the following requests into a SQL query. Always specify the column alias or name in the query.",
            "Always include the fully qualified table name for any query. Never cast a table name as a variable or alias it.",
            "Always specify the column alias or name in the query.",
        ]
        for instructions in instructions_list:
            features = get_instruction_features(instructions)
            self.assertIsInstance(features, InstructionFeatures)
            self.assertEqual(features.add_alias_prefix, True)
            self.assertEqual(features.idk, False)
            positive_features = features.positive_features()
            self.assertDictEqual(
                positive_features, {"instruction_add_alias_prefix": True}
            )

    def test_get_instruction_features_idk(self):
        instructions_list = idk_strings + [
            "TQA is the total quantity of assets, and one needs to sum the quantities from acct_asset. TVA is the total value of assets, and one needs to sum (value * quantity) from acct_asset. When asked for holdings, always only use rows with the latest date from acct_asset. If a date is specified, use the latest available date if the date specified does not have an entry.\nWhen the data fails to answer the question, please use the query 'SELECT 'I''m sorry, but I don''t possess the information needed to answer that question.' AS answer;'.",
            "If year is not mentioned, assume the current year. Keep all months even when the count is 0. Always refer to the `transaction_date` column for date aggregates. Return date-related columns as date types\nWhen the existing data isn't enough to provide an answer, give 'SELECT 'I''m sorry, but answering that is not feasible without the appropriate data.' AS answer;'.",
            "Only consider settlements associate with customer policies start date before January 1st, 2022 and end date after December 31st, 2023.\nIf answering the question is not feasible using the available data, then please give the query 'SELECT 'I''m sorry, but the data needed to respond to this is unavailable to me.' AS answer;'.",
            "Use TO_TIMESTAMP to convert unix timestamps to dates. Use ILIKE to match queries on text/varchar columns.\nReturn the query 'SELECT 'I''m sorry, but I don''t possess the information needed to answer that question.' AS answer;' if the current data doesn't permit an answer to the question.",
            "Use ILIKE to match queries on text/varchar columns. Only consider games where the team name contains 'Yankees'. Use the current date minus 3 months as the start date for counting the games.\nReturn the query 'SELECT 'I''m sorry, but the data needed to respond to this is unavailable to me.' AS answer;' if the current data doesn't permit an answer to the question.",
            "If there are no counts for a given month, show 0. Always assume per month metrics with a `year_month` column, formatted as a date.\nReturn the query 'SELECT 'I''m sorry, but answering that is not feasible without the appropriate data.' AS answer;' if the current data doesn't permit an answer to the question.",
        ]
        for instructions in instructions_list:
            features = get_instruction_features(instructions)
            self.assertIsInstance(features, InstructionFeatures)
            self.assertEqual(features.add_alias_prefix, False)
            self.assertEqual(features.idk, True)
            positive_features = features.positive_features()
            self.assertDictEqual(positive_features, {"instruction_idk": True})

    def test_get_instruction_features_alias_idk(self):
        instructions_list = [
            "Use abbreviated table aliases.\nIf the question cannot be answered by the available data, return the query 'SELECT 'Alas, without the necessary data, I can't provide an answer.' AS answer;'",
            "Prefix columns with a short table alias.\nWhen the data fails to answer the question, please use the query 'SELECT 'Answering that is beyond my capacity without the required data.' AS answer;'.",
            "Prefix columns with a short table alias.\nIf the data at hand is inadequate for the question, respond 'SELECT 'I''m unable to provide an answer as I lack the required information.' AS answer;'.",
            "Prefix columns with a short table alias.\nWhen the data fails to answer the question, please use the query 'SELECT 'I must express my regret for not having the data to answer that.' AS answer;'.",
        ]
        for instructions in instructions_list:
            features = get_instruction_features(instructions)
            print(instructions)
            print(features)
            self.assertIsInstance(features, InstructionFeatures)
            self.assertEqual(features.add_alias_prefix, True)
            self.assertEqual(features.idk, True)
            positive_features = features.positive_features()
            self.assertDictEqual(
                positive_features,
                {"instruction_add_alias_prefix": True, "instruction_idk": True},
            )

    def test_get_instruction_features_none(self):
        instructions = "This is a normal instruction."
        features = get_instruction_features(instructions)
        self.assertIsInstance(features, InstructionFeatures)
        self.assertEqual(features.add_alias_prefix, False)
        self.assertEqual(features.idk, False)
        positive_features = features.positive_features()
        self.assertDictEqual(positive_features, {})

    def test_compact(self):
        features = InstructionFeatures(add_alias_prefix=True, idk=True)
        features_compact = features.compact()
        self.assertEqual(features_compact, "1,1")
        features_from_compact = InstructionFeatures().from_compact(features_compact)
        self.assertEqual(features, features_from_compact)
