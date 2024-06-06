import re

# note that some of these are substrings of the original list in gen_cat_abc_train.ipynb,
# but without some leading words due to inconsistent quoting in our generated data
# eg sometimes we have `I''m` while other times we have `I'm`.
idk_strings = [
    "Sorry, I do not have the data to answer that",
    "Apologies, I lack the necessary data to provide an answer",
    "Regrettably, I''m without the required data to respond to that",
    "sorry, but I don''t possess the information needed to answer that question",
    "Unfortunately, I am unable to answer due to insufficient data",
    "I regret to inform you that I don''t have the data needed to answer",
    "My apologies, but answering that is not possible without the relevant data",
    "Sorry, the necessary data to respond to your query is not available to me",
    "unable to provide an answer as I lack the required information",
    "Regretfully, I don''t hold the data needed to provide a response",
    "I must apologize, as I do not possess the needed data to answer this",
    "Alas, without the necessary data, I can''t provide an answer",
    "unfortunate, but I don''t have the information required to answer",
    "sorry, but the data needed to respond to this is unavailable to me",
    "afraid I cannot answer that due to a lack of necessary data",
    "I must express my regret for not having the data to answer that",
    "Answering that is beyond my capacity without the required data",
    "My inability to answer stems from a lack of the necessary data",
    "I apologize, but I''m not equipped with the data to answer that question",
    "Regrettably, the data required to respond is not within my reach",
    "sorry, but answering that is not feasible without the appropriate data",
    "Alas, without the necessary data, I can't provide an answer",
]
# a regex pattern that matches if a string contains any of the idk_strings as a substring
idk_re_pattern = re.compile(
    r"|".join(re.escape(idk_string) for idk_string in idk_strings), re.IGNORECASE
)

cot_table_alias_instructions = [
    "List the table aliases for each table as comments, starting with the most relevant tables to the question.",
    "Generate pairs of tables and their aliases as comments, beginning with the most relevant tables to the question.",
    "Provide the aliases for all tables in the DDL as comments, starting from the most relevant tables to the question.",
    "Start by listing the aliases for each table as comments, beginning with the most relevant tables to the question.",
]
