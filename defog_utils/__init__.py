

import json
import pickle
import spacy
from sentence_transformers import SentenceTransformer
import time
import pandas as pd
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import tiktoken
import yaml
import re
from sqlalchemy import create_engine
from sql_formatter.core import format_sql
import os
import constants as c
import csv
from func_timeout import func_timeout


class WrongSetupError(Exception):
    pass


# load model for embedding question
os.environ["TOKENIZERS_PARALLELISM"] = "true"
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

def get_glossary_metadatacsv(metadata_file):
    """
    Get glossary and the metadata as csv from metadata json file
    """
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        table_metadata = metadata["table_metadata"]
        glossary = metadata["glossary"]

    table_metadata_string = []
    for table in table_metadata:
        table_items = table_metadata[table]
        for idx in range(len(table_items)):
            table_items[idx]["table_name"] = table
        table_metadata_string += table_items

    table_metadata_string = (
        pd.DataFrame(table_metadata_string)[
            ["table_name", "column_name", "data_type", "column_description"]
        ]
        .rename(columns={"data_type": "column_data_type"})
        .to_csv(index=False)
    )
    return glossary, table_metadata_string


def get_glossary_metadatasql(metadata_file, exclude_column_description=False):
    """
    Get glossary and metadata in sql format from metadata json file
    """
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        table_metadata = metadata["table_metadata"]
        glossary = metadata["glossary"]

    table_metadata_string = ""
    for table in table_metadata:
        sql_text = ""
        for item in table_metadata[table]:
            if item["column_name"] != "":
                if exclude_column_description:
                    sql_text += f"\n  {item['column_name']} {item['data_type']},"
                else:
                    sql_text += f"\n  {item['column_name']} {item['data_type']}, --{item['column_description']}"
        sql_text = sql_text + "\n"
        table_metadata_string += f"CREATE TABLE {table} ({sql_text})"
        table_metadata_string += "\n\n-----------\n\n"

    return glossary, table_metadata_string


def get_glossary_metadata_emb(
    question: str,
    metadata_file: str,
    metadata_format: str,
    verbose: bool,
    exclude_column_descriptions: bool = False,
    embedding_file_path: str = "metadata/embeddings.pkl",
    ner_metadata_file_path: str = "metadata/ner_metadata.pkl",
) -> tuple:
    """
    Thin wrapper over get_md_emb to return glossary in addition to metadata string
    """
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        glossary = metadata["glossary"]
    db_name = metadata_file.split("/")[-1].split(".")[0]
    emb, csv_descriptions = load_all_emb(embedding_file_path)
    columns_ner, columns_join = load_ner_md(ner_metadata_file_path)
    table_metadata_string = get_md_emb(
        question,
        emb[db_name],
        csv_descriptions[db_name],
        columns_ner[db_name],
        columns_join[db_name],
        metadata_format=metadata_format,
        verbose=verbose,
        exclude_column_descriptions=exclude_column_descriptions,
    )

    return glossary, table_metadata_string


def load_all_emb(embedding_file_path="metadata/embeddings.pkl") -> Tuple[Dict[str, torch.tensor], List[str]]:
    """
    Load all embeddings from pickle file.
    Note that you would need to run the script to generate the
    embeddings for each db first:
    `python auto_eval/gen_embeddings.py`
    """
    try:
        with open(embedding_file_path, "rb") as f:
            all_emb, col_descriptions = pickle.load(f)
            return all_emb, col_descriptions
    except FileNotFoundError:
        print(
            "Embeddings not found. Please run `python metadata/gen_embeddings.py` to generate the embeddings first."
        )
        exit(1)


def load_ner_md(embedding_file_path="metadata/ner_metadata.pkl") -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    Load all NER and join metadata from pickle file.
    Note that you would need to run the script to save the metadata to the pickle file first:
    `python auto_eval/gen_embeddings.py`
    """
    try:
        with open(embedding_file_path, "rb") as f:
            column_ner, column_join = pickle.load(f)
            return column_ner, column_join
    except FileNotFoundError:
        print(
            "NER and join metadata not found. Please run `python metadata/gen_embeddings.py` to generate the metadata first."
        )
        exit(1)


def knn(
    query: str,
    all_emb: torch.tensor,
    k: int,
    threshold: float,
    device: str = "cuda",
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Get top most similar columns' embeddings to query using cosine similarity.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    query_emb = encoder.encode(query, convert_to_tensor=True, device=device)
    similarity_scores = F.cosine_similarity(query_emb, all_emb.to(device))
    top_results = torch.nonzero(similarity_scores > threshold).squeeze()
    # if top_results is empty, return empty tensors
    if top_results.numel() == 0:
        return torch.tensor([]), torch.tensor([])
    else:
        try:
            top_k_scores, top_k_indices = torch.topk(
                similarity_scores[top_results], k=min(k, top_results.numel())
            )
            return top_k_scores, top_results[top_k_indices]
        except Exception as e:
            print(f"Error in knn\ntop_results:\n{top_results}\n{e}")
            return torch.tensor([]), torch.tensor([])


def get_entity_types(sentence, verbose: bool = False):
    """
    Get entity types from sentence using spaCy.
    """
    doc = nlp(sentence)
    named_entities = set()
    for ent in doc.ents:
        if verbose:
            print(f"ent {ent}, {ent.label_}")
        named_entities.add(ent.label_)

    return named_entities


def format_topk_csv(
    topk_table_columns: Dict[str, List[Tuple[str, str, str]]],
    exclude_column_descriptions: bool = False,
) -> str:
    md_list = []
    for table_name in topk_table_columns:
        for column_tuple in topk_table_columns[table_name]:
            if exclude_column_descriptions:
                md_list.append(f"{table_name}.{column_tuple[0]},{column_tuple[1]}")
            else:
                md_list.append(
                    f"{table_name}.{column_tuple[0]},{column_tuple[1]},{column_tuple[2]}"
                )
    md_str = "\n".join(md_list) + "\n"
    return md_str


def format_topk_sql(
    topk_table_columns: Dict[str, List[Tuple[str, str, str]]],
    exclude_column_descriptions: bool = False,
) -> str:
    md_str = ""
    for table_name in topk_table_columns:
        columns_str = ""
        for column_tuple in topk_table_columns[table_name]:
            if exclude_column_descriptions:
                columns_str += f"\n  {column_tuple[0]} {column_tuple[1]},"
            else:
                columns_str += (
                    f"\n  {column_tuple[0]} {column_tuple[1]}, --{column_tuple[2]}"
                )
        md_str += f"CREATE TABLE {table_name} ({columns_str}\n);\n"
    return md_str


def get_md_emb(
    question: str,
    column_emb: torch.tensor, 
    column_info_csv: List[str],
    column_ner: Dict[str, List[str]],
    column_join: Dict[Tuple[str, str], List[Tuple[str, str]]],
    metadata_format: str = "csv_emb",
    k: int = 20,
    threshold: float = 0.0,
    verbose: bool = False,
    exclude_column_descriptions: bool = False,
) -> str:
    """
    Given question, generated metadata csv string with top k columns and tables
    that are most similar to the question. `column_emb`, `column_info_csv`, `column_ner`,
    `column_join` are all specific to the db_name. `column_info_csv` is a list of csv strings
    with 1 row per column info, where each row is in the format:
    `table_name.column_name,column_type,column_description`.
    Steps are:
    1. Get top k columns from question to `column_emb` using `knn` and add
      the corresponding column info to topk_table_columns.
    2. Get entity types from question. If entity type is in `column_ner`, add the
      corresponding list of column info to topk_table_columns.
    3. Generate the metadata string using the column info so far. Can format as CSV/SQL.
    4. Get joinable columns between tables in topk_table_columns and add to final metadata string.
    """
    t = time.time()
    # 1) get top k columns
    top_k_scores, top_k_indices = knn(question, column_emb, k, threshold)
    topk_table_columns = {}
    table_column_names = set()
    for score, index in zip(top_k_scores, top_k_indices):
        table_name, column_info = column_info_csv[index].split(".", 1)
        column_tuple = tuple(column_info.split(",", 2))
        if table_name not in topk_table_columns:
            topk_table_columns[table_name] = []
        topk_table_columns[table_name].append(column_tuple)
        table_column_names.add(f"{table_name}.{column_tuple[0]}")
    # 2) get entity types from question + add corresponding columns
    entity_types = get_entity_types(question)
    for entity_type in entity_types:
        if entity_type in column_ner:
            for column_info in column_ner[entity_type]:
                table_column_name, column_type, column_description = column_info.split(",", 2)
                table_name, column_name = table_column_name.split(".", 1)
                if table_name not in topk_table_columns:
                    topk_table_columns[table_name] = []
                column_tuple = (column_name, column_type, column_description)
                if column_tuple not in topk_table_columns[table_name]:
                    topk_table_columns[table_name].append(column_tuple)
    topk_tables = sorted(list(topk_table_columns.keys()))
    # 3) get table pairs that can be joined
    # create dict of table_column_name -> column_tuple for lookups
    column_name_to_tuple = {}
    ncols = len(column_info_csv)
    for i in range(ncols):
        table_column_name, column_type, column_description = column_info_csv[i].split(",", 2)
        table_name, column_name = table_column_name.split(".", 1)
        column_tuple = (column_name, column_type, column_description)
        column_name_to_tuple[table_column_name] = column_tuple
    # go through list of top k tables and see if pairs can be joined
    join_list = []
    for i in range(len(topk_tables)):
        for j in range(i + 1, len(topk_tables)):
            table1, table2 = topk_tables[i], topk_tables[j]
            assert table1 <= table2
            if (table1, table2) in column_join:
                for table_col_1, table_col_2 in column_join[(table1, table2)]:
                    # add to topk_table_columns
                    if table_col_1 not in table_column_names:
                        column_tuple = column_name_to_tuple[table_col_1]
                        topk_table_columns[table1].append(column_tuple)
                        table_column_names.add(table_col_1)
                    if table_col_2 not in table_column_names:
                        column_tuple = column_name_to_tuple[table_col_2]
                        topk_table_columns[table2].append(column_tuple)
                        table_column_names.add(table_col_2)
                    # add to join_list
                    join_str = f"{table_col_1} can be joined with {table_col_2}"
                    if join_str not in join_list:
                        join_list.append(join_str)
    # 4) format metadata string
    if metadata_format == "csv_emb":
        md_str = format_topk_csv(topk_table_columns, exclude_column_descriptions)
    elif metadata_format == "sql_emb":
        md_str = format_topk_sql(topk_table_columns, exclude_column_descriptions)
    else:
        raise ValueError(f"metadata_format {metadata_format} not supported")
    if join_list:
        md_str += "```\nHere is a list of joinable columns:\n```\n"
        md_str += "\n".join(join_list)
    # if verbose:
    #     duration = time.time() - t
    #     print_str = f"\n\tget_md_str took {duration:.2f} seconds\n" + \
    #         f"\tquestion:\n\t{question}\n" + \
    #         f"\tmd_str:\n\t" + md_str.replace('\n','\n\t')
    #     print(print_str)
    return md_str


def load_yaml(yaml_file):
    with open(yaml_file) as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


class YAMLError(Exception):
    pass


def get_completion_dict(yaml_str):
    try:
        results = yaml.safe_load(yaml_str)
    except Exception as e:
        print(f"Error unmarshaling YAML: {yaml_str}\nError: {e}")
        try:
            print("Correcting YAML indentation...")
            yaml_str = correct_yaml_indentation(yaml_str)
            results = yaml.safe_load(yaml_str)
        except Exception as e:
            print("Unable to correct YAML indentation. Raise exception")
            raise YAMLError(f"Error correcting YAML: {yaml_str}\n Error: {e}")

    if results["sql"].startswith("sql") or results["sql"].startswith("SQL"):
        results["sql"] = results["sql"][3:]
    if results["sql"] != "":
        formatted_sql = format_sql(results["sql"])
        results["sql"] = formatted_sql
    return results


def correct_yaml_indentation(yaml_string):
    lines = yaml_string.split("\n")
    correct_lines = []

    for line in lines:
        if line.endswith("|"):
            correct_lines.append(line)
            continue
        else:
            line = line.strip()
            line = "    " + line
            correct_lines.append(line)

    return "\n".join(correct_lines)


def filter_qns(gen_qn_list, train_qn_list, test_qn_list, batch_new_qns):
    gen_qn_list = [
        q for q in gen_qn_list if q not in train_qn_list
    ]  # Remove duplicate questions
    gen_qn_list = [
        q for q in gen_qn_list if q not in test_qn_list
    ]  # Remove questions in test set
    gen_qn_list = [
        q for q in gen_qn_list if q not in batch_new_qns
    ]  # Remove questions in batch_new_qns
    gen_qn_list = [
        q for q in gen_qn_list if "specific" not in q.lower() and "particular" not in q.lower() and "certain" not in q.lower()
    ]  # Remove questions with 'specific' in them
    return gen_qn_list


def str_to_list(qn_str):
    """Converts a string of questions to a list of questions without numbers"""

    # Remove any line that does not start with a number
    cleaned = []
    for line in qn_str.split('\n'):
        if re.match(r'^\d+', line):
            cleaned.append(line)

    # Remove number at start of in list
    qn_list = [re.sub(r"^\d+\s*\.*\)*\s*", "", q) for q in cleaned]
    return qn_list


def list_to_str(qn_list):
    """Converts a list of questions to a string of questions with numbers"""
    qn_str = "\n".join(["{}.\t{}".format(i + 1, q) for i, q in enumerate(qn_list)])
    return qn_str


def results_to_dict(
    model,
    db_name,
    db_type,
    user_question,
    valid_point,
    error_msg,
    sql,
    reason,
    difficulty,
    table_metadata_string
):
    resp_save = {}
    resp_save["model"] = model
    resp_save["db_name"] = db_name
    resp_save["db_type"] = db_type
    resp_save["user_question"] = user_question
    resp_save["validity"] = valid_point
    resp_save["error_msg"] = error_msg
    resp_save["sql"] = sql
    resp_save["reason"] = reason
    resp_save["difficulty"] = difficulty
    resp_save["table_metadata_string"] = table_metadata_string

    return resp_save


def might_be_subquery(sql_query):
    # Remove strings and comments that might contain the word "select"
    sql_query = re.sub(r"--.*", "", sql_query)  # Remove comments
    sql_query = re.sub(r"\'[^\']*\'", "", sql_query)  # Remove strings
    sql_query = sql_query.lower()  # Convert to lowercase for consistent matching

    select_count = sql_query.count("select")
    from_count = sql_query.count("from")

    # If we have more than one 'select' and at least one 'from', it might be a subquery
    if select_count > 1 and from_count >= 1:
        return True
    else:
        return False


def evaluate_sql_difficulty(query):
    sql_components_1 = [
        "WHERE",
        "GROUP BY",
        "ORDER BY",
        "LIMIT",
        "JOIN",
        "LIKE",
        "HAVING",
    ]
    sql_components_2 = ["EXCEPT", "UNION", "INTERSECT"]
    others_regex_patterns = [
        r"COUNT\(.*\)|SUM\(.*\)|AVG\(.*\)|MIN\(.*\)|MAX\(.*\)",  # Aggregate functions
        r"(?:SELECT\s+.*?)(?:(?:,\s+.*?)+)",  # Multiple select columns
        r"(?:WHERE\s+.*?)(?:(?:AND\s+.*?)+)",  # Multiple where conditions
        r"(?:GROUP BY\s+.*?)(?:(?:,\s+.*?)+)",  # Multiple group by clauses
    ]

    count_sql_1 = sum(query.upper().count(comp) for comp in sql_components_1)
    count_sql_2 = sum(query.upper().count(comp) for comp in sql_components_2)
    if might_be_subquery(query):
        count_sql_2 += 1
    count_others = sum(
        1 for pattern in others_regex_patterns if re.search(pattern, query.upper())
    )

    if count_sql_1 <= 1 and count_others == 0 and count_sql_2 == 0:
        return "Easy"
    elif (
        count_others <= 2
        and count_sql_1 <= 1
        and count_sql_2 == 0
        or count_sql_1 == 2
        and count_others < 2
        and count_sql_2 == 0
    ):
        return "Med"
    elif (
        count_others > 2
        and count_sql_1 <= 2
        and count_sql_2 == 0
        or 2 < count_sql_1 <= 3
        and count_others <= 2
        and count_sql_2 == 0
        or count_sql_1 <= 1
        and count_others == 0
        and count_sql_2 == 1
    ):
        return "Hard"
    else:
        return "XHard"


def get_prompt_comp_pair(
    db_type, glossary, user_question, table_metadata_string, sql, reason, version
):
    reason = reason.strip()
    sql = sql.strip().replace("\n", "\n  ")

    prompt_yaml = load_yaml(
        "prompt_templates/" + version + "/prompt_trainpcpair_" + version + ".yaml"
    )
    prompt = prompt_yaml["prompt"].format(
        db_type=db_type,
        glossary=glossary,
        user_question=user_question,
        table_metadata_string=table_metadata_string,
    )
    completion = prompt_yaml["completion"].format(reason=reason, sql=sql)

    prompt_comp_pair = {}
    if glossary == "":  # If glossary is empty, remove glossary from prompt
        prompt = prompt.replace(
            "Use the following instructions if and only if they are relevant to the question:\n``\n",
            "",
        )
    prompt_comp_pair["prompt"] = prompt
    prompt_comp_pair["completion"] = completion
    return prompt_comp_pair


def count_tokens(model, string):
    encoding = tiktoken.encoding_for_model(model)
    n_tokens = len(encoding.encode(string))
    return n_tokens


def count_prompt_tokens(messages, model):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-0613"]:
        # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return count_prompt_tokens(messages, model="gpt-3.5-turbo-0301")
    elif model in ["gpt-4", "gpt-4-0613"]:
        # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return count_prompt_tokens(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""count_prompt_tokens() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def append_prompt_messages(
    sys_prompt,
    user_prompt,
    assistant_prompt,
    prev_user_prompt,
    prev_assistant_prompt,
    previous_context,
    n_prev_qns=1,
):
    messages = []
    messages.append({"role": "system", "content": sys_prompt})

    # Add previous questions to prompt
    if previous_context and len(previous_context) > 0:
        for i in range(n_prev_qns, 0, -1):
            if len(previous_context) < i * 2:
                continue
            prev_user_prompt = prev_user_prompt.format(prev_qn=previous_context[-i * 2])
            prev_assistant_prompt = prev_assistant_prompt.format(
                prev_sql=previous_context[-i * 2 + 1]
            )

            messages.append({"role": "user", "content": prev_user_prompt})

            messages.append({"role": "assistant", "content": prev_assistant_prompt})

    # Add main question to prompt
    messages.append({"role": "user", "content": user_prompt})

    messages.append({"role": "assistant", "content": assistant_prompt})

    return messages


like_pattern = r"LIKE[\s\S]*'"


def escape_percent(match):
    # Extract the matched group
    group = match.group(0)
    # Replace '%' with '%%' within the matched group
    escaped_group = group.replace("%", "%%")
    # Return the escaped group
    return escaped_group


def test_valid(
    query, db_name
):  # Test to check that SQL query is valid, empty results considered valid here
    try:
        if db_name in c.train_db_postgres or db_name in c.autoeval_db_postgres:
            db_creds = c.creds_local_pg
            db_url = f"postgresql://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/{db_name}"
            engine = create_engine(db_url)
            escaped_query = re.sub(
                like_pattern, escape_percent, query, flags=re.IGNORECASE
            )  # ignore case of LIKE
            results_df = func_timeout(
                10, pd.read_sql_query, args=(escaped_query, engine)
            )
            engine.dispose()  # close connection
        else:
            print("Database not found")
            return 0.0, "Database not found"
            # con = psycopg2.connect(**c.creds_local_pg)
            # cur = con.cursor()
            # cur.execute("SET statement_timeout = 10000")
        # elif db_name == "turns":
        #     con = mysql.connector.connect(**creds[db_name])
        #     cur = con.cursor()
        #     cur.execute("SET SESSION MAX_EXECUTION_TIME=10000")
        # cur.execute(response)
        # cur.close()
        # con.close()
    except Exception as e:  # If unable to run SQL on database, test fails
        if engine:
            engine.dispose()  # close connection if query fails/timeouts
        return 0.0, str(e)

    return 1.0, "-"


def add_to_yaml(yaml_file, new_qns):
    # Make directory if necessary
    directory = os.path.dirname(yaml_file)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    except OSError as e:
        print(f"An error occurred while creating the directory '{directory}': {e}")
        return

    try:
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
    except:
        data = []
    data.extend(new_qns)
    with open(yaml_file, "w") as file:
        yaml.dump(data, file)


def write_jsonl(output_file, synth_data):
    # Make directory if necessary
    directory = os.path.dirname(output_file)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    except OSError as e:
        print(f"An error occurred while creating the directory '{directory}': {e}")
        return
    with open(output_file, "a") as f:
        for entry in synth_data:
            f.write(json.dumps(entry) + "\n")


def write_json(results_file, data):
    # Make directory if necessary
    directory = os.path.dirname(results_file)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    except OSError as e:
        print(f"An error occurred while creating the directory '{directory}': {e}")
        return

    # Load existing JSON data from the file
    try:
        with open(results_file, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
    # Append new data to existing data
    existing_data.extend(data)
    # Write the updated data back to the JSON file
    with open(results_file, "w") as f:
        json.dump(existing_data, f, indent=4)


def write_csv(output_file, synth_data):
    # Make directory if necessary
    directory = os.path.dirname(output_file)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    except OSError as e:
        print(f"An error occurred while creating the directory '{directory}': {e}")
        return

    # Extract keys
    if synth_data:
        keys = synth_data[0].keys()
    else:
        return

    # Check if file exists and is non-empty
    file_exists = os.path.isfile(output_file) and os.stat(output_file).st_size > 0

    with open(output_file, "a", newline="") as csv_file:
        dict_writer = csv.DictWriter(csv_file, fieldnames=keys)
        if not file_exists:
            dict_writer.writeheader()
        dict_writer.writerows(synth_data)