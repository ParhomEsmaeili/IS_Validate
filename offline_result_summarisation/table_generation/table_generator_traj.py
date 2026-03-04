import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.general_utils.dict_utils import extractor, dict_path_modif
from typing import Dict, List, Any, Tuple
import glob
import pandas as pd
import copy

RESULT_SEARCHSTRING = {
    'Dice': 'Dice',
    'NSD': 'NSD',
    'NoI': ['NOI', 'Failure_Cases_Fraction']
}
MAPPING_SUBCATEGORIES = {
    'auc': 'nAUC',
    'Init.': 'Init.',
    'Interactive Edit Iter': 'Interactive Edit Iter',
    'NOI': 'nNoI',
    'Failure_Cases_Fraction': 'NoF'
}

TABLE_MAP = {
    'Dice_median': '',
    'NSD_median': '',
    'Dice_mean': '',
    'NSD_mean': '',
    'Dice_auc_median': 'nAUC',
    'NSD_auc_median': 'nAUC',
    'Dice_auc_mean': 'nAUC',
    'NSD_auc_mean': 'nAUC',
    'Normalised_Median_NOI': 'nNoI',
    'Normalised_Mean_NOI': 'nNoI',
    'Interactive Edit Iter': 'Iter.',
    'Failure_Cases_Fraction': 'NoF'
}

RANKING_NON_ITERABLE_MAP = { #This is the mapping for the ranking table to what is displayed in the printed results
    'NoI':{ #for each parent, we denote the substring in the child which needs to be matched in the
        #ranking table
        'NOI':'NOI',
        'Failure_Cases_Fraction': 'NoF'
    },
    'Dice': {
        'auc':'Dice_AUC',
    },
    'NSD': {
        'auc':'NSD_AUC',
    }
}

COLUMN_ORDERING = [
    'Init.',
    'Interactive Edit Iter',
    'nAUC',
    'nNoI',
    'NoF'
]

DATASET_MAPPING = {
    'Dataset001_BrainTumour': 'Brain Tumour Core',
    'Dataset002_Heart': 'Heart',
    'Dataset003_Liver': 'Whole Liver',
    'Dataset004_Hippocampus': 'Whole Hippocampus',
    'Dataset005_Prostate': 'Whole Prostate',
    'Dataset006_Lung': 'Lung Lesion',
    'Dataset007_Pancreas': 'Whole Pancreas',
    'Dataset008_HepaticVessel': 'Hepatic Vessels',
    'Dataset009_Spleen': 'Spleen',
    'Dataset010_Colon': 'Colon'
}

ALGORITHM_MAPPING = {
    'sam2v1': 'SAM2',
    'sammed2dv1': 'SAM-Med2D',
    'sammed3dv1': 'SAM-Med3D',
    'segvolv1': 'SegVol',
    'nnintv1': 'nnInteractive',
    'adaptiveISv1': 'AdaptiveIS'
}

iterable_metrics = {
    'Dice_median',
    'Dice_mean',
    'NSD_median',
    'NSD_mean'
}
#!/usr/bin/env python3
"""
table_generator.py

Generate a summary table (Excel + LaTeX) from a list of algorithm folders.

metric_config.json format (example):
{
  "metric_type_1_path.csv": ["metric_type_1_column_1", "metric_type_1_column_2"],
  "metric_type_2_path.json": ["metric_type_2_column_2", "metric_type_2_column_2"]
}  
Values of the dictionary indicate the columns which need to be extracted for printing, and they also serve as descriptors for
the metrics too when generating tables.
"""

import re

def map_algorithm_name(short_name):
    """Map shortened algorithm names to full display names"""
    # Numeric adatest mapping
    NUMERIC_MAP = {
        "2": "CLoPA-Inst",
        "5": "CLoPA-ConvNorm",
        "6": "CLoPA-ConvNorm",
        # Add more mappings as needed
    }

    patterns = [
        (r'adatest(\d+)-episode(\w+)', None),
        (r'adatest(\d+)', None),
        (r'adadesign(\d+)', None),
    ]

    for pattern, replacement in patterns:
        m = re.match(pattern, short_name)
        if m:
            if pattern.startswith('adatest'):
                num = m.group(1)
                mapped = NUMERIC_MAP.get(num, f'adatest{num}')
                # Handle episodic variant
                if '-episode' in short_name:
                    episode = m.group(2)
                    return f'{mapped}-Episode{episode}' if episode != 'final' else f'{mapped}'
                else:
                    return mapped
            elif pattern.startswith('adadesign'):
                num = m.group(1)
                mapped = NUMERIC_MAP.get(num, f'adadesign{num}')
                return mapped
            else:
                return re.sub(pattern, replacement, short_name)

    return short_name  # Return original if no match
def filter_string_table(input_str: str):
    for key, val in TABLE_MAP.items():
        if key in input_str:
            input_str = input_str.replace(key, val)
    return input_str 

def load_metrics_config(json_dict:str) -> Dict[str, List[str]]:
    cfg = json.loads(json_dict)
    if not isinstance(cfg, dict):
        raise ValueError("metrics config must be a JSON object mapping metric -> [metric information]")
    return cfg


# Example usage:
# my_dict = {'Dice_ ': 1, 'NSD_  ': 2, 'Other': 3}
# my_dict = regex_replace_dict_keys(my_dict)
def sort_by_substring_order(input_list: List[str], priority_list: List[str]) -> List[str]:
    def sort_key(input_list):
        for i, substr in enumerate(priority_list):
            if substr.lower() in input_list.lower():
                return i
        return len(input_list)  # put unmatched at the end
    return sorted(input_list, key=sort_key)

def parse_csv_file(path: str) -> Dict[str, Any]:
    """
    Attempt to parse a CSV of metrics. We support two common layouts:
      - Single row with columns being metric names
      - Two columns: metric_name, value (multiple rows)
    Returns a flat dict metric_name -> value
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.shape[0] == 1:
        # use column names
        row = df.iloc[0]
        return {str(col): row[col] for col in df.columns}
    # else, if DataFrame has columns like ['metric', 'value'] or similar,
    # try to detect metric/value columns
    candidates = [c.lower() for c in df.columns]
    if "metric" in candidates and ("value" in candidates or "val" in candidates):
        metric_col = df.columns[candidates.index("metric")]
        if "value" in candidates:
            value_col = df.columns[candidates.index("value")]
        else:
            value_col = df.columns[candidates.index("val")]
        return {str(r[metric_col]): r[value_col] for _, r in df.iterrows()}
    # fallback: if first column is metric and second is value
    if df.shape[1] >= 2:
        return {str(r[df.columns[0]]): r[df.columns[1]] for _, r in df.iterrows()}
    # otherwise nothing useful
    return {}

def gather_metrics_from_folder(folder_path: str, metrics_config: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Searches a given folder for the metrics files specificied in metrics_config, then extracts the relevant metrics.
    Returns a dictionary of extracted metrics flattened into dictionaries with single layer depth.

    """
    metric_dict: Dict[str, Any] = {}
    for metric_file, metric_info in metrics_config.items():
        #First read the file.
        # print(folder_path)
        # print(metric_file)
        # print(metric_info)
        table = pd.read_csv(os.path.join(folder_path,metric_file))

        for metric_name, extraction_info in metric_info.items():
            if extraction_info == None:
                #In this case, there should only be a single row. Lets verify that, then extract the relevant metric.
                if table.shape[0] != 1:
                    raise ValueError(f"Expected a single row in {metric_file} for metric {metric_name}, but found {table.shape[0]} rows. \n"
                                     "Please amend the metrics config to reflect the correct extraction strategy.")
                metric_dict[metric_name] = table.at[0, metric_name]
            else:
                #In this case, we have some extraction information to use, e.g. specific rows we need to extract.
                if 'rows' in extraction_info:
                    #We need to extract specific rows based on the iteration information provided. We will assume that 
                    #the first column contains the iteration information, but won't actually use this. Instead we will use
                    #string matching. 
                    for search_str in extraction_info['rows']:
                        mask = table.apply(lambda row: row.astype(str).str.contains(search_str, na=False)).any(axis=1)
                        indices = table.index[mask].tolist()
                        if len(indices) != 1:
                            raise ValueError(f"{len({indices})} rows found matching '{search_str}' in {metric_file} for metric {metric_name}. \n")
                        
                        if metric_name in iterable_metrics:
                            if indices[0] == 0:
                                metric_dict[f'{metric_name} Init.'] = table.at[indices[0], metric_name]
                            else:
                                metric_dict[f'{metric_name} Interactive Edit Iter {indices[0]}'] = table.at[indices[0], metric_name]
                        else:
                            metric_dict[metric_name] = table.at[indices[0], metric_name]
                else:
                    raise NotImplementedError("Unsupported extraction information provided in metrics config. Only iters is a supported \n" \
                    "configuration for multi-row extraction at the moment.")
            
        # metric_dict[metric_file] = metric_dict
    

    #Now we will reformat the dict so that the strings can be easily mapped to a table downstream.
    # reformatted_out = {v:out[k] for k,v in MAPPING_METRIC_CATEGORIES.items() if k in out}
    return metric_dict

def pull_rankings_for_task_and_metric(folder_path: str, metrics: List[str]) -> Dict[str, Any]:
    """
    Searches a given folder for the rankings files specificied in metrics, then extracts the relevant rankings.
    Returns a dictionary of extracted rankings flattened into dictionaries with single layer depth.

    """
    pass 
    
def build_table(
    ranking_root: str,
    folders: dict[str],
    task_name: str,
    metrics_config: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Create a DataFrame where rows are algorithms (by folder name) and columns are
    metric_type:metric_name (or just metric_name), in config order.
    
    inputs: 
    folders: dict mapping algorithm name to folder path
    metrics_config: dict mapping metric file names to configuration information for extraction.
    """

    metrics_storage = dict() 
    for folder, folder_path in folders.items():
        metrics_storage[folder] = gather_metrics_from_folder(folder_path, metrics_config)
        
    #Now using the extracted metrics to build a table with all of the relevant information placed together.


    #We will partition the table based on the metric type:
    #Dice, NSD, NoI. This is a fixed property for now. We will search through the dict of extracted metrics by creating a bank
    #of substrings to search for among each type.


    first_alg = list(metrics_storage.keys())[0]
    counts = dict()
    extracted_submetrics = dict()
    for k,v in RESULT_SEARCHSTRING.items():
        #We now count the number of submetrics for each metric type.
        #We do this by counting the number of items which have keys containing a substring matching that.

        #I.e., Dice_auc_median, or Dice_median. In particular, it should only flag for metrics which may have multiple sub
        #metrics that aren't distinct metric types. E.g., median dice at iter 0 and iter 100. 
        if isinstance(v, list):
            submetrics = [key for key in metrics_storage[first_alg] if any([substring in key for substring in v])]
        else:
            submetrics = [key for key in metrics_storage[first_alg] if v in key]

        counts[k] = len(submetrics)
        extracted_submetrics[k] = submetrics

    columns = [[k] * v for k,v in counts.items()]
    columns = [
        i
        for sublist in columns
        for i in sublist
    ]
    ordered_dict = {k:sort_by_substring_order(v, COLUMN_ORDERING) for k,v in extracted_submetrics.items()}
    subcolumns = [
        i
        for sublist in ordered_dict.values()
        for i in sublist
    ]
    #Now we create a multiindex for the columns.
    arrays = [
        columns,
        subcolumns
    ]
    tuples = list(zip(*arrays))
    flat_cols = [('Task', ''), ('Algorithm', '')]
    full_cols = pd.MultiIndex.from_tuples(flat_cols + tuples, names=["Metric Type", "Submetric"]) 
    rows = [] 
    
    if ranking_root is not None: #Its not necessary to create a table but we should have it.
        rankings = pd.read_csv(os.path.join(ranking_root, task_name, 'per_metric_algorithm_rankings.csv'), index_col=0)
        # print(rankings)
    
    bold_cells = set()

    for idx, (alg_name, metrics) in enumerate(metrics_storage.items()):
        if ranking_root is not None:
            rankings_alg = rankings.loc[alg_name]
            # print(rankings_alg)
        row = [task_name, alg_name]
        for col_idx, col in enumerate(tuples):  
            submetric = col[-1]
            metric_value = metrics[submetric] if len(col) > 1 else metrics[col]
            if ranking_root is not None:
                # Cross-match ranking value for this submetric
                #Lets find the name of the ranking metric in the rankings table convention.
                # print(col)         
                is_iterable = [metric for metric in iterable_metrics if metric in submetric]
                assert len(is_iterable) <= 1, "Multiple iterable metrics matched, unexpected behaviour."
                if is_iterable:
                    #Then we need to look at the parent metric type to pull.
                    parent_metric_type = col[0]
                    #The column we look for has to hit on both the parent metric and
                    #a substring of the submetric.
                    #Lets strip the metric component from the submetric.
                    #e.g. Dice_median Init to just Init.
                    copy_submetric = copy.deepcopy(submetric)
                    # print(is_iterable, parent_metric_type, submetric, copy_submetric)
                    stripped_submetric = copy_submetric.replace(is_iterable[0], '').strip()
                    #Next we strip the trajectory_AUC component.
                    if '_trajectory_auc' in stripped_submetric:
                        stripped_submetric = stripped_submetric.replace('_trajectory_auc', '')
                    ranking_name = parent_metric_type + f' {stripped_submetric}'
                    #lets extract the rank now.
                    
                    ranking_value = rankings_alg[ranking_name]
                    if ranking_value == 1:
                        best_val = True
                        #then it was the best one!
                    else:
                        best_val = False
                else:
                    # Handle non-iterable logic
                    parent_metric_type = col[0]
                    #Lets extract the colname in ranking from the dict.
                    if parent_metric_type not in RANKING_NON_ITERABLE_MAP:
                        raise ValueError(f"Parent metric type {parent_metric_type} not found in RANKING_NON_ITERABLE_MAP. Please update this mapping to include the relevant parent metric types and their corresponding ranking column names.")
                    #Lets find which key we need to pull from.
                    key_to_pull = [key for key in RANKING_NON_ITERABLE_MAP[parent_metric_type] if key in submetric]
                    if len(key_to_pull) != 1:
                        raise ValueError(f"Expected to find exactly one matching key in RANKING_NON_ITERABLE_MAP for parent metric type {parent_metric_type} and submetric {submetric}, but found {len(key_to_pull)}. Please update the mapping or check the submetric naming.")
                    # print(submetric, key_to_pull, RANKING_NON_ITERABLE_MAP[parent_metric_type][key_to_pull[0]])
                    ranking_value = rankings_alg[RANKING_NON_ITERABLE_MAP[parent_metric_type][key_to_pull[0]]]
                    if ranking_value == 1:
                        best_val = True
                    else:
                        best_val = False
            if ranking_root is not None and best_val:
                bold_cells.add((idx, col))  # Store row index and column tuple
            row.append(metric_value)
        rows.append(row)
    df = pd.DataFrame(rows, columns=full_cols)
    
    #We filter out the bold cells where the column tuple has another match with another cell. this means they were
    #tied for first place, so we should bold neither.
    for col in set([col for _, col in bold_cells]):
        tied_cells = [cell for cell in bold_cells if cell[1] == col]
        if len(tied_cells) > 1:
            for cell in tied_cells:
                bold_cells.remove(cell)

    # Round numeric columns to 3 decimal places
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].round(3)

    # Convert NOI columns to percentage for readability
    noi_cols = [col for col in df.columns if 'NOI' in col[1]]
    for col in noi_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') * 100

    # Format numeric columns: 3 decimal places for Dice/NSD, 3 sig figs for NOI
    for col in df.columns:
        if col[0] in ['Dice', 'NSD']:
            df[col] = df[col].apply(lambda x: f"{float(x):.3f}" if pd.notna(x) and str(x) != '' else str(x))
        elif col[0] == 'NoI':
            df[col] = df[col].apply(lambda x: f"{float(x):.3g}" if pd.notna(x) and str(x) != '' else str(x))

    # Apply bold formatting to best-performing cells
    for row_idx, col_tuple in bold_cells:
        col_pos = df.columns.get_loc(col_tuple)
        df.iloc[row_idx, col_pos] = f"**{df.iloc[row_idx, col_pos]}**"
    
    #Now we strip the _traj_auc component from the column names for submetric row for better display.
    # print(df)
    df.columns = pd.MultiIndex.from_tuples([
        (lvl0, lvl1.replace('_trajectory_auc', '')) for lvl0, lvl1 in df.columns
    ])
    return df

# def write_excel(df: pd.DataFrame, folder_root: str) -> None:
#     """
#     Write DataFrame to Excel with bold formatting applied to cells marked with **.
#     """
#     from openpyxl.styles import Font
    
#     # Ensure parent dir exists
#     os.makedirs(folder_root, exist_ok=True)
#     excel_path = os.path.join(folder_root, 'summary.xlsx')
#     df.to_excel(excel_path, sheet_name="summary", index=False)
    
#     # Apply bold formatting to cells with ** markers
#     from openpyxl import load_workbook
#     wb = load_workbook(excel_path)
#     ws = wb.active
    
#     for row in ws.iter_rows():
#         for cell in row:
#             if cell.value and isinstance(cell.value, str) and '**' in cell.value:
#                 # Remove the ** markers and apply bold
#                 cell.value = cell.value.replace('**', '')
#                 cell.font = Font(bold=True)
    
#     wb.save(excel_path)

def write_csv(df: pd.DataFrame, folder_root: str) -> None:
    """
    Write DataFrame to CSV. Overwrites if exists.
    """
    os.makedirs(folder_root, exist_ok=True)
    df.to_csv(os.path.join(folder_root, 'summary.csv'), index=False)

    # Open the workbook and select the sheet
    # wb = load_workbook(os.path.join(folder_root, 'summary.xlsx'))
    # ws = wb.active

    # # Auto-adjust column width and enable wrap text
    # for col in ws.columns:
    #     max_length = 0
    #     col_letter = get_column_letter(col[0].column)
    #     for cell in col:
    #         try:
    #             if cell.value:
    #                 max_length = max(max_length, len(str(cell.value)))
    #         except:
    #             pass
    #     ws.column_dimensions[col_letter].width = min(max_length + 2, 50)  # limit max width
    #     for cell in col:
    #         cell.alignment = cell.alignment.copy(wrap_text=True)

    # wb.save(os.path.join(folder_root, 'summary.xlsx'))

def generate_latex(df: pd.DataFrame, caption: str = "", label: str = "", output_path: str = "") -> str:
    """
    Generate a LaTeX table string via pandas' to_latex.
    The returned string can be pasted into a LaTeX document.
    """

    # Generate LaTeX with MultiIndex columns
    df['Task'] = df['Task'].map(DATASET_MAPPING).fillna(df['Task'])

    #Lets apply a filter on the algorithm mapping. 
    df['Algorithm'] = df['Algorithm'].map(ALGORITHM_MAPPING).fillna(df['Algorithm']).map(map_algorithm_name)
    
    # Convert ** markers to LaTeX bold before converting to LaTeX
    df = df.map(lambda x: re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', str(x)) if isinstance(x, str) else x)
    
    latex_str = df.to_latex(index=False, multirow=True, multicolumn=True, longtable=False, caption=caption, label=label, na_rep="", float_format=lambda x: f'{x:g}')
    latex_str = merge_task_column_latex(latex_str, task_col=0)

    # Post-process LaTeX to merge 'Algorithm' header and remove subcolumn split for it
    lines = latex_str.splitlines()
    # Find the header line with column names (usually 4th line)
    for i, line in enumerate(lines):
        if '\\toprule' in line:
            header_start = i + 1
            lines[i] = '\hline' #We want a hline, not a toprule.
            break


    # The next two lines are the multiindex headers
    col_header = lines[header_start]
    subcol_header = lines[header_start + 1]
    
    #Lets tidy up this table header and the initial table object.
    subcol_number = df.columns.get_level_values(0).value_counts().to_dict() #This gives us a dict describing the number of subcolumns
    #per metric type. This is required to construct the table in the first place.
    tabular_shape = ''.join(['|' + 'c' * subcol_number[col_name] for col_name in df.columns.get_level_values(0).unique()]) + '|'
    table_env = [f'\\begin{{tabular}}{{{tabular_shape}}}', '\\hline']
    #First the task column.
    # col_header = ['\\multirow{2}{*}{Task}', '\\multirow{2}{*}{Algorithm}', '\\']
    col_header = [f'\\multirow{{2}}{{*}}{{{name}}}' if len(df[name].shape) == 1 else f'\\multicolumn{{{df[name].shape[1]}}}{{c|}}{{{name}}}' for name in df.columns.get_level_values(0).unique()]
    #Phew. This is a very involved line. Essentially we check if the column has subcolumns. If it does then we use multi-column structure. Otherwise
    #we use multirow structure. 
    
    #Now we need to construct the subcolumn header. Need to extract the indices of the final submetrics for each submetric
    #as this is required for the multicolumn structure.
    final_submetrics = {df[metric].columns[-1]:df.columns.get_level_values(1).get_loc(df[metric].columns[-1]) for metric in df.columns.get_level_values(0).unique() if len(df[metric].shape) > 1}
    subcol_header = ['' if submetric == '' else (f'\\multicolumn{{1}}{{c|}}{{{filter_string_table(submetric)}}}' if submetric not in final_submetrics else filter_string_table(submetric)) for submetric in df.columns.get_level_values(1)]

    # del lines[header_start - 2: header_start + 3] 
    lines[header_start - 2] = table_env[0]
    lines[header_start - 1] = table_env[1]
    lines[header_start] = ' & '.join(col_header) + ' \\\\ ' + f'\\cline{{3-{sum(subcol_number.values())}}}'
    lines[header_start + 1] = ' & '.join(subcol_header) + ' \\\\ \\hline'

    del lines[header_start + 2]
    #Now we filter the results rows. 

    for i, line in enumerate(lines):
        if '\\bottomrule' in line:
            result_end= i
            del lines[i] # lines[i] = '\hline' #We want a hline, not a toprule.
            break
    
    num_algo = len(df['Algorithm'].unique())
    #now filtering.
    for row_index, i in enumerate(range(header_start + 2, result_end)):
        split_row = lines[i].split('&')
        filtered_row = [substring if idx in final_submetrics.values()  or idx in [0,1] else f'\\multicolumn{{1}}{{c|}}{{{substring}}}' for idx, substring in enumerate(split_row)]
        if (row_index + 1) % num_algo:
            filtered_row.append('\\cline{{2-{}}}'.format(sum(subcol_number.values())))
        else:
            filtered_row.append('\\hline')
        lines[i] = ' & '.join(filtered_row[:-1]) + filtered_row[-1]
    
    # num_task = len(df['Task'].unique())
    
    # #Creating a duplicate list:
    # final_lines = lines[:header_start + 2]
    # #Now we will merge rows into single strings separated by &
    # for i in range(num_task):
    #     start_idx = header_start + 2 + i * num_algo
    #     end_idx = start_idx + num_algo
    #     merged_row = ''.join([lines[j] for j in range(start_idx, end_idx)])
    #     final_lines.append(merged_row)
    # final_lines.extend(lines[result_end:])

    # Add \usepackage{multirow} comment
    # final_lines.insert(0, '% Requires \\usepackage{multirow}')
    joined_lines = '\n'.join(lines)
    #We also need to add space for each & used.
    joined_lines = joined_lines.replace('&', '\n &')
    with open(output_path, 'w') as f:
        f.write(
            joined_lines
        )
    return

def merge_task_column_latex(latex_str, task_col=0):
    lines = latex_str.splitlines()
    start = next(i for i, l in enumerate(lines) if '\\midrule' in l) + 1
    end = next(i for i, l in enumerate(lines) if '\\bottomrule' in l)
    task_vals = [lines[i].split('&')[task_col].strip() for i in range(start, end)]
    i = 0
    while i < len(task_vals):
        val = task_vals[i]
        run = 1
        while i + run < len(task_vals) and task_vals[i + run] == val:
            run += 1
        if run > 1:
            lines[start + i] = lines[start + i].replace(val, f'\\multirow{{{run}}}{{*}}{{{val}}}', 1)
            for j in range(1, run):
                parts = lines[start + i + j].split('&')
                parts[task_col] = ' '
                lines[start + i + j] = '&'.join(parts)
        i += run
    lines.insert(0, '% Requires \\usepackage{multirow}')
    return '\n'.join(lines)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate summary table (Excel + LaTeX) from algorithm result folders.")
    parser.add_argument("--ranking_root", type=str, required=False, help="Root path to the folder which contains the rankings across all the algorithms.")
    parser.add_argument("--metrics_root", type=str, required=True, help="Root path to the folder which contains the metrics across all the algorithms.")
    parser.add_argument("--algorithm_names", nargs="+", required=True, help="Names of algorithms to summarise.")
    #We explicitly pass algorithm names as we will use this to print the table!
    parser.add_argument("--experiment_subpath", nargs="+", required=True, help="Subpath under each algorithm folder where metrics are stored.")
    #NOTE: The subpath is assumed to be the same for all algorithms! 
    parser.add_argument("--metrics_config", required=True, help="JSON file mapping metrics to information required for pulling.")
    parser.add_argument("--output_root", required=True, help="Output root path for Excel (.xlsx) and LaTeX (.tex) files.")
    parser.add_argument("--caption", default="", help="Caption to include in LaTeX table.")
    parser.add_argument("--label", default="", help="Label to include in LaTeX table.")
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = load_metrics_config(args.metrics_config)
    df_cumulative = pd.DataFrame()
    for experiment_subpath in args.experiment_subpath:
        folders = {alg_name: os.path.join(args.metrics_root, alg_name, experiment_subpath) for alg_name in args.algorithm_names}
        if os.name == 'posix':
            df_cumulative = pd.concat([df_cumulative, build_table(args.ranking_root, folders, experiment_subpath.split('/')[0], cfg)])
            
            
        else:
            raise NotImplementedError("Windows OS is not currently supported for table generation.")

    write_csv(df_cumulative, args.output_root)
    generate_latex(df_cumulative, caption=args.caption, label=args.label, output_path=args.output_root + '/summary_tex.txt')

if __name__ == "__main__":
    main()