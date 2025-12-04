#src/parser.py
import re

def extract_markdown_table(md: str):
    """
    Find the first markdown table that looks like it has 6 columns and return rows as list of lists.
    """
    lines = md.splitlines()
    # find table header line (containing pipes and at least 6 headers)
    table_start = None
    for i, ln in enumerate(lines):
        if '|' in ln and ln.count('|') >= 6:
            # next line should be a separator like |---|---|
            if i+1 < len(lines) and re.search(r'\|\s*-+', lines[i+1]):
                table_start = i
                break
    if table_start is None:
        # fallback: find lines containing 6 pipes and treat them as rows
        rows = [ln for ln in lines if ln.count('|') >= 6]
    else:
        rows = []
        for ln in lines[table_start:]:
            if not ln.strip():
                break
            if ln.count('|') >= 2:
                rows.append(ln)
            else:
                break
    # normalize rows: split into columns and trim
    parsed = []
    for r in rows:
        # split by '|' and remove leading/trailing empty if table uses surrounding pipes
        parts = [p.strip() for p in r.split('|')]
        # remove empty leading/trailing
        if parts and parts[0] == '':
            parts = parts[1:]
        if parts and parts[-1] == '':
            parts = parts[:-1]
        parsed.append(parts)
    # If header + separator at top, remove separator row
    if len(parsed) >= 2 and re.match(r'^-+$', ''.join(parsed[1]).replace(' ', '').replace('|','').replace(':','').replace('-','-')):
        parsed.pop(1)
    # Return only rows with exactly 6 columns (most likely data rows)
    data_rows = [r for r in parsed if len(r) == 6]
    return data_rows

def row_to_issue_payload(row):
    """
    map 6 columns -> structured dict:
    0: Test Case ID
    1: Business Scenario
    2: Preconditions
    3: Test Steps
    4: Expected Result
    5: Test Data / Notes
    """
    return {
        "testcase_id": row[0],
        "business_scenario": row[1],
        "preconditions": row[2],
        "test_steps": row[3],
        "expected_result": row[4],
        "test_data_notes": row[5]
    }
