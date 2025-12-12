# src/jira_client.py
import json
import html
from jira import JIRA
from app.config import settings

def get_jira_client():
    """
    Returns an authenticated JIRA client.
    Raises ValueError if required env vars are missing.
    """
    if not settings.JIRA_SERVER or not settings.JIRA_API_TOKEN or not settings.JIRA_EMAIL:
        raise ValueError("JIRA_SERVER, JIRA_EMAIL and JIRA_API_TOKEN must be set as environment variables.")
    options = {'server': settings.JIRA_SERVER}
    jira = JIRA(options, basic_auth=(settings.JIRA_EMAIL, settings.JIRA_API_TOKEN))
    return jira

def _to_safe_str(value):
    """
    Convert arbitrary Python value to a safe HTML-escaped string for Jira descriptions.
    - Strings are escaped.
    - dicts, lists, numbers are json.dumps'ed and then escaped.
    - None returns empty string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        # Trim very long strings (avoid huge fields)
        s = value
    else:
        try:
            s = json.dumps(value, ensure_ascii=False)
        except Exception:
            # fallback to str()
            s = str(value)
    # Escape characters which may break Jira markup
    return html.escape(s)

def create_testcase_issue_from_payload(payload: dict, project_key: str = None):
    """
    payload expects keys:
      - testcase_id
      - business_scenario
      - preconditions
      - test_steps
      - expected_result
      - test_data_notes
      - related_requirement_ids
      - tags

    This function coerces all values safely to strings, constructs a readable description,
    and creates a Jira issue of issuetype 'Test' by default (change if your project uses a different type).
    Returns the created issue key or raises an Exception with the error message.
    """
    jira = get_jira_client()
    project = project_key or settings.JIRA_PROJECT_KEY

    # Safely convert fields to strings
    tc_id = _to_safe_str(payload.get("testcase_id"))
    business_scenario = _to_safe_str(payload.get("business_scenario"))
    preconditions = payload.get("preconditions")
    # preconditions might be semicolon-separated string or a list; normalize to a string
    if isinstance(preconditions, list):
        preconditions_str = "; ".join(str(x) for x in preconditions)
    else:
        preconditions_str = _to_safe_str(preconditions)

    test_steps = payload.get("test_steps")
    if isinstance(test_steps, list):
        test_steps_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(test_steps))
    else:
        test_steps_str = _to_safe_str(test_steps)

    expected_result = _to_safe_str(payload.get("expected_result"))
    test_data_notes = payload.get("test_data_notes")
    # test_data_notes might be dict -> convert to pretty JSON string
    if isinstance(test_data_notes, (dict, list)):
        try:
            test_data_str = json.dumps(test_data_notes, ensure_ascii=False, indent=2)
        except Exception:
            test_data_str = str(test_data_notes)
    else:
        test_data_str = _to_safe_str(test_data_notes)

    related = payload.get("related_requirement_ids")
    if isinstance(related, list):
        related_str = ", ".join(str(x) for x in related)
    else:
        related_str = _to_safe_str(related)

    tags = payload.get("tags")
    if isinstance(tags, list):
        tags_str = ", ".join(str(x) for x in tags)
    else:
        tags_str = _to_safe_str(tags)

    # Build description with clear labeled sections; use Jira's wiki/markdown-lite plain text.
    description_lines = [
        f"Test Case ID: {tc_id}",
        f"Business Scenario: {business_scenario}",
        f"Preconditions: {preconditions_str}",
        "Test Steps:",
        test_steps_str,
        f"Expected Result: {expected_result}",
        f"Test Data / Notes: {test_data_str}",
        f"Related Requirement IDs: {related_str}",
        f"Tags: {tags_str}"
    ]
    # Join with double newlines for readability in Jira description field
    description = "\n\n".join(description_lines)

    # Summary/summary fallback - short and safe
    summary = business_scenario or (tc_id or "TradeFin Test Case")
    if len(summary) > 140:
        summary = summary[:137] + "..."

    # Default issue type - change if your Jira uses a custom Test issue type
    issuetype_name = "Task"

    issue_dict = {
        'project': {'key': project},
        'summary': summary,
        'description': description,
        'issuetype': {'name': issuetype_name}
    }

    try:
        issue = jira.create_issue(fields=issue_dict)
        return issue.key
    except Exception as e:
        # Raise a clearer message for UI consumption
        raise RuntimeError(f"Failed to create Jira issue: {e}")
