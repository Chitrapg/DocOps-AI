# integrations/jira.py
"""
Jira integration - Push test cases to Jira.
Moved from src/jira_client.py
"""
import json
import html
from typing import List, Tuple, Dict, Any
from jira import JIRA
from core.config import settings


def get_jira_client() -> JIRA:
    """Get authenticated Jira client."""
    settings.require_jira()
    options = {'server': settings.JIRA_SERVER}
    return JIRA(options, basic_auth=(settings.JIRA_EMAIL, settings.JIRA_API_TOKEN))


def _to_safe_str(value) -> str:
    """Convert value to safe HTML-escaped string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return html.escape(value)
    try:
        return html.escape(json.dumps(value, ensure_ascii=False))
    except Exception:
        return html.escape(str(value))


def create_testcase_issue(payload: Dict[str, Any], project_key: str = None) -> str:
    """
    Create a Jira issue from test case payload.
    
    Returns the created issue key.
    """
    jira = get_jira_client()
    project = project_key or settings.JIRA_PROJECT_KEY

    # Build description
    tc_id = _to_safe_str(payload.get("testcase_id"))
    business_scenario = _to_safe_str(payload.get("business_scenario"))
    
    preconditions = payload.get("preconditions")
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
    
    test_data = payload.get("test_data_notes")
    if isinstance(test_data, (dict, list)):
        test_data_str = json.dumps(test_data, ensure_ascii=False, indent=2)
    else:
        test_data_str = _to_safe_str(test_data)

    description = "\n\n".join([
        f"Test Case ID: {tc_id}",
        f"Business Scenario: {business_scenario}",
        f"Preconditions: {preconditions_str}",
        "Test Steps:",
        test_steps_str,
        f"Expected Result: {expected_result}",
        f"Test Data: {test_data_str}",
    ])

    summary = business_scenario or tc_id or "Test Case"
    if len(summary) > 140:
        summary = summary[:137] + "..."

    issue = jira.create_issue(fields={
        'project': {'key': project},
        'summary': summary,
        'description': description,
        'issuetype': {'name': 'Task'}
    })
    return issue.key


def push_testcases_to_jira(testcases: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Push multiple test cases to Jira.
    
    Returns tuple of (created_keys, errors).
    """
    created = []
    errors = []
    
    for tc in testcases:
        try:
            key = create_testcase_issue(tc)
            created.append(key)
        except Exception as e:
            errors.append(str(e))
    
    return created, errors


# Backward compatibility
create_testcase_issue_from_payload = create_testcase_issue
