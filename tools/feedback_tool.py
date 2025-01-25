from langchain.tools import StructuredTool
from typing import Optional

def get_user_feedback(query: Optional[str] = None) -> str:
    """Triggers feedback request in Streamlit UI"""
    return "USER_FEEDBACK_REQUESTED"

get_user_feedback_tool = StructuredTool.from_function(
    func=get_user_feedback,
    name="get_user_feedback",
    description="Use this to get user feedback on generated designs"
)