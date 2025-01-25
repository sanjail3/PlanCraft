from typing import Optional
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import requests
import os

class Image3DArgsSchema(BaseModel):
    page_num: Optional[int] = Field(
        default=1,
        description="Page number for pagination. Starts and defaults to 1."
    )
    page_size: Optional[int] = Field(
        default=10,
        description="Page size limit. Defaults to 10 items. Maximum allowed is 50 items."
    )
    sort_by: Optional[str] = Field(
        default="-created_at",
        description="Sorting field: '+created_at' for ascending, '-created_at' for descending"
    )

def get_image_to_3d_tasks(
    page_num: int = 1,
    page_size: int = 10,
    sort_by: str = "-created_at"
) -> list:
    """
    Retrieves a list of Image-to-3D conversion tasks with pagination and sorting.
    
    Args:
        page_num: Page number for paginated results
        page_size: Number of items per page (max 50)
        sort_by: Sorting order (+created_at/-created_at)
        
    Returns:
        List of task objects containing model URLs, status, and metadata
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('MESHY_API_KEY')}"
    }
    
    response = requests.get(
        "https://api.meshy.ai/openapi/v1/image-to-3d",
        headers=headers,
        params={
            "page_num": page_num,
            "page_size": page_size,
            "sort_by": sort_by
        }
    )
    response.raise_for_status()
    return response.json()

def get_image_to_3d_tool() -> StructuredTool:
    return StructuredTool(
        name="image_to_3d_task_lister",
        description="Retrieves paginated list of Image-to-3D conversion tasks with sorting capabilities",
        args_schema=Image3DArgsSchema,
        func=get_image_to_3d_tasks
    )


tool = get_image_to_3d_tool()
result = tool.run({"page_size": 5, "sort_by": "+created_at"})