import requests
import time
import os
from dotenv import load_dotenv
from pydantic.v1 import BaseModel, Field
from langchain.tools import StructuredTool

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("MESHY_API_KEY")
API_URL = "https://api.meshy.ai/openapi/v2/text-to-3d"
class CreateText3DArgsSchema(BaseModel):
    prompt : str = Field(
        ...,
        description="Description of the desired 3D model"
    )
def create_preview_task(prompt):
    """
    Create a Text-to-3D preview task.

    Args:
        prompt (str): Description of the desired 3D model.

    Returns:
        str: Task ID of the created preview task.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "mode": "preview",
        "prompt": prompt,
        "art_style": "realistic",  # Options: 'realistic', 'sculpture'
        "should_remesh": True
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    task_id = response.json().get("result")
    if not task_id:
        raise ValueError("Failed to retrieve task ID.")
    return task_id

def retrieve_task_result(task_id):
    """
    Retrieve the result of a Text-to-3D task.

    Args:
        task_id (str): The ID of the task to retrieve.

    Returns:
        dict: Contains model URLs and thumbnail URL if successful.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    task_url = f"{API_URL}/{task_id}"

    while True:
        response = requests.get(task_url, headers=headers)
        response.raise_for_status()
        task_data = response.json()

        status = task_data.get("status")
        if status == "SUCCEEDED":
            return {
                "model_urls": task_data.get("model_urls"),
                "thumbnail_url": task_data.get("thumbnail_url")
            }
        elif status in ["FAILED", "CANCELLED"]:
            raise RuntimeError(f"Task {status.lower()}.")
        else:
            print("Task is processing, checking again in 20 seconds...")
            time.sleep(20)

def generate_3d_model(prompt:str):
    try:
        
        task_id = create_preview_task(prompt)
        result = retrieve_task_result(task_id)
        return {
            "observation": "3D model created successfully",
            "metadata": {
                "model_urls": result["model_urls"],
                "thumbnail": result["thumbnail_url"],
                "error": False
            }
        }
    except Exception as e:
        return {
            "observation": f"3D model creation failed: {str(e)}",
            "metadata": {
                "error": True
            }
        }

def image_to_3d_creator_tool() -> StructuredTool:
    return StructuredTool(
        name="3d_model_generator",
        description="Create 3D models from images and automatically retrieve results",
        args_schema=CreateText3DArgsSchema,
        func=generate_3d_model
    )


