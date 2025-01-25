from typing import Optional
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import requests
import os
from dotenv import load_dotenv
import time
import base64
load_dotenv() 


class CreateImage3DArgsSchema(BaseModel):
    image_url: str = Field(
        ...,
        description="Public URL or base64 data URI of the input image (jpg/jpeg/png)"
    )
    ai_model: Optional[str] = Field(
        default="meshy-4",
        description="AI model ID (meshy-4 for hard surfaces)"
    )
    topology: Optional[str] = Field(
        default="triangle",
        description="Mesh topology: 'quad' or 'triangle'"
    )
    target_polycount: Optional[int] = Field(
        default=30000,
        description="Target polygon count (10k-300k)"
    )
    should_remesh: Optional[bool] = Field(
        default=True,
        description="Enable remeshing phase"
    )
    enable_pbr: Optional[bool] = Field(
        default=False,
        description="Generate PBR texture maps"
    )
    should_texture: Optional[bool] = Field(
        default=True,
        description="Enable texturing phase"
    )
    symmetry_mode: Optional[str] = Field(
        default="auto",
        description="Symmetry mode: 'off', 'auto', or 'on'"
    )

# ================== CORE FUNCTIONS ================== #
def create_image_to_3d_task(**kwargs) -> str:
    """Create a new Image-to-3D conversion task with retries"""
    image_url = kwargs.get("image_url") #local image path
    if not image_url:
        raise ValueError("Image URL is required")
    with open(image_url, "rb") as img_file:
        imgbase64 = base64.b64encode(img_file.read()).decode("utf-8")

    headers = {"Authorization": f"Bearer {os.getenv('MESHY_API_KEY')}"}
    payload = {k: v for k, v in kwargs.items() if v is not None}
    payload["image_url"] =f"data:image/jpeg;base64,{imgbase64}"
    for attempt in range(3):  # Reduced retries
        try:
            response = requests.post(
                "https://api.meshy.ai/openapi/v1/image-to-3d",
                headers=headers,
                json=payload,
                timeout=10  # Add timeout
            )
            response.raise_for_status()
            return response.json()["result"]
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = (2 ** attempt) + 5  # Longer backoff
                # logger.warning(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
                
    raise Exception("Failed after 3 retries")

def create_and_retrieve_3d_model(**kwargs) -> dict:

    """Combined tool with safer polling"""
    task_id = create_image_to_3d_task(**kwargs)

    print(task_id)
    headers = {"Authorization": f"Bearer {os.getenv('MESHY_API_KEY')}"}
    
    max_retries = 5  # Reduced from 10
    base_delay = 20  # Increased base delay
    
    for attempt in range(max_retries):
        try:
            time.sleep(base_delay * (attempt + 1))  # Linear backoff
            
            # Use direct task endpoint if available
            response = requests.get(
                f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            task_data = response.json()
            
            if task_data["status"] == "SUCCEEDED":
                return {
                    "observation": "3D model created successfully",
                    "metadata": {
                        "model_urls": task_data["model_urls"],
                        "thumbnail": task_data["thumbnail_url"],
                        "textures": task_data.get("texture_urls", []),
                        "error": False
                    }
                    
                }
            elif task_data["status"] in ["FAILED", "CANCELLED"]:
                return {
                    "observation": "3D model creation failed",
                    "metadata": {
                        "error": True
                    }
                }

                
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
            
    return {
        "observation": "3D model creation failed",
        "metadata": {
            "error": True
        }
    }


# def get_image_to_3d_tool() -> StructuredTool:
#     return StructuredTool(
#         name="image_to_3d_task_lister",
#         description="Retrieves paginated list of Image-to-3D conversion tasks with sorting capabilities",
#         args_schema=Image3DArgsSchema,
#         func=get_image_to_3d_tasks
#     )


def image_to_3d_creator_tool() -> StructuredTool:
    return StructuredTool(
        name="3d_model_generator",
        description="Create 3D models from images and automatically retrieve results",
        args_schema=CreateImage3DArgsSchema,
        func=create_and_retrieve_3d_model
    )


# tool = image_to_3d_creator_tool()
# result = tool.run({
#     "image_url": "https://aisaasvalidator.blob.core.windows.net/media-uploads/d5334c16-3846-43c1-b7cc-af14b70933dd.jpg",
#     "enable_pbr": True
# })

# print(result)