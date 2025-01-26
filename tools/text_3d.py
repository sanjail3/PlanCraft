import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("MESHY_API_KEY")
API_URL = "https://api.meshy.ai/openapi/v2/text-to-3d"

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

if __name__ == "__main__":
    prompt = input("Enter a description for the 3D model: ")
    try:
        task_id = create_preview_task(prompt)
        print(f"Task created successfully. Task ID: {task_id}")
        result = retrieve_task_result(task_id)
        print("3D model generated successfully!")
        print("Model URLs:", result["model_urls"])
        print("Thumbnail URL:", result["thumbnail_url"])
    except Exception as e:
        print("An error occurred:", str(e))
