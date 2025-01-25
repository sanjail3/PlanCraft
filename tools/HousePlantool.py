from typing import Optional
from pydantic import BaseModel
from langchain.tools import StructuredTool
from gradio_client import Client

class HousePlanInputs(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    guidance_scale: Optional[float] = 9.0

def house_plan_generator(
    prompt: str,
    negative_prompt: Optional[str] = "",
    guidance_scale: Optional[float] = 9.0
) -> str:
    client = Client("stabilityai/stable-diffusion")
    result = client.predict(
        prompt=prompt,
        negative=negative_prompt,
        scale=guidance_scale,
        api_name="/infer_2"  # Specific endpoint for house plans
    )
    return "\n".join([img['image'] for img in result])

def get_house_plan_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=house_plan_generator,
        name="house_plan_generator",
        description="Generates architectural house plans from text descriptions",
        args_schema=HousePlanInputs
    )

tool = get_house_plan_tool()
try:
    result = tool.invoke({
        "prompt": "Modern two-story house with large windows",
        "negative_prompt": "traditional elements",
        "guidance_scale": 10.5
    })
    print("Generated images:", result)
except Exception as e:
    print("Validation failed:", e)

