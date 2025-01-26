from typing import Optional
from pydantic.v1 import BaseModel,Field
from langchain.tools import StructuredTool
from gradio_client import Client

class HousePlanInputs(BaseModel):
    prompt: str  = Field(description="Text description of the house plan")
    negative_prompt: Optional[str] = Field(description="Negative description of the house plan",default="")
    guidance_scale: Optional[float] = Field(description="Guidance scale for the model",default=9.0)

def house_plan_generator(
    prompt: str,
    negative_prompt: Optional[str] = "",
    guidance_scale: Optional[float] = 9.0
) -> str:
    try:
        # print("Prompt:", prompt)
        final_prompt="As a BIM engineer, please draw a 3D 1st floor plan design for new 2 BHK building , use not contrasting colors to make it look more appealing , use traditional elements for the each section" +prompt
        client = Client("stabilityai/stable-diffusion")
        result = client.predict(
            prompt=final_prompt,
            negative=negative_prompt,
            scale=guidance_scale,
            api_name="/infer_2"  # Specific endpoint for house plans
        )
        return {
            "observation": None,
            "metadata": {
                "error": False,
                "image_path": result
            }
        }
    except Exception as e:
        return {
            "observation": str(e),
            "metadata": {
                "error": True
            }
        }
def get_house_plan_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=house_plan_generator,
        name="house_plan_generator",
        description="Generates architectural house plans from text descriptions",
        args_schema=HousePlanInputs
    )

# tool = get_house_plan_tool()
# try:
#     result = tool.invoke({
#         "prompt": "As a BIM engineer, please draw a 3D 1st floor plan design for new 2 BHK building , use not contrasting colors to make it look more appealing , use traditional elements for the each section",
#         "negative_prompt": "traditional elements",
#         "guidance_scale": 10.5
#     })
#     print("Generated images:", result)
# except Exception as e:
#     print("Validation failed:", e)

