# from langchain.tools import StructuredTool
from gradio_client import Client
from typing import Optional


def house_plan_generator(
    prompt: str,
    negative_prompt: Optional[str] = "",
    guidance_scale: Optional[float] = 9.0
) -> str:
    try:
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
            
        client = Client("stabilityai/stable-diffusion")
        result = client.predict(
            prompt=prompt,
            negative=negative_prompt,
            scale=float(guidance_scale),  # Ensure numeric value
            api_name="/infer_2"
        )
        
        return "Generated images:\n" + "\n".join([img['image'] for img in result])
        
    except Exception as e:
        return f"Generation failed: {str(e)}"

result=house_plan_generator("As a BIM engineer, please draw a 3D 1st floor plan design for new building including  class room office with learning and sitting space for students  of 11 th new project university building")

print(result)

