from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel,Field

def add(a: int, b: int) -> int:
    try:
        sum = a + b
        return {
            "observation":str(sum),
            "metadata":{
                "error":False
            }
        }
    except Exception as e:
        return {
            "observation":str(e),
            "metadata":{
                "error":True
            }
        }
    
    

class AddToolInput(BaseModel):
    a: int = Field(..., title="First number")
    b: int = Field(..., title="Second number")
def get_add_tool() -> StructuredTool:

    return StructuredTool.from_function(
        func=add,
        name="Addition",
        description="Adds two numbers",
        input_model=AddToolInput

    )