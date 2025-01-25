import os
from dotenv import load_dotenv,find_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import Tool
from langchain.tools.base import StructuredTool
from pydantic.v1 import BaseModel, Field
from enum import Enum
from typing import Union
def clean_paranthesis(string):
    return string.replace("{","{{").replace("}","}}")

class BudgetCalculation:
    def __init__(self):
        print(load_dotenv(find_dotenv()))
        
        self.client = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY",),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")
        )

    def budget_cal_prompt(self):
        return """
        You are a construction project manager tasked with estimating the budget for a new residential building project. 

        Provide a comprehensive budget estimation with:
        1. Detailed Materials Breakdown
        2. Cost Analysis
        3. Project Timeline
        4. Financial Model

        CONSTRAINTS:
        - Budget should not exceed the estimated budget provided
        - Use tables for numerical data
        - Be precise and format properly
        """

    def calculate_budget_estimates(self, **budget_input):
        try:
            # print(self.client.invoke("hi"))
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.budget_cal_prompt()),
                ("human", "Details provide by user for house data:{input}")
            ])
            chain = prompt | self.client | StrOutputParser()
            response = chain.invoke({"input": clean_paranthesis(str(budget_input))})
            return {
                "observation": None,
                "metadata": {
                    "error": False,
                    "message": response
                }
            }
        except Exception as e:
            print(f"Error: {str(e)}")
            return {
                "observation": str(e),
                "metadata": {
                    "error": True
                }
            }

# Enum definitions
class location_enum(str, Enum):
    Urban = "Urban"
    Suburban = "Suburban"
    Rural = "Rural"    

class main_road_enum(str, Enum):
    Yes = "Yes"
    No = "No"
    Nearby = "Nearby" 

class guest_room_enum(str, Enum):
    Yes = "Yes"
    No = "No"   

class basement_enum(str, Enum):
    Yes = "Yes"
    No = "No"

class parking_enum(str, Enum):
    Yes = "Yes"
    No = "No"    

class quality_type_enum(str, Enum):
    Basic = "Basic"
    Standard = "Standard"
    Premium = "Premium"
    Luxury = "Luxury"

# Pydantic model for input validation
class BudgetInputs(BaseModel):
    square_ft: Union[int,float] = Field(..., description="Total square footage of the project")
    location: location_enum = Field(description="Location of the project")
    estimated_budget: float = Field(..., description="Estimated budget for the project")
    demographics: str = Field(..., description="Demographics of the area")
    no_of_bks: int = Field(..., description="Number of blocks in the project")
    stories: int = Field(..., description="Number of stories in the project")
    main_road: main_road_enum = Field(description="Proximity to main road")
    guest_rooms: guest_room_enum = Field(description="Presence of guest rooms")
    basements: basement_enum = Field(description="Presence of basements")
    parking: parking_enum = Field(description="Parking availability")
    quality_type: quality_type_enum = Field(description="Construction quality tier")

# Corrected function with proper parameter handling
def budget_calculation(
    square_ft: int,
    location: location_enum,
    estimated_budget: float,
    demographics: str,
    no_of_bks: int,
    stories: int,
    main_road: main_road_enum,
    guest_rooms: guest_room_enum,
    basements: basement_enum,
    parking: parking_enum,
    quality_type: quality_type_enum,
) -> str:
    """Calculate construction budget estimates based on project specifications."""
    budget_calculator = BudgetCalculation()
    project_details = {
        "square_ft": square_ft,
        "location": location,
        "estimated_budget": estimated_budget,
        "demographics": demographics,
        "no_of_bks": no_of_bks,
        "stories": stories,
        "main_road": main_road,
        "guest_rooms": guest_rooms,
        "basements": basements,
        "parking": parking,
        "quality_type": quality_type,
    }
    return budget_calculator.calculate_budget_estimates(**project_details)

def get_budget_calculation_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=budget_calculation,
        name="budget_calculator",
        description="Generates detailed construction budget estimates based on project specifications",
        args_schema=BudgetInputs
    )

def test_budget_calculation_simple():
    # Mocking the AzureChatOpenAI response
    class MockBudgetCalculation(BudgetCalculation):
        def calculate_budget_estimates(self, **budget_input):
            print(budget_calculation(**budget_input))
    
    # Test input data
    test_data = {
        "square_ft": 1500,
        "location": location_enum.Suburban,
        "estimated_budget": 300000,
        "demographics": "Middle-income families",
        "no_of_bks": 2,
        "stories": 3,
        "main_road": main_road_enum.Nearby,
        "guest_rooms": guest_room_enum.No,
        "basements": basement_enum.Yes,
        "parking": parking_enum.Yes,
        "quality_type": quality_type_enum.Standard,
    }

    # Override the BudgetCalculation class with a mock
    budget_calculator = MockBudgetCalculation()
    result = budget_calculator.calculate_budget_estimates(**test_data)

    # Output the results
    print("Test Result:")
    print(result)


# Call the simple test function
if __name__ == "__main__":
    test_budget_calculation_simple()
