import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import Tool
from pydantic import BaseModel
from langchain.tools.base import StructuredTool
from pydantic.v1 import BaseModel,Field

class BudgetCalculation:
    def __init__(self):

        load_dotenv()  
        
        self.client = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"), # type: ignore
            azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )

    def budget_cal_prompt(self):

        return """
        You are a construction project manager tasked with estimating the budget for a new residential building project. 
        Project Specifications:
        {project_details}

        Provide a comprehensive budget estimation with:
        1. Detailed Materials Breakdown
            - Structural materials
            - Finishing materials
            - Interior materials
            - Quantities and specifications

        2. Cost Analysis
            - Raw material costs
            - Labor expenses
            - Electrical and plumbing costs
            - Interior furnishing
            - Overhead costs

        3. Project Timeline
            - Estimated duration
            - Workforce requirements
            - Critical milestones

        4. Financial Model
            - Itemized cost table
            - Budget allocation percentages
            - Contingency provisions
            - Cost optimization strategies

        Deliver a precise, data-driven estimation with clear methodology and potential cost variables.
        CONSTRAINTS:
        - Budget should not exceed the estimated budget provided , If it exceeds, provide a detailed breakdown of the additional costs.
        - Ensure the percentage data , numbericals are in table format and other details are in points.
        - ensure the response is precise with proper format.
        - DONT ADD ANY EXTRA DETAILS IN THE RESPONSE.
        """

    def calculate_budget_estimates(self, budget_input):

        try:
           
            prompt = ChatPromptTemplate.from_template(self.budget_cal_prompt())
            chain = prompt | self.client | StrOutputParser()
            response = chain.invoke({"project_details": str(budget_input)})
            
            return response
        
        except Exception as e:
            return f"Budget Calculation Error: {str(e)}"

class BudgetInputs(BaseModel):
    square_ft: float = Field(..., description="Total square footage of the project")
    location: str = Field(..., description="Location of the project")
    estimated_budget: float = Field(..., description="Estimated budget for the project")
    demographics: str = Field(..., description="Demographics of the area")
    no_of_bks: int = Field(..., description="Number of blocks in the project")
    stories: int = Field(..., description="Number of stories in the project")
    main_road: str = Field(..., description="Is the project located on a main road? (Yes/No)")
    guest_rooms: int = Field(..., description="Number of guest rooms")
    basements: int = Field(..., description="Number of basements in the project")
    parking: int = Field(..., description="Number of parking spaces")
    quality_type: str = Field(..., description="Type of quality (e.g., High, Medium, Low)")

def budget_calculation(project_details: dict,) -> str:
    """Calculate construction budget estimates based on project specifications."""
    budget_calculator = BudgetCalculation()
    return budget_calculator.calculate_budget_estimates(project_details)

def get_budget_calculation_tool(input:BudgetInputs) -> StructuredTool:
    return StructuredTool.from_function(
        func=budget_calculation,
        name="budget_calculator",
        description="Generates detailed construction budget estimates based on project specifications including materials, costs, timeline, and financial breakdown.",
        args_schema=BudgetInputs # type: ignore
    )

