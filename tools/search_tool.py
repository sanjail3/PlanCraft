import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from pydantic.v1 import BaseModel,Field
from langchain.tools.base import StructuredTool
load_dotenv()
import streamlit as st

class DisasterDevelopmentTool:
    def __init__(self):
        # Initialize Azure OpenAI client
        self.client = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"), # type: ignore
            azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
        st.session_state.llm = self.client
        
        self.serp_search = SerpAPIWrapper(
            serpapi_api_key=os.getenv('SERPAPI_API_KEY')
        )
        
        self.search_tool = Tool(
            name="serpapi_search",
            func=self.serp_search.run,
            description="Useful for searching the internet for recent information"
        )
        
        
        prompt = hub.pull("hwchase17/react")
        
        self.agent = create_react_agent(
            llm=self.client, 
            tools=[self.search_tool], 
            prompt=prompt
        )
        # Create an agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=[self.search_tool], 
            verbose=True
        )

    def search_recent_disasters(self, location, years=5):
        """Search recent disasters in specified location"""
        query = f"Major disasters in {location} in past {years} years"
        return self.agent_executor.invoke({"input": query})

    def search_government_developments(self, location):
        """Search government infrastructure developments"""
        query = f"List out current or any upcoming Government infrastructure projects in {location}"
        return self.agent_executor.invoke({"input": query})

    def generate_comprehensive_report(self, location):
        """Generate comprehensive location report"""
        disasters = self.search_recent_disasters(location)
        developments = self.search_government_developments(location)
        
        return f"""
        LOCATION REPORT: {location}

        DISASTERS:
        {disasters['output']}

        DEVELOPMENTS:
        {developments['output']}
        """


class DisasterDevelopmentInput(BaseModel):
    geo_loc : str = Field(description="Geographical location to search for recent disasters and government developments")

def development_tool(geo_loc: str,) -> str:
    """Calculate construction budget estimates based on project specifications."""
    budget_calculator = DisasterDevelopmentTool()
    return budget_calculator.generate_comprehensive_report(geo_loc)

def get_geo_loc_info_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=development_tool,
        name="disaster_development_tool",
        description="Generates a comprehensive report on recent disasters and government developments in a specified location.",
        args_schema=DisasterDevelopmentInput # type: ignore
    )
