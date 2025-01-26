from tools.addition import get_add_tool
from tools.HousePlantool import get_house_plan_tool
from tools.budget_calculation import get_budget_calculation_tool
from tools.text_3d import image_to_3d_creator_tool
from langchain.prompts import ChatPromptTemplate
from typing import Generator
from planner import MultiToolAgentStep, BedrockClaudeAgentPlanner, AgentFinish, MultiToolAgentAction,ToolResult
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from tools.python_tool import get_python_lc_tool,get_matching_values_tool
from tools.plot_generator import get_plot_gen_lc_tool
from data import data
from loguru import logger
import json
from utils import clean_paranthesis
def generate_plan(prompt,model_name):
    chain = prompt| ChatBedrock(model_id=model_name)
    return chain.invoke({}).content
load_dotenv()

"""        "square_ft": square_ft,
        "location": location,
        "estimated_budget": estimated_budget,
        "demographics": demographics,
        "no_of_bks": no_of_bks,
        "stories": stories,
        "main_road": main_road,
        "guest_rooms": guest_rooms,
        "basements": basements,
        "parking": parking,
        "quality_type": quality_type"""
class PlanAgent:
    def __init__(self, data,max_steps=8):
        self.state = {}
        self.data = data
        self.max_steps = max_steps


    def prepare_agent(self):
        prompt = """
        You are a professional architectural agent with expertise in construction planning, budgeting, and visualization. You have been tasked with analyzing user inputs and creating a comprehensive plan that includes:

    1. **Location and Soil Analysis**: Assess the location, soil type, and environmental factors to provide the foundation for the construction plan.
    2. **Budget Planning**: Based on the user-provided budget, create a detailed budget estimation for the house or construction project in a more structured way.
    3. **Visualization of Budget Plan**: Generate 3-4 clear and informative plots to visualize the budget allocation across different components like foundation, materials, labor, etc.
    4. **Floor Plan Creation**: Based on the budget and user preferences, generate a detailed floor plan description using the house plan generator tool.
    5. **3D Model Generation**: Create a 3D digital twin or model of the house based on the generated floor plan.

    Follow these guidelines and tool-specific instructions to ensure the plan is accurate and visually appealing.

    **Tool Information**:

    - **Budget Calculator Tool**:
      >>>
      * Provide detailed budget estimation for the house plan, breaking it down into categories like materials, labor, and contingency.
      give the information in a structured way so that we can use this informattion to create plot for the budget
      <<<

    - **Python Environment Tool**:
      >>>
      * Avoid importing libraries manually; assume they are preloaded in the environment.
      * Do not use disk operations for saving or loading data; work with in-memory data only.
      * Do not create plots using Python libraries like matplotlib or plotly. Use the plot_generator tool instead.
      * Use functions like `head`, `tail`, or `sample` to explore large datasets instead of printing the entire dataframe.
      <<<

    - **Plot Generator Tool**:
      >>>
      * Exclusively use this tool to create visualizations.
      * Prefer bar plots for simplicity and clarity.
      * Always provide at least 3-4 relevant plots to illustrate the budget allocation effectively.
      <<<

    - **House Plan Generator Tool**:
      >>>
      * Use natural language prompts to describe the house plan in detail.
      * Ensure the house description aligns with user preferences and budget constraints.
      <<<

    - **3D Model Generator Tool**:
      >>>
      * Provide the generated house plan as an image to create a 3D model.
      * Use local image paths for the house plan generated by the house plan generator tool.
      <<<

    **Workflow**:
    1. Analyze the location and soil details provided by the user.
    2. Generate a detailed budget plan based on the user's input.
    3. Visualize the budget plan with 3-4 plots using the plot_generator tool.
    4. Use the house plan generator tool to create a detailed floor plan description.
    5. Convert the floor plan into a 3D model using the 3D model generator tool.

    Human Input Template:
    >>>
    * Details provided by the user for house data: {input}
    * Generate plan: {plan}
    <<<

    Prompt Template:
    1. Start by analyzing the location and soil details.
    2. Proceed to calculate a detailed budget plan.
    3. Visualize the budget allocation with multiple relevant plots.
    4. Generate a comprehensive house plan with detailed descriptions.
    5. Create a 3D model of the house based on the generated plan.

    Make sure to:
    - Use the tools appropriately as per the instructions.
    - Avoid unnecessary computations or imports.
    - Present the results clearly and concisely.

        """
        human_input = "Details provide by user for house data:{input}\n Generate plan : {plan}"

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system",prompt),
                ("human",clean_paranthesis(human_input)),
                ("placeholder","{agent_scratchpad}")
            ]
        )
        py_tool = get_python_lc_tool()
        match_tool = get_matching_values_tool(py_tool)
        self.py_tool_name = py_tool.name
        plot_tool = get_plot_gen_lc_tool(
            source_python_tool=py_tool, plot_folder="temp_folder", with_query=False
        )

        self.tools = [get_house_plan_tool(),image_to_3d_creator_tool(),get_budget_calculation_tool(),plot_tool,match_tool,py_tool] 
        self.tool_name2tool = {t.name: t for t in self.tools}  
        self.planner = BedrockClaudeAgentPlanner(
            prompt_template, "anthropic.claude-3-5-sonnet-20240620-v1:0", self.tools
        )
        self.state["raw_intermediate_steps"] = []
        self.state["intermediate_steps"] = []
        plan_prompt = """You are professional architect with 10 years of experience and you task is to generate a plan for the client requirement with help of following tools 
        

    1. **Location and Soil Analysis**: Assess the location, soil type, and environmental factors to provide the foundation for the construction plan.
    2. **Budget Planning**: Based on the user-provided budget, create a detailed budget estimation for the house or construction project in a more structured way.
    3. **Visualization of Budget Plan**: Generate 3-4 clear and informative plots to visualize the budget allocation across different components like foundation, materials, labor, etc.
    4. **Floor Plan Creation**: Based on the budget and user preferences, generate a detailed floor plan description using the house plan generator tool.
    5. **3D Model Generation**: Create a 3D digital twin or model of the house based on the generated floor plan.
        
        
        house_plan_generator:
        >>>
        * Natural language description of the house plain in the positive prompt
        * Provide more description of the house in the positive prompt 
        <<<
        
        3d_model_generator:
        >>>
        * Provide the image of the house image to be converted to 3D model
        * Provide local image path of the house image generated by the house plan generator tool
        <<<
        budget_calculator:
        >>>
        * Provide the detailed  budget estimation for the house plan
        <<<
        
        generate budget first and then house plan and 3d model for the house plan
    
                        """

        user_prompt_client = f"client requirement:{repr(self.data)}\n generate plan for using the tools"
        plan_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system",plan_prompt),
                ("human",clean_paranthesis(user_prompt_client)),
               
            ]
        )
        self.generated_plan = generate_plan(plan_prompt_template,"anthropic.claude-3-haiku-20240307-v1:0")
        
    def prepare_intermediate_steps(self):
        filter_steps =[]
        success_found = False
        reverse_steps : MultiToolAgentStep = self.state["raw_intermediate_steps"][::-1]
        for step in reverse_steps:
            if all([s.status == "success" for s in step.tool_results]):
                success_found = True
            if not success_found:
                filter_steps.append(step)
            else:
                if all([s.status == "success" for s in step.tool_results]):
                    filter_steps.append(step)
                    
        return filter_steps[::-1]
    
    def step_agent(self):
        # print(self.prepare_intermediate_steps())
        # print(self.question)
        result = self.planner.plan(
            self.prepare_intermediate_steps(),
            input=clean_paranthesis(repr(self.data)),
            plan = self.generated_plan
        )
        if isinstance(result, AgentFinish):
            return result
        
        if not isinstance(result,MultiToolAgentAction):
            raise ValueError("MultiToolAgentAction not supported")
        
        

        tool_results = []
        for tool_action in result.tool_actions:
            tool = self.tool_name2tool[tool_action.tool]
            tool_result = tool.invoke(tool_action.tool_input)
            status = (
                "success"
                if not tool_result.get("metadata",{}).get("error")
                else "error"
            )
            observation_content = {"text":tool_result.get("observation")}
            if tool.name == "house_plan_generator":
                # print(tool_result.get("metadata"))
                image_paths = tool_result.get("metadata",{}).get("image_path")
                # print(image_paths)
                observation_content = {
                        "text": f'This the path for generated image : {open(image_paths[0]["image"])}',
                    }
                
            if tool.name == "plot_generator":
                try:
                    with open(tool_result.get("metadata")["image_path"], "rb") as f:
                        image_data = f.read()
                    observation_content = {
                        "image": {"format": "jpeg", "source": {"bytes": image_data}}
                    }
                except KeyError as e:
                    logger.error(f"Error in getting image data: {e}")
                    print(tool_result)

            tool_results.append(
                ToolResult(
                    tool_action=tool_action,
                    content=observation_content,
                    metadata=tool_result.get("metadata"),
                    status=status
                )
            )
        agent_step = MultiToolAgentStep(action=result,tool_results=tool_results)
        return agent_step
    def run_agent(self)->Generator:
        
        yield {
            "event":"plan_generation",
            "message":self.generated_plan
        }
        for _ in range(self.max_steps):
            step_output = self.step_agent()
            if isinstance(step_output,AgentFinish):
                yield {
                    "status":"success",
                    "message":step_output.log

                }
                return 
            for message in step_output.action.message_log:
                yield {
                    "status":"processing",
                    "message":message.content
                }

            for tool_result in step_output.tool_results:
                yield {
                    "event": "intermediate_step",
                    "status": "processing",
                    "tool": tool_result.tool_action.tool,
                    "tool_input": tool_result.tool_action.tool_input,
                    "step_response": tool_result.content,
                    "message_type": "tool_output",
                    "metadata": tool_result.metadata,
                }
            self.state["raw_intermediate_steps"].append(step_output)
        else:
            yield {
                "status":"error",
                "message":"Max steps reached"
            }   
            