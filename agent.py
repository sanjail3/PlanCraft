from tools.addition import get_add_tool
from tools.HousePlantool import get_house_plan_tool
from tools.budget_calculation import get_budget_calculation_tool
from tools.tool3d import image_to_3d_creator_tool
from langchain.prompts import ChatPromptTemplate
from typing import Generator
from planner import MultiToolAgentStep, BedrockClaudeAgentPlanner, AgentFinish, MultiToolAgentAction,ToolResult
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from data import data
import json
from utils import clean_paranthesis
def generate_plan(prompt,model_name):
    chain = prompt|ChatBedrock(model_id=model_name)
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
    def __init__(self, data,max_steps=5):
        self.state = {}
        self.data = data
        self.max_steps = max_steps
    def prepare_agent(self):
        prompt = """
        You are professional architect and you have been tasked with designing a house plan, you should provide a detailed description of the house plan you want to design and also provide the cost estimation for the house plan,
        
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


        generate budget only for the house plan

        """
        human_input = "Details provide by user for house data:{input}\n Generate plan : {plan}"

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system",prompt),
                ("human",clean_paranthesis(human_input)),
                ("placeholder","{agent_scratchpad}")
            ]
        )
        self.tools = [get_house_plan_tool(),image_to_3d_creator_tool(),get_budget_calculation_tool()] 
        self.tool_name2tool = {t.name: t for t in self.tools}  
        self.planner = BedrockClaudeAgentPlanner(
            prompt_template, "anthropic.claude-3-5-sonnet-20240620-v1:0", self.tools
        )
        self.state["raw_intermediate_steps"] = []
        self.state["intermediate_steps"] = []
        plan_prompt = """You are professional architect with 10 years of experience and you task is to generate a plan for the client requirement with help of following tools 
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
        if len(reverse_steps)>0:
            if any([True for s in reverse_steps[0].tool_results if s.tool_action.tool == "house_plan_generator"]):
                if data.get("is_plane_approved") == "approved":
                    pass
                else:
                    
                    for tool_result in reverse_steps[0].tool_results:
                        if tool_result.tool_action.tool == "house_plan_generator":
                            tool_result.content["text"] += f'\n\n The generated plan is rejected by the user with feedback : {data.get("message")}'
                    data["is_plane_generated"] = False
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
                data["is_plane_generated"] = True

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
            