from agent import PlanAgent


def test_plan_agent():
    agent = PlanAgent("Design a personalize house plan for 2BHK")
    agent.prepare_agent()
    for step in agent.run_agent():
        print(step)


test_plan_agent()