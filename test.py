from agent import PlanAgent


def test_plan_agent():
    data = {
  "square_ft": 4500.0,
  "location": "Urban",
  "estimated_budget": 2500000.0,
  "demographics": "High-income residential area with young professionals and families",
  "no_of_bks": 5,
  "stories": 3,
  "main_road": "Yes",
  "guest_rooms": "Yes",
  "basements": "No",
  "parking": "Nearby",
  "quality_type": "Premium"
}
    agent = PlanAgent(data=data)
    agent.prepare_agent()
    for step in agent.run_agent():
        print(step)


test_plan_agent()