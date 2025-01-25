import streamlit as st
import pandas as pd
from tools.budget_calculation import get_budget_calculation_tool, BudgetInputs
from tools.search_tool import get_geo_loc_info_tool
from langchain.agents import AgentExecutor, initialize_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from tools.budget_calculation import BudgetCalculation
from tools.search_tool import DisasterDevelopmentTool
import os 

load_dotenv()

def main():
    st.title('House Construction Budget Planner')
    
    # Input sections
    col1, col2, col3,col4,col5 = st.columns(5)
    
    with col1:
        square_ft = st.number_input('Total Square Footage', min_value=500, max_value=10000, value=2000,step=100)
        location = st.selectbox('Location Type', ['Urban', 'Suburban', 'Rural'])
    
    with col2:
        estimated_budget = st.number_input('Estimated Budget ($)', min_value=50000, max_value=5000000, value=300000,step=1000)
        demographics = st.text_input('Demographic Location')

    with col3:
        no_of_bks = st.number_input('No of BHK', min_value=1, max_value=10, value=1)
        stories = st.number_input('Stories',min_value=1, max_value=10, value=1)
    
    with col4:
        main_road = st.selectbox('Main Road', ['Yes', 'No', 'Nearby'])
        guest_rooms = st.selectbox('Guest Room',['Yes', 'No'])

    with col5:
        basements = st.selectbox('Basements', ['Yes', 'No'])
        parking = st.number_input('Parking',min_value=1, max_value=5, value=1)


    quality_type = st.selectbox('Construction Quality', ['Basic', 'Standard', 'Premium', 'Luxury'])
    
    # Create BudgetInputs object
    budgetary_inputs = {
        "square_ft":square_ft,
        "location":location,
        "estimated_budget":estimated_budget,
        "demographics":demographics,
        "no_of_bks":no_of_bks,
        "stories":stories,
        "main_road":main_road,
        "guest_rooms":guest_rooms,
        "basements":basements,
        "parking":parking,
        "quality_type":quality_type
    }
    
# #Budget calculation
#     if st.button('Calculate Budget Breakdown'):
#         budgetCalculation = BudgetCalculation()
#         results = budgetCalculation.calculate_budget_estimates(budgetary_inputs)
#         st.write(results)

    # if st.button('Get Geo Location Info'):
    #     geo_loc = DisasterDevelopmentTool()
    #     results = geo_loc.generate_comprehensive_report(budgetary_inputs.get('demographics'))
    #     st.write(results)    

if __name__ == '__main__':
    main()
    