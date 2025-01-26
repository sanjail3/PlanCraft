import streamlit as st
from agent import PlanAgent
from typing import Dict, Any
from PIL import Image
from io import BytesIO
import os
from langchain_openai import AzureChatOpenAI
import base64
from langchain_core.messages import HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

system_prompt = """You are a viewer agent assigned to extract and format all information from images provided by a constructing planning agent. Strictly follow these guidelines:

Infographics (Charts, Figures, Graphs):

    Extract all data and present it in a clear table format.
    If specific values are missing, indicate ranges without estimating based on visual proportions. 
Tables:

    Recreate the content in an exact table format.
Flow Diagrams:

    Break down and present the process in step-by-step sequence flow.
    Avoid adding extra information to the steps.
Logos or Icons:
    Provide a concise description in 3 to 5 words only.
Random Images (e.g., crowds, people):

Extract no content; return an empty string.
 """

def get_image_data(image)->str:

    llm =  AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY",),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")
        )
    base64_data = base64.b64encode(image.getvalue()).decode("utf-8")

    human_message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"},
            },
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            human_message,
        ],
    )

    chain = prompt | llm

    response = chain.invoke({})
    return response.content

def render_message_content(message: Dict[str, Any]) -> None:
    """Render different types of message content with appropriate formatting."""
    content = message.get("content", "")
    metadata = message.get("metadata", {})
    
    # Render text content
    if content:
        st.markdown(content)
    
    # Render images
    if "images" in metadata:
        st.markdown("*Generated House Plans:*")
        cols = st.columns(2)
        for idx, img_path in enumerate(metadata["images"]):
            cols[idx % 2].image(img_path, use_column_width=True)
    
    # Render 3D models
    if "model_urls" in metadata:
        st.markdown("*3D Model Resources:*")
        for url in metadata["model_urls"]:
            st.markdown(f"- [3D Model Download]({url})")
    
    # Render thumbnail if available
    if "thumbnail" in metadata and metadata["thumbnail"]:
        st.image(metadata["thumbnail"], caption="3D Model Preview", width=300)
    
    # Show errors if any
    if "errors" in metadata:
        for error in metadata["errors"]:
            st.error(f"*Error:* {error}")

def main():
    st.set_page_config(page_title="Architect AI", page_icon="🏠", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .stChatMessage { padding: 1.5rem; border-radius: 10px; }
        .stProgress > div > div > div { background-color: #4CAF50; }
        .error-message { color: #ff4444; padding: 10px; border-radius: 5px; }
        .model-link { color: #2196F3; text-decoration: none; }
        .model-link:hover { text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True)


    with st.sidebar:
        st.title('🏠 Project Parameters')
        st.markdown("Configure your home specifications")
        
        with st.container():
            st.subheader("Basic Details")
            square_ft = st.number_input('Total Square Footage', min_value=500, max_value=10000,value=2000, step=100)
            estimated_budget = st.number_input('Estimated Budget ($)', min_value=50000, max_value=5000000, value=300000,step=1000)
            location = st.selectbox('Location Type', ['Urban', 'Suburban', 'Rural'])
            quality_type = st.selectbox('Construction Quality', ['Basic', 'Standard', 'Premium', 'Luxury'])
            
        with st.container():
            st.subheader("Layout Details")
            col1, col2 = st.columns(2)
            with col1:
                no_of_bhk = st.number_input('BHK', min_value=1, max_value=10, value=3)
            with col2:
                stories = st.number_input('Stories', min_value=1, max_value=10, value=2)
            
            parking = st.number_input('Parking Spaces', min_value=0, max_value=5, value=2)
            
        with st.container():
            st.subheader("Additional Features")
            main_road = st.selectbox('Main Road Access', ['Direct', 'Nearby', 'No'])
            guest_rooms = st.selectbox('Guest Room', ['Yes', 'No'])
            basements = st.selectbox('Basement', ['Yes', 'No'])

        # Store in session state
        st.session_state.budgetary_inputs = {
            "square_ft": square_ft,
            "location": location,
            "quality_type": quality_type,
            "no_of_bhk": no_of_bhk,
            "stories": stories,
            "parking": parking,
            "main_road": main_road,
            "guest_rooms": guest_rooms,
            "basements": basements,
            "estimated_budget": estimated_budget
        }

    st.markdown("""
    <div style='text-align: center; margin-bottom: 3rem;'>
        <div style='
            width: 80px;
            height: 80px;
            margin: 0 auto 1.5rem;
            background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
            border-radius: 16px;
            transform: rotate(45deg);
            display: flex;
            align-items: center;
            justify-content: center;
        '>
            <span style='transform: rotate(-45deg); font-size: 32px;'>🚀</span>
        </div>
        <h1 style='
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #8B5CF6 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        '>
            Build Your Dream Home
        </h1>
        <p style='color: #6B7280; font-size: 1.1rem;'>
            Let's bring your architectural vision to life with AI-powered planning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_message_content(message)

    # File upload handler
    uploaded_file = st.file_uploader("Upload a house plan image (JPEG, JPG, PNG)", type=["jpeg", "jpg", "png"])
    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded House Plan", width=300)
        
        picture_description = get_image_data(uploaded_file)
        st.session_state.messages.append({
            "role": "user",
            "content": "Uploaded an image.",
            "metadata": {"images": [uploaded_file]}
        })
    else:
        picture_description = ""

    if prompt := st.chat_input("Describe your dream house..."):
        st.session_state.processing = True
        combined_prompt = f"{prompt}\n\n{picture_description}\n\nHouse description: {str(st.session_state.budgetary_inputs)}"
        st.session_state.messages.append({"role": "user", "content": combined_prompt})
        
        # with st.chat_message("user"):
        #     st.markdown(combined_prompt)

        with st.chat_message("ai"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            collected_artifacts = {"images": [], "model_urls": [], "errors": []}
            final_response = []

            try:
                agent = PlanAgent(combined_prompt)
                agent.prepare_agent()
                
                for event in agent.run_agent():
                   
                    if event.get("event") == "plan_generation":
                        progress_bar.progress(20)
                        status_text.markdown("📝 *Generating initial plan...*")
                        st.toast("Plan Generated", icon="⚙")
                        with st.expander("⚙ Plan"):
                            st.markdown(event["message"])

                        
                       
                    elif event.get("event") == "intermediate_step":
                        progress_bar.progress(60)
                        tool_name = event.get("tool")
                        metadata = event.get("metadata", {})

                        if tool_name == "budget_calculator":
                            status_text.markdown("Defing budget parameters...*")
                            st.toast("Budget Parameters Defined", icon="⚙")
                            st.markdown("*Budget Parameters:*")
                            st.write(metadata.get("message"," "))
                        
                        if tool_name == "house_plan_generator":
                            status_text.markdown("🎨 *Generating floor plans...*")
                            st.toast("Floor Plans Generated", icon="🎨")
                            st.markdown("*Floor Plans:*")
                            with st.expander(event["tool"], expanded=True):
                                if not metadata.get("error"):
                                    collected_artifacts["images"] = metadata.get("image_path", [])
                                    
                                    # Create a 2x2 column layout
                                    image_paths = metadata.get("image_path", [])
                                    num_images = len(image_paths)
                                    cols = st.columns(2)  # Create two columns

                                    for i, img in enumerate(image_paths):
                                        from pathlib import Path
                                        
                                        # Convert to Path object for better path handling
                                        image_path = Path(img["image"])
                                        
                                        # Determine which column to place the image in
                                        col = cols[i % 2]  # Alternate between the two columns
                                        with col:
                                            st.image(str(image_path), caption=f"Generated Floor Plan {i + 1}", width=400)
                                else:
                                    collected_artifacts["errors"].append("Failed to generate floor plans")

                        elif event["tool"] == "plot_generator":
                                    with st.popover("Plot Intermediate Info"):
                                        st.markdown("### Plot Instruction:")
                                        st.markdown(
                                            event["tool_input"]["plot_instruction"]
                                        )
                                        st.code(event["metadata"].get("plot_gen_code"))
                                    if not event.get("metadata", {}).get("error"):
                                        image_data = event["step_response"]["image"][
                                            "source"
                                        ]["bytes"]
                                        st.image(Image.open(BytesIO(image_data)))
                                        # st.plotly_chart(event["metadata"]["plotly_fig"])
                                    else:
                                        st.caption(
                                            f'```\n{event["step_response"].get("text")}\n```'
                                        )
                        elif tool_name == "3d_model_generator":
                            status_text.markdown("🛠 *Creating 3D model...*")
                            if not metadata.get("error"):
                                collected_artifacts["model_urls"] = metadata.get("model_urls", [])
                                collected_artifacts["thumbnail"] = metadata.get("thumbnail", "")
                                import requests
                                print(collected_artifacts["thumbnail"],"thumbnail")
                                

                                if metadata.get("thumbnail"):
                                    st.image(collected_artifacts["thumbnail"], caption="3D Model Preview", width=300)
                                    pass
                                # for url in metadata.get("model_urls", []):
                                st.markdown(f"[Download 3D Model]({metadata['model_urls']['obj']})")
                            else:
                                collected_artifacts["errors"].append("Failed to generate 3D model")
                    elif event.get("status") == "success":
                        progress_bar.progress(100)
                        status_text.markdown("✅ *Complete!*")
                        final_response.append(event["message"])

               
                st.session_state.messages.append({
                    "role": "ai",
                    "content": "\n".join(final_response),
                    "metadata": collected_artifacts
                })

            except Exception as e:
                st.error(f"⚠ *An error occurred:* {str(e)}")
                collected_artifacts["errors"].append(str(e))
                st.session_state.messages.append({
                    "role": "ai",
                    "content": "Sorry, I encountered an error processing your request.",
                    "metadata": collected_artifacts
                })
            finally:
                st.session_state.processing = False
                progress_bar.empty()
                status_text.empty()

if __name__ == "__main__":
    main()