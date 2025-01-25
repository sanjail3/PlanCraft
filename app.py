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

    st.title("🏠 Architect AI Assistant")
    st.markdown("Transform your architectural visions into detailed plans and 3D models!")

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
        combined_prompt = f"{prompt}\n\n{picture_description}"
        st.session_state.messages.append({"role": "user", "content": combined_prompt})
        
        with st.chat_message("user"):
            st.markdown(combined_prompt)

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
                        
                        if tool_name == "house_plan_generator":
                            status_text.markdown("🎨 *Generating floor plans...*")
                            st.toast("Floor Plans Generated", icon="🎨")
                            st.markdown("*Floor Plans:*")
                            with st.expander(event["tool"],expanded=True):
                                if not metadata.get("error"):
                                    collected_artifacts["images"] = metadata.get("image_path", [])
                                    for img in metadata.get("image_path", []):
                                        from pathlib import Path

                                        # Convert to Path object for better path handling
                                        image_path = Path(img["image"])
                                        st.image(str(image_path), caption="Generated Floor Plan", width=400)
                                        pass
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
                                if metadata.get("thumbnail"):
                                    st.image(f'r{metadata["thumbnail"]}', caption="3D Model Preview", width=300)
                                    pass
                                for url in metadata.get("model_urls", []):
                                    st.markdown(f"[Download 3D Model]({url})")
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