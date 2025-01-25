import streamlit as st
from agent import PlanAgent
from typing import Dict, Any
from pathlib import Path
from data import data

def render_message_content(message: Dict[str, Any]) -> None:
    """Render different types of message content with appropriate formatting."""
    content = message.get("content", "")
    metadata = message.get("metadata", {})
    
    if content:
        st.markdown(content)
    
    if "images" in metadata:
        st.markdown("**Generated House Plans:**")
        cols = st.columns(2)
        for idx, img_path in enumerate(metadata["images"]):
            cols[idx % 2].image(img_path, use_column_width=True)
    
    if "model_urls" in metadata:
        st.markdown("**3D Model Resources:**")
        for url in metadata["model_urls"]:
            st.markdown(f"- [3D Model Download]({url})")
    
    if "thumbnail" in metadata and metadata["thumbnail"]:
        st.image(metadata["thumbnail"], caption="3D Model Preview", width=300)
    
    if "errors" in metadata:
        for error in metadata["errors"]:
            st.error(f"**Error:** {error}")

def parse_user_prompt(prompt):
    """Parse user prompt to extract house details."""
    return {
        "square_ft": None,
        "location": None,
        "estimated_budget": None,
        "demographics": None,
        "no_of_bks": 3,
        "stories": 2,
        "main_road": True,
        "guest_rooms": 1,
        "basements": 0,
        "parking": 2,
        "quality_type": "standard",
        "description": prompt
    }

def main():
    st.set_page_config(page_title="Architect AI", page_icon="🏠", layout="wide")
    
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
    session_defaults = {
        "messages": [],
        "processing": False,
        "current_step": None,
        "generated_plan": None,
        "plan_approval": None,
        "plan_feedback": None,
        "agent": None
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_message_content(message)

    # Handle new user input
    if prompt := st.chat_input("Describe your dream house..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        house_data = parse_user_prompt(prompt)
        st.session_state.agent = PlanAgent(house_data)
        st.session_state.current_step = "plan_generation"
        st.session_state.processing = True
        st.rerun()

    # Handle workflow processing
    if st.session_state.processing and st.session_state.agent:
        with st.chat_message("ai"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            collected_artifacts = {"images": [], "model_urls": [], "errors": []}
            
            try:
                # Plan Generation Step
                if st.session_state.current_step == "plan_generation":
                    progress_bar.progress(20)
                    status_text.markdown("📝 **Generating initial plan...**")
                    
                    plan_event = next(st.session_state.agent.run_agent())
                    st.session_state.generated_plan = plan_event
                    
                    with st.expander("⚙️ Generated Plan"):
                        st.markdown(plan_event["message"])
                        if "images" in plan_event.get("metadata", {}):
                            for img in plan_event["metadata"]["images"]:
                                st.image(img, caption="Floor Plan", use_column_width=True)
                    
                    st.session_state.current_step = "awaiting_approval"
                    st.rerun()

                # Approval Handling Step
                elif st.session_state.current_step == "awaiting_approval":
                    progress_bar.progress(40)
                    status_text.markdown("⏳ **Waiting for your approval...**")
                    
                    st.markdown("### Generated Floor Plans")
                    if st.session_state.generated_plan and "images" in st.session_state.generated_plan.get("metadata", {}):
                        cols = st.columns(2)
                        for idx, img in enumerate(st.session_state.generated_plan["metadata"]["images"]):
                            cols[idx % 2].image(img, use_column_width=True)
                    
                    # Approval Interface
                    if st.session_state.plan_approval is None:
                        st.radio(
                            "Does this plan meet your requirements?",
                            ["Yes", "No"],
                            key="plan_approval_input",
                            index=None
                        )
                        if st.button("Submit Approval Decision"):
                            st.session_state.plan_approval = st.session_state.plan_approval_input
                            st.rerun()
                    else:
                        if st.session_state.plan_approval == "No":
                            if st.session_state.plan_feedback is None:
                                st.text_input(
                                    "What changes would you like to see?",
                                    key="plan_feedback_input"
                                )
                                if st.button("Submit Feedback"):
                                    st.session_state.plan_feedback = st.session_state.plan_feedback_input
                                    st.session_state.agent.house_data["feedback"] = st.session_state.plan_feedback
                                    st.session_state.current_step = "plan_generation"
                                    st.session_state.plan_approval = None
                                    st.session_state.plan_feedback = None
                                    st.rerun()
                            else:
                                st.session_state.current_step = "plan_generation"
                                st.rerun()
                        else:
                            st.session_state.current_step = "model_generation"
                            st.rerun()

                # 3D Model Generation Step
                elif st.session_state.current_step == "model_generation":
                    progress_bar.progress(60)
                    status_text.markdown("🛠️ **Generating 3D model...**")
                    
                    model_event = next(st.session_state.agent.run_agent())
                    collected_artifacts["model_urls"] = model_event.get("metadata", {}).get("model_urls", [])
                    collected_artifacts["thumbnail"] = model_event.get("metadata", {}).get("thumbnail", "")
                    
                    if collected_artifacts["thumbnail"]:
                        st.image(collected_artifacts["thumbnail"], caption="3D Model Preview", width=300)
                    
                    st.session_state.messages.append({
                        "role": "ai",
                        "content": "Here's your final 3D model and plans:",
                        "metadata": collected_artifacts
                    })
                    
                    progress_bar.progress(100)
                    status_text.markdown("✅ **Generation complete!**")
                    st.session_state.processing = False
                    st.session_state.current_step = None
                    st.rerun()

            except StopIteration:
                st.session_state.processing = False
                st.session_state.current_step = None
                progress_bar.empty()
                status_text.empty()
            
            except Exception as e:
                st.error(f"⚠️ **Error:** {str(e)}")
                collected_artifacts["errors"].append(str(e))
                st.session_state.messages.append({
                    "role": "ai",
                    "content": "An error occurred during processing",
                    "metadata": collected_artifacts
                })
                st.session_state.processing = False
                progress_bar.empty()
                status_text.empty()

if __name__ == "__main__":
    main()