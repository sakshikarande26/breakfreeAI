import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

API_URL = "http://127.0.0.1:8000"  # FastAPI server URL

load_dotenv()

# Streamlit UI
def main():
    """Main application function."""
    # Initialize session state if not exists
    if "prompts" not in st.session_state:
        st.session_state.prompts = []

    if "responses" not in st.session_state:
        st.session_state.responses = []

    # Sidebar configuration
    st.sidebar.title("Adjust Content Generation Parameters")

    generation_config = {
        "temperature": st.sidebar.slider(
            "Temperature (Controls randomness)",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1
        ),
        "top_p": st.sidebar.slider(
            "Top P (Controls nucleus sampling)",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.01
        ),
        "top_k": st.sidebar.number_input(
            "Top K (Limits the vocabulary for predictions)",
            min_value=1,
            max_value=100,
            value=40,
            step=1
        ),
        "max_output_tokens": st.sidebar.number_input(
            "Max Output Tokens",
            min_value=1,
            max_value=8192,
            value=8192,
            step=1
        ),
        "response_mime_type": "text/plain"
    }

    # Initialize model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        system_instruction="You are expert at prompt engineering and your goal is to write prompts helping the trainers to create professional and relevant content."
    )

    st.title("LLM-based Content Generator")

    # Input fields
    content_type = st.selectbox(
        "Content Type",
        options=[
            "Training Modules",
            "E-Learning Courses",
            "Case Studies/caselets",
            "Role Plays",
            "Interactive Quizzes",
            "Assessments",
            "Videos/Animations",
            "Infographics",
            "Worksheets/Job Aids",
            "Proposals",
            "Content Outline",
            "Feedback Templates",
            "Questionnaires",
            "Simulations",
            "Activities"
        ],
        index=0
    )

    audience_type = st.selectbox(
        "Audience Type",
        [
            "Entry-Level Employees",
            "Mid-Level Professionals",
            "Senior Management",
            "Trainers/Facilitators",
            "Students",
            "Specialized Roles (e.g., Sales, Customer Support)",
            "Client Point of Contact/ LnD Head"
        ],
        index=0
    )

    delivery_method = st.selectbox(
        "Delivery Method",
        [
            "In-Person Training",
            "Virtual Instructor-Led Training (VILT)",
            "Self-Paced Learning",
            "Blended Learning",
            "Microlearning",
            "Outbound Training",
            "Experiential Learning"
        ],
        index=0
    )

    content_theme = st.selectbox(
        "Content Theme",
        [
            "Leadership and Management",
            "Customer Service",
            "Communication Skills",
            "Technical Training",
            "Soft Skills",
            "Industry-Specific Skills",
            "Assessment Center"
        ],
        index=0
    )

    target_industry = st.selectbox(
        "Target Industry",
        [
            "Technology and IT",
            "Healthcare",
            "Finance and Banking",
            "Manufacturing",
            "Retail and E-commerce",
            "Hospitality",
            "Education and Academics"
        ],
        index=0
    )

    # Generate prompts on submit
    if st.button("Submit"):
        with st.spinner("Generating prompts..."):
            generated_prompts = generate_prompts(
                model, content_type, audience_type, delivery_method,
                content_theme, target_industry
            )
            st.session_state.prompts = generated_prompts

    # Display prompts and handle responses
    if st.session_state.prompts:
        st.subheader("Generated Prompts")

        for idx, prompt in enumerate(st.session_state.prompts, start=1):
            if isinstance(prompt, dict):
                detailed_prompt = prompt.get('detailed_prompt', 'No content available')
                summary = prompt.get('summary', 'No summary available')
            else:
                detailed_prompt = str(prompt)
                summary = f"{str(prompt)[:100]}..."

            with st.expander(f"Prompt {idx}: {summary}"):
                st.markdown(f"""
                <div style="border: 1px solid white; border-radius: 8px; padding: 15px; 
                     margin-bottom: 10px; background-color: transparent; box-shadow: none;">
                    <strong>Detailed Prompt:</strong>
                    <p>{detailed_prompt}</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Ask (Prompt {idx})", key=f"ask_button_{idx}"):
                    with st.spinner(f"Asking Gemini with Prompt {idx}..."):
                        response = ask_prompt_to_gemini(detailed_prompt)
                        st.session_state.responses.append({
                            "prompt": detailed_prompt,
                            "summary": summary,
                            "response": response
                        })
                        st.subheader(f"Response to Prompt {idx}")
                        st.write(response)


if __name__ == "__main__":
    main()
