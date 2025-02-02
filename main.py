import streamlit as st
from promptAPI.server1 import generate_prompts, summarize_prompts
import requests

# FastAPI server URL
API_URL = "http://127.0.0.1:8000"


def main():
    st.title("Prompt Generation Agent")

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

    if st.button("Generate Prompts"):
        prompts = generate_prompts(content_type, audience_type, delivery_method, content_theme, target_industry)
        
        summaries = summarize_prompts(prompts, content_type, audience_type, delivery_method, content_theme, target_industry)

        for i, summary in enumerate(summaries, 1):
            with st.container():
                st.write(summary)

if __name__ == "__main__":
    main()
