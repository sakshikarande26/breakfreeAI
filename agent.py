from dotenv import load_dotenv
import os
from phi.agent import Agent
from phi.model.groq import Groq
import streamlit as st


load_dotenv()
api_key = os.getenv('GROQ_API_KEY')


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


    
def generate_prompts(content_type, audience_type, delivery_method, content_theme, target_industry):
    prompt_agent = Agent(
        name="Prompt Generation Agent",
        model=Groq(id="llama-3.3-70b-versatile"),
        markdown=True,
        instructions=f"Generate 4 distinct content creation prompts to help trainers who train employees generate content based on the following user inputs: Content Type - {content_type} Audience Type - {audience_type} Delivery Method - {delivery_method} Content Theme - {content_theme} Target Industry - {target_industry}. For each prompt, provide a A detailed version for content generation without a title.",
        description="You are an AI agent specializing in helping trainers develop tailored content for employee training sessions. By inputting specific parameters such as content type, audience type, delivery method, content theme, and target industry, you generate creative prompts that inspire engaging and effective training materials. These prompts can later be used to provide detailed content generation."
    )

    response = prompt_agent.run("generate 4 distinct prompts that can be used to generate relevant detailed content. without title ")
    return response.content.split("\n")

def summarize_prompts(prompts, content_type, audience_type, delivery_method, content_theme, target_industry):
    summary_agent = Agent(
        name="Prompt Summary Generation Agent",
        model=Groq(id="llama-3.3-70b-versatile"),
        markdown=True,
        instructions=f"Generate a 4 individual summaries of 2-3 sentences for the each of following prompts while maintaining the context. The user selections are: Content Type - {content_type}, Audience Type - {audience_type}, Delivery Method - {delivery_method}, Content Theme - {content_theme}, Target Industry - {target_industry}. Prompts: {', '.join(prompts)}",
        description="You are an agent that summarizes prompts while preserving the context and incorporating user selections."
    )

    response = summary_agent.run("Generate a concise summary for each of the provided prompts and give heading as summary (1/2/3/4): (generated summaries)")
    return response.content.split("\n")

def generate_content(prompt):
    content_agent = Agent(
        name="Content Generation Agent",
        model=Groq(id="llama-3.3-70b-versatile"),
        markdown=True,
        instructions="Generate detailed content based on the following prompt: {prompts}",
        description="You are an agent that generates comprehensive and structured training content based on the provided detailed prompt."
    )

    response = content_agent.run(f"Generate detailed content based on the following prompt: {prompt}")
    return response.content.split("\n")
    

if __name__ == "__main__":
    main()




