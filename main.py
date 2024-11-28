import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import json

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to extract numbered points from response
def extract_numbered_points(text):
    try:
        # Find the JSON array within the text (between ``` markers)
        json_start = text.find('```json\n[')
        json_end = text.find(']\n```')
        if json_start != -1 and json_end != -1:
            json_str = text[json_start + 7:json_end + 1]

            # Parse JSON array into list
            return json.loads(json_str)
        else:
            return [text.strip()]
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return [f"Malformed JSON: {e}"]
    except Exception as e:
        print(f"Unexpected error: {e}")
        return [f"Unexpected error: {e}"]


# Function to generate prompts using LLM
def generate_prompts(model, content_type, audience_type, delivery_method, content_theme, target_industry):
    try:
        # Start chat and send a message
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        f"""Generate 4 content creation prompts to help trainers generate content based on the following inputs: 
                        \nContent Type - {content_type}\nAudience Type - {audience_type}\nDelivery Method - {delivery_method}
                        \nContent Theme - {content_theme}\nTarget Industry - {target_industry}
                        \nPlease format your response as a JSON array.""",
                    ],
                }
            ]
        )
        response = chat_session.send_message("Generate the content creation prompts.")

        # Extract prompts directly from the response text
        prompts = extract_numbered_points(response.text)
        if not prompts:
            return ["Error extracting prompts. Please check the API response format."]

        return prompts

    except Exception as e:
        print(f"Error generating prompts: {e}")
        return [f"Error: {e}"]


# Streamlit UI
def main():

    # Sidebar for adjusting LLM config
    st.sidebar.title("Adjust Content Generation Parameters")

    temperature = st.sidebar.slider(
        "Temperature (Controls randomness)",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1
    )
    top_p = st.sidebar.slider(
        "Top P (Controls nucleus sampling)",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01
    )
    top_k = st.sidebar.number_input(
        "Top K (Limits the vocabulary for predictions)",
        min_value=1,
        max_value=100,
        value=40,
        step=1
    )
    max_output_tokens = st.sidebar.number_input(
        "Max Output Tokens (Maximum length of generated contet)",
        min_value=1,
        max_value=8192,
        value=8192,
        step=1
    )

    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        generation_config=generation_config,
        system_instruction="You are expert at prompt engineering and your goal is to write prompts helping the trainers to create professional and relevant content. The user will provide you the domain, the sub domain, the target audience and the target industry as the input. Your objective is to write clear, top class prompts based on the given input which will help the user create high quality training content. Please note your users will mostly be from the training and consulting industry.",
    )

    st.title("LLM-Powered Content Generator")

    # Dropdown menus for user input
    st.subheader("Input Filters")

    content_type = st.selectbox(
        "Content Type",
        [
            "Training Modules",
            "E-Learning Courses",
            "Case Studies/ caselets",
            "Role Plays",
            "Interactive Quizzes",
            "Assessments",
            "Videos/Animations",
            "Infographics",
            "Worksheets/Job Aids",
            "Proposals",
            "Content Outline",
            "Feedback Templates",
            "Questionnaires - Self Assessments/ Reflections/ research etc",
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

    # Submit button to generate prompts
    if st.button("Submit"):
        with st.spinner("Generating prompts..."):
            prompts = generate_prompts(
                model, content_type, audience_type, delivery_method, content_theme, target_industry
            )

        # Display generated prompts
        if prompts:
            st.subheader("Generated Prompts")

            # Initialize session state to track which prompt is clicked
            if "clicked_prompt" not in st.session_state:
                st.session_state.clicked_prompt = None

            for idx, prompt in enumerate(prompts, start=1):
                # Check if `prompt` is a dictionary or string
                if isinstance(prompt, dict):
                    content = prompt.get('prompt', 'No content available')
                    notes = prompt.get('notes', '')
                else:
                    content = prompt
                    notes = ''

                # Create the clickable box by embedding an anchor link in the div
                box_id = f"prompt_{idx}"
                clickable_content = f"""
                <div id="{box_id}" style="border: 1px solid white; border-radius: 8px; padding: 15px; margin-bottom: 10px; background-color: transparent; cursor: pointer; box-shadow: none; transition: background-color 0.3s;">
                    <strong>Prompt {idx}:</strong>
                    <p>{content}</p>
                    <p><i>{notes}</i></p>
                </div>
                <style>
                    #prompt_{idx}:hover {{
                        background-color: #333333; /* Slightly lighter than black */
                    }}
                </style>
                """

                # Use st.markdown to render the HTML and make the div clickable
                st.markdown(clickable_content, unsafe_allow_html=True)

                # If the prompt box is clicked, store its ID in session_state
                if st.session_state.clicked_prompt == box_id:
                    st.markdown(
                        f"""
                        <div style="border: 1px solid white; border-radius: 8px; padding: 15px; margin-bottom: 10px; background-color: transparent; box-shadow: none;">
                            <strong>Prompt {idx}:</strong>
                            <p>{content}</p>
                            <p><i>{notes}</i></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Logic to handle click event (simulate click detection)
            # If a prompt box is clicked, store the box ID to show its content
            for idx in range(1, len(prompts) + 1):
                box_id = f"prompt_{idx}"
                if st.session_state.clicked_prompt is None:
                    st.session_state.clicked_prompt = box_id
                # Detect click event on each box and update session_state
                if st.session_state.clicked_prompt == box_id:
                    st.session_state.clicked_prompt = None  # Reset after click for showing it
        else:
            st.error("No prompts were generated. Please try again.")


if __name__ == "__main__":
    main()
