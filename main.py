import json
import os
import google.generativeai as genai
#from google.ai.generativelanguage_v1beta.types import content
import streamlit as st
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def extract_numbered_points(text):
    try:
        # Fin8d the JSON array within the text (between ``` markers)
        print(text)
        json_start = text.find('```json\n[')
        json_end = text.find(']\n```')
        print("JSON start:", json_start)
        print("JSON end:", json_end)
        if json_start != -1 and json_end != -1:
            json_str = text[json_start + 7:json_end + 1]
            # Parse JSON array into list
            prompts_dict = json.loads(json_str)
            return prompts_dict
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []

# Function to generate prompts using LLM
def generate_prompts(model, content_type, audience_type, delivery_method, content_theme, target_industry):
    try:
        chat_session = model.start_chat(
        history=[
            {
            "role": "user",
            "parts": [
                f"""Generate 4 content creation prompts to help trainers generate content based on the following inputs: \nContent Type - {content_type}\nAudience Type - {audience_type}\nDelivery Method of the content - {delivery_method}\nContent Theme - {content_theme}\nTarget Industry - {target_industry}\n\nPlease note - The prompts should be designed such that content relevant to the above given Content Type, Audience Type, Delivery Method, Content Theme and Target Industry can be generated by trainers. 

                Please format your response as a JSON array, where each element is a prompt string. 

                Here's an example of the desired JSON format:
                [
                "Prompt 1:        ",
                "Prompt 2: ",
                "Prompt 3: ",
                "Prompt 4: "
                ]""",
            ],
            },
        ]
        )

        response = chat_session.send_message(f"""Generate 4 content creation prompts for:
                                            Content Type: {content_type}
                                            Content Theme: {content_theme}
                                            Target audience: {audience_type}
                                            Delivery Method: {delivery_method}
                                            Target industry: {target_industry}""")
        
        #print("Response text:", response.text)
        #st.subheader("Generated Prompts")
        #st.write(response.text)
        
        # Assuming the response is a JSON string containing a list of prompts
        # Extract prompts from JSON response
        prompts = extract_numbered_points(response.text)
        print("Extracted prompts:", prompts)

        return [p for p in prompts if p]
    
    except Exception as e:
        print(f"Error generating prompts: {e}")
        return []
    
# Streamlit UI
def main():

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
            prompts = generate_prompts(model, content_type, audience_type, delivery_method, content_theme, target_industry)
            print("Debug - Full prompts list:", prompts)  # Debug print
            print("Debug - Length of prompts:", len(prompts))  # Debug print
            
            if prompts and len(prompts) >= 2:  # Check if we have at least 2 elements
                print(prompts[1])
            #else:
                #st.error(f"Not enough prompts generated. Only got {len(prompts)} prompts.")

        if prompts:
            st.subheader("Generated Prompts")
            selected_prompt = st.radio(
                "Select a prompt to send to ChatGPT:",
                options=prompts,
                format_func=lambda x: x if len(x) <= 100 else x[:97] + "..."
            )

            # Option to send selected prompt to ChatGPT API
            if st.button("Send Prompt to ChatGPT"):
                with st.spinner("Sending prompt to ChatGPT..."):
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": selected_prompt}
                            ],
                            temperature=0.7,
                            max_tokens=200
                        )
                        st.success("Response from ChatGPT:")
                        st.write(response['choices'][0]['message']['content'])
                    except Exception as e:
                        st.error(f"Error communicating with ChatGPT API: {e}")

if __name__ == "__main__":
    main()
