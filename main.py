import os
import json
import streamlit as st
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Function to initialize the LLM
def initialize_llm(temperature, top_p):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=temperature,
        model_kwargs={"top_p": top_p / 100},  # Convert top_p to a float between 0 and 1
        max_retries=2,
    )

# Function to create a prompt template
def create_prompt_template(is_json_format=True):
    return """ 
    You are an AI agent that helps trainers create tailored content for employee training sessions.
    Generate 4 distinct content creation prompts for trainers based on the following user inputs:
    - Content Type: {content_type}
    - Audience Type: {audience_type}
    - Delivery Method: {delivery_method}
    - Content Theme: {content_theme}
    - Target Industry: {target_industry}

    For each prompt, provide:
    1. A detailed prompt (without a title) that explains the type of content to create.
    2. A short 2-3 sentence version summarizing the above prompt.

    ### Return the output as a JSON object:
    ```json
    {{
      "prompts": [
        {{
          "prompt1": "Detailed prompt 1 here",
          "summary1": "Summary for prompt 1 here"
        }},
        {{
          "prompt2": "Detailed prompt 2 here",
          "summary2": "Summary for prompt 2 here"
        }},
        {{
          "prompt3": "Detailed prompt 3 here",
          "summary3": "Summary for prompt 3 here"
        }},
        {{
          "prompt4": "Detailed prompt 4 here",
          "summary4": "Summary for prompt 4 here"
        }}
      ]
    }}
    """

# Function to generate initial prompts
def generate_initial_prompts(llm, content_type, audience_type, delivery_method, content_theme, target_industry):
    template = create_prompt_template(is_json_format=True)
    prompt_template = PromptTemplate(
        input_variables=["content_type", "audience_type", "delivery_method", "content_theme", "target_industry"],
        template=template,
    )
    output_parser = JsonOutputParser()
    chain = LLMChain(llm=llm, prompt=prompt_template, output_parser=output_parser)
    
    # Prepare inputs dictionary
    inputs = {
        "content_type": content_type,
        "audience_type": audience_type,
        "delivery_method": delivery_method,
        "content_theme": content_theme,
        "target_industry": target_industry
    }
    
    response = chain.run(inputs)
    return json.dumps(response, indent=2)


# Function to generate content based on selected prompt
def generate_content_from_prompt(llm, selected_prompt):
    content_generation_template = """ 
    Based on the following training content prompt, generate a detailed training content guide:
    
    Prompt: {selected_prompt}

    The content should be well-structured, detailed, and useful for trainers.
    """
    prompt_template = PromptTemplate(input_variables=["selected_prompt"], template=content_generation_template)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    response = chain.run({"selected_prompt": selected_prompt})
    return response


def chat_assistant(llm):
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    if "chat_active" not in st.session_state:
        st.session_state.chat_active = False  # Track chat state

    st.write("### Chat Assistant")

    if not st.session_state.chat_active:
        start_chat = st.button("Start Chat Assistant", key="start_chat_button")
        if start_chat:
            st.session_state.chat_active = True

    if st.session_state.chat_active:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for chat in st.session_state.chat_history:
            st.write(f"**You:** {chat['user']}")
            st.write(f"**Assistant:** {chat['assistant']}")

        user_input = st.text_input("You: ", key="user_input")

        if user_input:
            if user_input.lower() == "exit":
                st.session_state.chat_active = False
                st.experimental_rerun()

            if user_input.lower() == "clear memory":
                st.session_state.conversation_memory.clear()
                st.session_state.chat_history = []
                st.write("🧹 Conversation memory has been cleared.")
                st.experimental_rerun()

            # Define Chat Prompt with context from generated content
            context = st.session_state.get("generated_content", "")
            chat_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful AI assistant working in the training industry. Your responsibility is to refine and modify training content based on user feedback."),
                MessagesPlaceholder(variable_name="chat_history"),
                AIMessage(content=f"Provide modifications or refinements to the generated training content. Previous content: {context}")
            ])

            chain = LLMChain(llm=llm, prompt=chat_prompt, memory=st.session_state.conversation_memory)

            try:
                response = chain.predict(input=user_input)

                # Store conversation in session state
                st.session_state.chat_history.append({"user": user_input, "assistant": response})

                # Save context for multi-turn conversation
                st.session_state.conversation_memory.save_context(
                    {"input": user_input},
                    {"output": response}
                )

                # Display the assistant's response
                st.write(f"**Assistant:** {response}")

            except Exception as e:
                st.write(f"Error: {str(e)}")


def main():
    # Sidebar for adjusting LLM config
    st.sidebar.title("Content Generation Parameters")

    temperature = st.sidebar.slider(
        "Temperature (Controls randomness)",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01
    )
    top_p = st.sidebar.slider(
        "Top P (Controls nucleus sampling)",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01
    )

    # Input Filters
    st.title("LLM-Powered Content Generator")
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

    if "initial_response" not in st.session_state:
        st.session_state.initial_response = None

    if st.button("Generate Prompts"):
        llm = initialize_llm(temperature, top_p)
        st.session_state.initial_response = generate_initial_prompts(
            llm, content_type, audience_type, delivery_method, content_theme, target_industry
        )

    if st.session_state.initial_response:
        st.write("### Generated Training Content:")
        prompts = json.loads(st.session_state.initial_response)["prompts"]

        selected_prompt_index = st.selectbox(
            "Select a prompt:",
            [f"Prompt {i+1}: {prompt[f'summary{i+1}']}" for i, prompt in enumerate(prompts)]
        )

        selected_index = int(selected_prompt_index.split(":")[0].split(" ")[1]) - 1
        selected_prompt = prompts[selected_index][f"prompt{selected_index + 1}"]

        st.write("### Selected Prompt:")
        st.write(selected_prompt)

        if st.button("Generate Content from Prompt"):
            llm = initialize_llm(temperature, top_p)
            st.session_state.generated_content = generate_content_from_prompt(llm, selected_prompt)

        if "generated_content" in st.session_state:
            st.write("### Generated Training Content:")
            st.write(st.session_state.generated_content)

            if st.button("Start Chat Assistant"):
                chat_assistant(initialize_llm(temperature, top_p))


if __name__ == "__main__":
    main()
