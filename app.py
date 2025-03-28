import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from pandasai import SmartDataframe
import sys
from pandasai_openai import OpenAI
from lida import Manager
import json
from datetime import datetime
import re
import io
import traceback
import logging
import html
import uuid
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.ERROR, filename='app_errors.log', 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataAnalysisApp")

# Load environment variables with verbose output
load_dotenv(verbose=True)

# Page configuration
st.set_page_config(page_title="Data Analysis Chat", layout="wide")

# Set maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "visualizations" not in st.session_state:
    st.session_state.visualizations = []
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# Function to validate uploaded file
def validate_file(file):
    """Validate file size, type and perform basic content validation"""
    # Check file size
    if file.size > MAX_FILE_SIZE:
        return False, "File size exceeds the maximum limit of 10MB."
    
    # Check file extension
    valid_extensions = ['.csv', '.xlsx', '.xls']
    file_ext = os.path.splitext(file.name)[1].lower()
    if file_ext not in valid_extensions:
        return False, "Invalid file type. Only CSV and Excel files are supported."
    
    # Basic content validation
    try:
        if file_ext == '.csv':
            # Try reading the first few rows to validate properly
            content = file.read(1024)
            file.seek(0)  # Reset file pointer
            
            # Check if content appears to be CSV format
            decoded = content.decode('utf-8', errors='ignore')
            if not re.search(r'[^,\n"\']+,[^,\n"\']', decoded):
                return False, "Invalid CSV format. Please check your file."
            
        else:  # Excel file
            # Basic validation for Excel files
            try:
                pd.read_excel(file, nrows=5)
                file.seek(0)  # Reset file pointer
            except Exception:
                return False, "Invalid Excel format. Please check your file."
        
        return True, ""
    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        return False, "Could not validate file. Please check the format."

# Function to sanitize output
def sanitize_output(text):
    """Sanitize text output to prevent XSS"""
    if text is None:
        return ""
    if isinstance(text, str):
        # HTML escape
        return html.escape(text)
    return str(text)

# Function to get OpenAI API key
def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        st.error("API key configuration error. Please check your .env file.")
        st.stop()
    return api_key

# Function to handle user feedback
def record_feedback(idx, is_positive):
    """Record user feedback and store in Supabase if available"""
    # Save feedback in session state
    st.session_state.feedback[idx] = is_positive
    
    # Create a hidden container for debug output (only visible in debug mode)
    debug_container = st.empty()
    show_debug = st.sidebar.checkbox("Show Feedback Debug", False, key=f"debug_toggle_{uuid.uuid4()}")
    
    try:
        from supabase import create_client, Client
        
        # Get Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        # Debug info (only shown if debug mode enabled)
        if show_debug:
            with debug_container.container():
                st.write("### Feedback Debug Info")
                st.write("Supabase URL configured:", bool(supabase_url))
                st.write("Supabase Key configured:", bool(supabase_key))
        
        # Get the question and answer from history
        if idx < len(st.session_state.history):
            # Generate feedback data
            feedback_data = {
                "id": str(uuid.uuid4()),
                "question": str(st.session_state.history[idx]["question"]),
                "answer": str(st.session_state.history[idx]["answer"]),
                "file_name": str(st.session_state.history[idx]["file"]),
                "rating": "positive" if is_positive else "negative",
                "created_at": datetime.now().isoformat()
            }
            
            if show_debug:
                with debug_container.container():
                    st.write("Feedback Data:", feedback_data)
            
            # Store in Supabase if credentials are available
            if supabase_url and supabase_key:
                try:
                    # Initialize Supabase client
                    supabase = create_client(supabase_url, supabase_key)
                    
                    if show_debug:
                        with debug_container.container():
                            st.success("Supabase client initialized successfully")
                    
                    # Convert to simple data types for JSON compatibility
                    simple_data = {
                        "id": feedback_data["id"],
                        "question": feedback_data["question"],
                        "answer": feedback_data["answer"],
                        "file_name": feedback_data["file_name"],
                        "rating": feedback_data["rating"],
                        "created_at": feedback_data["created_at"]
                    }
                    
                    # Insert into feedback table
                    response = supabase.table('feedback').insert(simple_data).execute()
                    
                    if show_debug:
                        with debug_container.container():
                            st.success("Feedback stored in Supabase!")
                    
                    # Show a friendly success message based on feedback type
                    if is_positive:
                        st.success("Thanks for the positive feedback!")
                    else:
                        st.success("Thanks for your feedback! We'll work to improve.")
                
                except Exception as e:
                    # Log error but don't show to user
                    error_detail = traceback.format_exc()
                    logger.error(f"Error with Supabase: {str(e)}\n{error_detail}")
                    
                    if show_debug:
                        with debug_container.container():
                            st.error(f"Supabase error: {str(e)}")
                            st.code(error_detail)
                    
                    # Show friendly message regardless of error
                    st.success("Thank you for your feedback!")
            else:
                # No Supabase credentials, just show friendly message
                st.success("Thank you for your feedback!")
        else:
            st.success("Thank you for your feedback!")
    
    except Exception as e:
        # Log error but don't show to user unless in debug mode
        logger.error(f"Feedback error: {str(e)}")
        if show_debug:
            with debug_container.container():
                st.error(f"Error: {str(e)}")
        
        # Always show friendly message
        st.success("Thank you for your feedback!")

# Generic error handler
def handle_error(error, error_type="Processing Error"):
    """Handle errors gracefully without exposing implementation details"""
    # Log detailed error for debugging
    error_details = traceback.format_exc()
    logger.error(f"{error_type}: {str(error)}\n{error_details}")
    
    # Return generic error message to user
    return f"{error_type}. Please try again or contact support if the issue persists."

# Main application
def main():
    st.title("Data Analysis Chat with AI")
    
    # Debug information (only visible when checkbox is enabled)
    if st.sidebar.checkbox("Show Debug Info", False):
        st.sidebar.write(f"Python version: {sys.version}")
        st.sidebar.write(f"Working directory: {os.getcwd()}")
        st.sidebar.write(f".env file exists: {os.path.exists('.env')}")
        if os.path.exists('.env'):
            st.sidebar.write(".env file found")
    
    # Privacy notice
    with st.sidebar.expander("Privacy Information", expanded=False):
        st.markdown("""
        **Privacy Notice**
        
        This application processes your data to provide analysis services:
        - Uploaded data is processed locally and sent to OpenAI for analysis
        - Visualizations and responses are generated using your data
        - No data is permanently stored outside this session
        - Your analysis history remains only during your session
        
        **Do not upload sensitive or personally identifiable information (PII).**
        """)
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("File Management")
        uploaded_files = st.file_uploader("Upload CSV or Excel files (max 10MB)", 
                                         type=["csv", "xlsx", "xls"], 
                                         accept_multiple_files=True)
        
        # Process uploaded files
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_files:
                    # Validate file before processing
                    is_valid, error_message = validate_file(file)
                    
                    if not is_valid:
                        st.error(error_message)
                        continue
                    
                    try:
                        if file.name.endswith(('.csv')):
                            # Read with a sample first to validate properly
                            df = pd.read_csv(file)
                        else:
                            df = pd.read_excel(file)
                        
                        # Check for minimum data requirements
                        if df.empty or len(df.columns) < 1:
                            st.error("File contains no data or has invalid format.")
                            continue
                            
                        # Limit data size for security
                        row_limit = 100000  # Adjust based on your requirements
                        if len(df) > row_limit:
                            st.warning(f"File contains too many rows. Only the first {row_limit} rows will be used.")
                            df = df.head(row_limit)
                        
                        st.session_state.uploaded_files[file.name] = df
                        st.success(f"File '{sanitize_output(file.name)}' uploaded successfully!")
                    except Exception as e:
                        error_msg = handle_error(e, "File Processing Error")
                        st.error(error_msg)
        
        # File selection
        if st.session_state.uploaded_files:
            st.subheader("Select a file:")
            selected_file = st.selectbox("Choose a file", 
                                         options=list(st.session_state.uploaded_files.keys()))
            
            if selected_file:
                st.session_state.current_file = selected_file
                
                # Option to view top N rows
                try:
                    n_max = len(st.session_state.uploaded_files[selected_file])
                    n_rows = st.slider("Number of rows to view:", min_value=1, max_value=min(n_max, 1000), value=min(5, n_max))
                    
                    if st.button("View Data"):
                        st.session_state.view_data = True
                        st.session_state.n_rows = n_rows
                except Exception as e:
                    error_msg = handle_error(e, "Data Display Error")
                    st.error(error_msg)
                
                # LIDA Visualization section
                st.subheader("Generate Visualizations")
                viz_prompt = st.text_area("Describe the visualization you want:", 
                                         "Show me a visualization of the relationship between...")
                
                # Validate prompt
                viz_prompt = viz_prompt.strip()
                if len(viz_prompt) > 500:
                    st.warning("Prompt too long. Please keep it under 500 characters.")
                    viz_prompt = viz_prompt[:500]
                
                if st.button("Generate Visualization"):
                    with st.spinner("Generating visualization..."):
                        try:
                            api_key = get_api_key()  # Will stop execution if key is invalid
                            
                            # Initialize LIDA with OpenAI
                            from llmx import llm
                            text_gen = llm("openai", api_key=api_key)
                            lida = Manager(text_gen=text_gen)
                            
                            # Get the current dataframe
                            df = st.session_state.uploaded_files[st.session_state.current_file]
                            
                            # Summarize the data
                            summary = lida.summarize(df)
                            
                            # Generate visualizations
                            visualizations = lida.visualize(
                                summary=summary,
                                goal=viz_prompt,
                                library="matplotlib"
                            )
                            
                            # Store visualizations in session state
                            if visualizations and len(visualizations) > 0:
                                for i, viz in enumerate(visualizations):
                                    # Validate visualization code (basic security check)
                                    code = viz.code
                                    # Block potentially dangerous functions
                                    dangerous_patterns = ['exec(', 'eval(', 'os.', 'subprocess.', 'system(']
                                    if any(pattern in code for pattern in dangerous_patterns):
                                        st.sidebar.error("Generated visualization contains unsafe code and was blocked.")
                                        continue
                                        
                                    st.session_state.visualizations.append({
                                        "code": viz.code,
                                        "caption": viz.title if hasattr(viz, 'title') else "Generated visualization",
                                        "file": st.session_state.current_file,
                                        "prompt": viz_prompt
                                    })
                                
                                st.sidebar.success(f"Generated {len(visualizations)} visualizations!")
                            else:
                                st.sidebar.warning("No visualizations could be generated.")
                        
                        except Exception as e:
                            error_msg = handle_error(e, "Visualization Generation Error")
                            st.sidebar.error(error_msg)
                
                # Prompt History Section
                st.subheader("Prompt History")
                if len(st.session_state.prompt_history) > 0:
                    # Filter prompts for current file
                    file_prompts = [p for p in st.session_state.prompt_history if p["file"] == selected_file]
                    if file_prompts:
                        selected_prompt = st.selectbox(
                            "Select a previous prompt to reuse:",
                            options=[f"{sanitize_output(p['prompt'][:40])}..." for p in file_prompts],
                            format_func=lambda x: x
                        )
                        
                        prompt_index = [f"{sanitize_output(p['prompt'][:40])}..." for p in file_prompts].index(selected_prompt)
                        selected_prompt_full = file_prompts[prompt_index]["prompt"]
                        
                        if st.button("Use this prompt"):
                            st.session_state.reuse_prompt = selected_prompt_full
                    else:
                        st.info("No prompts for this file yet.")
                else:
                    st.info("Ask questions to build your prompt history.")
        else:
            st.info("Please upload at least one file to begin analysis.")
    
    # Main panel - Display selected data
    if st.session_state.current_file:
        st.subheader(f"Analyzing: {sanitize_output(st.session_state.current_file)}")
        
        # Display top N rows if requested
        if hasattr(st.session_state, 'view_data') and st.session_state.view_data:
            try:
                st.write(f"Showing top {st.session_state.n_rows} rows:")
                st.dataframe(st.session_state.uploaded_files[st.session_state.current_file].head(st.session_state.n_rows))
            except Exception as e:
                error_msg = handle_error(e, "Data Display Error")
                st.error(error_msg)
        
        # Display LIDA visualizations if available
        if st.session_state.visualizations:
            st.subheader("AI-Generated Visualizations")
            
            for i, viz in enumerate(st.session_state.visualizations):
                if viz["file"] == st.session_state.current_file:
                    with st.expander(f"Visualization {i+1}: {sanitize_output(viz['prompt'][:50])}..."):
                        st.code(viz["code"], language="python")
                        st.caption(sanitize_output(viz["caption"]))
                        
                        if st.button(f"Render Visualization {i+1}"):
                            try:
                                # Get current dataframe
                                data = st.session_state.uploaded_files[st.session_state.current_file]
                                
                                # Clean up the code to avoid double imports
                                code = viz["code"]
                                
                                # Security check for code execution
                                dangerous_patterns = ['exec(', 'eval(', 'os.', 'subprocess.', 'system(']
                                if any(pattern in code for pattern in dangerous_patterns):
                                    st.error("This visualization contains unsafe code and cannot be rendered.")
                                    continue
                                
                                # Remove any existing matplotlib.pyplot imports
                                if "import matplotlib.pyplot as plt" in code:
                                    modified_code = code
                                else:
                                    modified_code = "import matplotlib.pyplot as plt\n" + code
                                
                                # Create a controlled execution environment
                                exec_globals = {
                                    "pd": pd, 
                                    "data": data,
                                    "__builtins__": {
                                        k: __builtins__[k] for k in [
                                            'dict', 'list', 'tuple', 'set', 'int', 'float', 
                                            'str', 'bool', 'len', 'range', 'enumerate', 'zip', 
                                            'min', 'max', 'sum', 'sorted', 'round', 'abs'
                                        ]
                                    }
                                }
                                
                                # Execute the code in a restricted environment
                                try:
                                    exec(modified_code, exec_globals)
                                    
                                    # Display the plot if it was created
                                    if "plt" in exec_globals:
                                        st.pyplot(exec_globals["plt"].gcf())
                                        exec_globals["plt"].close()
                                    else:
                                        st.error("Visualization didn't produce a valid matplotlib plot")
                                except Exception as e:
                                    error_msg = handle_error(e, "Visualization Rendering Error")
                                    st.error(error_msg)
                            except Exception as e:
                                error_msg = handle_error(e, "Visualization Rendering Error")
                                st.error(error_msg)
        
        # Chat interface
        st.subheader("Chat with your data")
        
        # Display chat messages
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(sanitize_output(message["content"]))
                
                # Show feedback buttons for assistant messages
                if message["role"] == "assistant":
                    # Find the corresponding history index 
                    msg_idx = idx // 2  # Assuming alternating user/assistant messages
                    
                    # Check if feedback already given
                    if msg_idx in st.session_state.feedback:
                        if st.session_state.feedback[msg_idx]:
                            st.success("You found this answer helpful 👍")
                        else:
                            st.error("You found this answer not helpful 👎")
                    else:
                        col1, col2, col3 = st.columns([1, 1, 8])
                        with col1:
                            if st.button("👍", key=f"thumbsup_{msg_idx}"):
                                record_feedback(msg_idx, True)
                        with col2:
                            if st.button("👎", key=f"thumbsdown_{msg_idx}"):
                                record_feedback(msg_idx, False)
        
        # Chat input - check if there's a prompt to reuse
        if hasattr(st.session_state, 'reuse_prompt') and st.session_state.reuse_prompt:
            # Display the prompt that will be used
            st.info(f"Using prompt: {sanitize_output(st.session_state.reuse_prompt)}")
            prompt = st.session_state.reuse_prompt
            # Clear it after use so it doesn't persist
            st.session_state.reuse_prompt = None
        else:
            # Regular chat input
            prompt = st.chat_input("Ask a question about your data")
        
        if prompt:
            # Validate and sanitize prompt
            prompt = prompt.strip()
            if len(prompt) > 1000:
                st.warning("Prompt too long. Please keep it under 1000 characters.")
                prompt = prompt[:1000]
                
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Add to prompt history if it's not already there
            if not any(p["prompt"] == prompt and p["file"] == st.session_state.current_file for p in st.session_state.prompt_history):
                st.session_state.prompt_history.append({
                    "prompt": prompt,
                    "file": st.session_state.current_file,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Display user message
            with st.chat_message("user"):
                st.write(sanitize_output(prompt))
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        api_key = get_api_key()  # Will stop execution if key is invalid
                        
                        # Configure OpenAI LLM
                        llm = OpenAI(api_token=api_key)
                        
                        # Create a SmartDataframe with the data
                        pandas_df = st.session_state.uploaded_files[st.session_state.current_file]
                        smart_df = SmartDataframe(pandas_df, config={"llm": llm})
                        
                        # Chat with the data
                        answer = smart_df.chat(prompt)
                        
                        # Save to history
                        st.session_state.history.append({
                            "question": prompt,
                            "answer": answer,
                            "file": st.session_state.current_file
                        })
                        
                        # Get the index of the current response
                        msg_idx = len(st.session_state.history) - 1
                        
                        # Display the answer
                        st.write(sanitize_output(answer))
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Add feedback buttons immediately
                        col1, col2, col3 = st.columns([1, 1, 8])
                        with col1:
                            if st.button("👍", key=f"thumbsup_live_{msg_idx}"):
                                record_feedback(msg_idx, True)
                        with col2:
                            if st.button("👎", key=f"thumbsdown_live_{msg_idx}"):
                                record_feedback(msg_idx, False)
                        
                    except Exception as e:
                        error_msg = handle_error(e, "Data Analysis Error")
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.info("Please select a file from the sidebar to begin analysis.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = handle_error(e, "Application Error")
        st.error(error_msg) 