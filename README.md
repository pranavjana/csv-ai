# AI-Powered Data Analysis Application

A secure, interactive application for analyzing CSV and Excel files using AI capabilities. This project enables users to upload data files, ask questions in natural language, and generate visualizations without writing code.

## Features

### Data Upload and Management
- Upload CSV and Excel files (with size and content validation)
- View configurable number of rows from your datasets
- File validation for security and data integrity

### AI-Powered Analysis
- **Natural Language Queries**: Ask questions about your data in plain English
- **Smart Visualization Generation**: Create charts by describing what you want to see
- **Integrated with OpenAI**: Leverages advanced language models for data understanding

### User Experience
- **Prompt History**: Save and reuse previous questions
- **Feedback System**: Rate AI responses with feedback stored in Supabase
- **Chat-like Interface**: Intuitive conversational UI for data exploration

### Security Features
- **File Validation**: Size limits, content verification, and extension checking
- **Input Sanitization**: Protects against XSS and injection attacks
- **Controlled Code Execution**: Restricted environment for visualization rendering
- **Error Handling**: Generic user-facing errors with detailed backend logging
- **Privacy Focused**: Clear notices about data handling and processing

## Installation

### Prerequisites
- Python 3.11+ (required for full compatibility)
- pip or Poetry for package management

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   SUPABASE_URL=your_supabase_url_here
   SUPABASE_KEY=your_supabase_anon_key_here
   ```

## Usage

1. **Run the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload Data Files**
   - Use the file uploader in the sidebar to add CSV or Excel files
   - Files must be under 10MB and contain valid data

3. **Explore Your Data**
   - Select a file from the dropdown
   - Choose how many rows to display
   - Click "View Data" to see a preview

4. **Ask Questions**
   - Type natural language questions in the chat box
   - View AI-generated answers and explanations
   - Rate responses with thumbs up/down

5. **Generate Visualizations**
   - Describe the visualization you want in plain language
   - Explore generated code and rendered charts
   - Use the prompt history to reuse effective queries

## Security Considerations

This application implements several security features:

### Data Protection
- No data persistence beyond the current session
- Clear privacy notice about data handling
- Prevention of sensitive PII upload

### Input/Output Safety
- HTML escaping for all user inputs and AI outputs
- Length limits on prompts and visualizations
- Content validation for all file uploads

### Code Execution Safety
- Restricted execution environment for visualization code
- Dangerous function detection and blocking
- Limited builtin function access

### Error Handling
- Generic user-facing error messages
- Detailed server-side logging for troubleshooting
- Centralized error handling system

## Architecture

### Key Components
- **Streamlit**: Frontend interface and session management
- **PandaAI**: Natural language processing for data querying
- **Microsoft LIDA**: AI-powered visualization generation
- **Pandas**: Data handling and manipulation
- **Supabase**: Database for storing user feedback and ratings

### Data Flow
1. User uploads files → Validation → Storage in session state
2. User asks questions → PandaAI processes with OpenAI → Returns answers
3. User requests visualization → LIDA generates code → Renders charts
4. User rates responses → Feedback stored in Supabase → Used for future improvements

### Security Layers
- Input validation → Content sanitization → Restricted execution → Output sanitization

## Acknowledgments

- [PandaAI](https://github.com/gventuri/pandas-ai) - For natural language data analysis
- [Microsoft LIDA](https://github.com/microsoft/lida) - For AI visualization generation
- [Streamlit](https://streamlit.io/) - For the interactive web interface 
