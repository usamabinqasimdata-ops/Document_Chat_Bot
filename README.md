# Document Chat Bot

## Overview
Document Chat Bot is a Python application that allows users to interact with documents using natural language processing. It leverages OpenAI's language models to answer questions based on the content of uploaded files.

## Features
- Upload PDF and DOCX files for analysis. It process embed then you answer question you text.
- Ask questions about the content of the documents.
- Utilizes OpenAI's GPT-3.5-turbo model for conversational responses.

## Requirements
- Python 3.11
- pip install -r requirements.txt

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd Document_Chat_Bot
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=<your-api-key>
   ```
2. Run the application:
   ```bash
   streamlit run main.py
   ```
3. Open your browser and go to `http://localhost:8501` to interact with the app.

## Contributing
Feel free to submit issues or pull requests for improvements.

## License
This project is licensed under the MIT License.
