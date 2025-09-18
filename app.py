from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local testing)
load_dotenv()

app = Flask(__name__)
CORS(app) # Allows your frontend to talk to this backend

# --- This switch is crucial for local testing vs. deployment ---
IS_DEV_MODE = os.getenv("DEV_MODE") == "True"
agent = None # Initialize agent as None

# --- GLOBAL SETUP: Only run when deployed (NOT in dev mode) ---
if not IS_DEV_MODE:
    print("🚀 Starting in FULL mode. Initializing AI Agent...")
    try:
        from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
        from langchain_google_genai import GoogleGenerativeAI

        # 1. Load your final cleaned data file
        df_cleaned = pd.read_csv("Water_Resources_Cleaned_Final.csv")
        print("✅ Data loaded successfully.")

        # 2. Initialize the AI Agent
        # Standardized the key name to GOOGLE_API_KEY for best practice
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment for full mode")

        llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
        # Setting verbose=False removes the "thinking process" logs
        agent = create_pandas_dataframe_agent(llm, df_cleaned, verbose=False, allow_dangerous_code=True)
        print("✅ LangChain Agent is ready.")

    except Exception as e:
        print(f"🚨 An error occurred during startup in full mode: {e}")
else:
    print("💻 Starting in local DEV_MODE (dummy mode). AI Agent is disabled.")


@app.route('/ask', methods=['POST'])
def ask_agent():
    # --- DUMMY MODE LOGIC (for local testing) ---
    if IS_DEV_MODE:
        prompt = request.json.get('prompt', '')
        print(f"-> Received request in dummy mode for prompt: '{prompt[:30]}...'")
        return jsonify({
            "answer": f"This is a dummy response. Your website is connected correctly!",
            "image_url": None
        })

    # --- FULL MODE LOGIC (for Render deployment) ---
    if agent is None:
        return jsonify({"error": "Agent not initialized. Check server logs."}), 500
    
    state_to_analyze = request.json.get('prompt')
    if not state_to_analyze:
        return jsonify({"error": "No state name provided"}), 400

    analysis_template = f"""
    For the state of '{state_to_analyze}', please perform the following analysis using the seaborn/matplotlib library:
    1.  **Bar Chart:** Create a bar chart showing the average water 'LEVEL' for each 'DISTRICT'.
    2.  **Trend Plot:** Ensure the 'Date' column is a datetime type, then create a line plot showing how the average water 'LEVEL' for the whole state has changed over the years.
    3.  **Distribution Plot:** Create a box plot to compare the distribution of water 'LEVEL' readings across the top 5 districts with the most data.
    4.  **Final Explanation:** After creating the plots, provide a detailed but simple explanation for all of them in common, non-technical language. Explain what each chart means for the region. Make the colors of all diagrams different and visually appealing.
    You MUST save the combined output of all plots into a single image file.
    """
    
    plot_path = os.path.join('static', 'plot.png')
    if os.path.exists(plot_path):
        os.remove(plot_path)

    full_agent_prompt = f"{analysis_template}\n\nIMPORTANT: Save the final combined image of all plots to '{plot_path}'."
    
    try:
        response = agent.invoke({"input": full_agent_prompt})
        answer = response.get('output', 'Sorry, I could not find an answer.')

        image_url = None
        if os.path.exists(plot_path):
            image_url = url_for('static', filename='plot.png', _external=True)
            
        return jsonify({"answer": answer, "image_url": image_url})

    except Exception as e:
        print(f"🚨 An error occurred while running the agent: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)