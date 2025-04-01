# My AI Bro

1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-ai-bro
   ```
2. Initialize environment:

   ```
   source venv/bin/activate
   ```

3. Install Dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Initialize desired server:
   ```
   uvicorn main:app --reload
   ```
5. Initialize Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```
6. Enter http://localhost:8501/ to test the app
