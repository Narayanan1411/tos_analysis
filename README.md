Comprehensive TOS Analyzer Chrome Extension
=============================================

Overview
--------
The Comprehensive TOS Analyzer Chrome Extension is a smart tool that integrates with a FastAPI backend to automatically extract, analyze, and interact with the Terms of Service (TOS) and policy text of any website. The extension performs the following tasks:

1. **Website Policy Extraction:**  
   Fetches the current webpage using a server-side scraper, extracts policy-related text (e.g., privacy policies, terms and conditions), and retrieves a website safety risk score from IPQualityScore.

2. **TOS Analysis:**  
   Automatically extracts the visible TOS text from the webpage and calls multiple API endpoints to:
   - Perform fraud analysis (identifying if the TOS is fraudulent or not).
   - Evaluate data access clauses (determining if the TOS’s data access is acceptable).
   - Generate a concise summarization of the TOS.

3. **Chatbot Q&A:**  
   Provides an interactive chatbot interface that answers user queries related to the TOS content.

System Architecture
-------------------
- **Frontend (Chrome Extension):**  
  - Uses Manifest V3.
  - Injects a content script to extract on-page text.
  - Calls local API endpoints for analysis.
  - Displays results in a polished popup UI with HTML and CSS.

- **Backend (FastAPI Server):**  
  - Exposes endpoints such as `/extract-policy`, `/analyze-tos`, `/check-access`, `/summarize-tos`, and `/chat-tos`.
  - Utilizes pretrained models for fraud detection, data access evaluation, summarization (BERT-based), and a question-answering chatbot.
  - Integrates with IPQualityScore API to provide a website risk score.

Requirements
------------
- **Chrome Browser:**  
  A Chromium-based browser for running the extension.
  
- **FastAPI Backend:**  
  Python 3.7+ with packages:
  - fastapi
  - uvicorn
  - transformers
  - requests
  - beautifulsoup4
  - (others as required for your models)

- **API Keys:**  
  - A valid IPQualityScore API key must be configured in the extension (in popup.js) and/or the backend.

Installation & Setup
--------------------
1. **Backend Setup:**
   - Install required Python packages:
     ```
     pip install fastapi uvicorn transformers requests beautifulsoup4
     ```
   - Place your FastAPI code (with endpoints such as `/extract-policy`, `/analyze-tos`, etc.) in a file (e.g., `main.py`).
   - Run the server:
     ```
     uvicorn main:app --reload
     ```
   - Ensure the server is running at `http://localhost:8000`.

2. **Chrome Extension Setup:**
   - Download or clone the extension files (manifest.json, popup.html, popup.css, popup.js) into a folder.
   - Open Chrome and navigate to `chrome://extensions/`.
   - Enable “Developer mode” (toggle in the top right).
   - Click “Load unpacked” and select the folder containing the extension files.
   - The extension should now appear in your Chrome toolbar.

Usage Instructions
------------------
- **Automatic Analysis:**
  - Visit any website.
  - Click the extension icon to open the popup.
  - The extension will:
    - Extract and display the website’s URL, a risk score, and a snippet of the policy text.
    - Extract TOS text from the page and automatically call the backend API endpoints to perform fraud analysis, data access evaluation, and summarization.
    - Display the results in the popup UI.

- **Chatbot Interaction:**
  - In the chatbot section, enter your question related to the TOS.
  - Click “Ask” to receive an answer from the integrated chatbot API.

Customization
-------------
- **Frontend:**
  - Modify `popup.html` and `popup.css` to customize the layout and style.
  - Update `popup.js` if you need to adjust API endpoint URLs or processing logic.

- **Backend:**
  - Replace placeholder model identifiers with your actual pretrained models.
  - Customize endpoints and processing logic to suit your requirements.

License & Contact
-----------------
For questions, suggestions, or support, please contact: narayanan.p1411@gmail.com.

Enjoy your comprehensive analysis and interaction with website policies and terms!
