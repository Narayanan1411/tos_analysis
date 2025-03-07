// --- Configuration ---
// Update with your actual IPQualityScore API key if used server-side in /extract-policy.
const ipqsApiKey = "your-ipqs-api-key";

// --- Helper Functions ---

// Extract TOS portion based on keywords from a block of text.
function extractTOS(text) {
  const lowerText = text.toLowerCase();
  let index = lowerText.indexOf("terms and conditions");
  if (index === -1) {
    index = lowerText.indexOf("terms of service");
  }
  // If found, return the substring starting at the detected index.
  if (index !== -1) {
    return text.substring(index);
  }
  // Fallback: return the entire text if keywords are not found.
  return text;
}

// Generic function to call a local API endpoint.
function callLocalAPI(endpoint, data, callback) {
  fetch(`http://localhost:8000/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  })
    .then(response => response.json())
    .then(result => callback(null, result))
    .catch(err => callback(err, null));
}

// Generic function to call the extract-policy endpoint (using query param "url").
function callExtractPolicy(url, callback) {
  // Append the URL as a query parameter.
  const apiUrl = `http://localhost:8000/extract-policy?url=${encodeURIComponent(url)}`;
  fetch(apiUrl)
    .then(response => response.json())
    .then(result => callback(null, result))
    .catch(err => callback(err, null));
}

// --- Main Logic ---

let tosText = ""; // Will hold the extracted TOS text.

// When the popup loads, process both website info and TOS analysis.
window.addEventListener("load", () => {
  // Query the active tab.
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const activeTab = tabs[0];
    const currentURL = activeTab.url;

    // --- Call extract-policy endpoint to fetch policy text & website score ---
    callExtractPolicy(currentURL, (err, result) => {
      const websiteInfoDiv = document.getElementById("website-info");
      if (err || result.error) {
        websiteInfoDiv.textContent = "Error retrieving website info.";
      } else {
        websiteInfoDiv.textContent =
          `URL: ${result.url}\n` +
          `Risk Score: ${result.website_score}\n` +
          `Policy Text (snippet): ${result.policy_text.substring(0, 300)}...`;
      }
    });

    // --- Extract page text from the active tab using chrome.scripting ---
    chrome.scripting.executeScript({
      target: { tabId: activeTab.id },
      func: () => document.body.innerText
    }, (results) => {
      if (chrome.runtime.lastError) {
        console.error(chrome.runtime.lastError.message);
        return;
      }
      if (results && results[0] && results[0].result) {
        let pageText = results[0].result;
        // Extract TOS-related portion.
        tosText = extractTOS(pageText).trim();
        if (!tosText || tosText.length < 50) {
          tosText = pageText; // Fallback to full text if no clear TOS found.
        }
        
        // --- Call TOS Analysis endpoints ---
        // Fraud Analysis
        callLocalAPI("analyze-tos", { tos_text: tosText }, (err, result) => {
          const fraudDiv = document.getElementById("fraud-result");
          if (err) {
            fraudDiv.textContent = "Fraud Analysis Error: " + err;
          } else {
            fraudDiv.textContent = `Fraud Analysis: ${result.fraudulent} (Score: ${result.score})`;
          }
        });
        
        // Data Access Evaluation
        callLocalAPI("check-access", { tos_text: tosText }, (err, result) => {
          const accessDiv = document.getElementById("access-result");
          if (err) {
            accessDiv.textContent = "Access Evaluation Error: " + err;
          } else {
            accessDiv.textContent = `Data Access: ${result.access} (Score: ${result.score})`;
          }
        });
        
        // Summarization
        callLocalAPI("summarize-tos", { tos_text: tosText }, (err, result) => {
          const summaryDiv = document.getElementById("summary-result");
          if (err) {
            summaryDiv.textContent = "Summarization Error: " + err;
          } else {
            summaryDiv.textContent = `Summary: ${result.summary}`;
          }
        });
      }
    });
  });
});

// --- Chatbot Interaction ---
document.getElementById("chat-btn").addEventListener("click", () => {
  const questionInput = document.getElementById("chat-question");
  const question = questionInput.value;
  const chatDiv = document.getElementById("chat-result");
  if (!question) {
    chatDiv.textContent = "Please enter a question.";
    return;
  }
  // Call the Chatbot endpoint.
  callLocalAPI("chat-tos", { question: question, tos_text: tosText }, (err, result) => {
    if (err) {
      chatDiv.textContent = "Chatbot Error: " + err;
    } else {
      chatDiv.textContent = `Answer: ${result.answer} (Score: ${result.score})`;
    }
  });
});
