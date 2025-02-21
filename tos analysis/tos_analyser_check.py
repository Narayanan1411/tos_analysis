import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# Load the fine-tuned tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("\fastapi_model")
model = BertForSequenceClassification.from_pretrained("\fastapi_model")
model.eval()

# Define some sample inputs to test the model.
# This example tests a case where the Medical department should have access to Health Records.
sample_texts = [
    """

1. Introduction
Welcome to Fake Company Inc. ("we," "us," or "our"). These Fake Terms and Conditions ("Terms") govern your use of our website, services, and applications (collectively, the "Service"). By accessing or using the Service, you agree to be bound by these Terms. If you do not agree with any part of these Terms, please refrain from using our Service.

2. Acceptance of Terms
By using our Service, you represent that you have read, understood, and agree to these Terms. We reserve the right to update, modify, or replace these Terms at any time without prior notice. Your continued use of the Service constitutes acceptance of any such changes.

3. Eligibility
You must be at least 18 years old to use our Service. By accessing the Service, you affirm that you are at least 18 years of age and capable of entering into these Terms.

4. Use of the Service
Account Creation: Some features of the Service may require you to create an account. You agree to provide accurate and complete information during registration and update such information as needed.
Prohibited Conduct: You agree not to misuse the Service. Prohibited activities include, but are not limited to, attempting to disrupt the Service, engaging in fraudulent activities, or violating any applicable laws.
5. Intellectual Property
All content, logos, trademarks, graphics, and data on the Service are the property of Fake Company Inc. or its licensors. You may not reproduce, distribute, or create derivative works from any part of the Service without our express written permission.

6. Disclaimer of Warranties
The Service is provided on an "AS IS" and "AS AVAILABLE" basis. We disclaim all warranties, express or implied, including but not limited to the implied warranties of merchantability, fitness for a particular purpose, and non-infringement. We do not warrant that the Service will be uninterrupted, secure, or error-free.

7. Limitation of Liability
In no event shall Fake Company Inc. be liable for any indirect, incidental, special, consequential, or punitive damages, or any loss of profits or revenues, whether incurred directly or indirectly, or any loss of data, use, goodwill, or other intangible losses resulting from:

Your access to or use of or inability to access or use the Service;
Any conduct or content of any third party on the Service;
Unauthorized access, use, or alteration of your transmissions or content.
8. Governing Law
These Terms shall be governed by and construed in accordance with the laws of [Fake Jurisdiction], without regard to its conflict of law principles. Any legal actions or proceedings arising out of or relating to these Terms or the Service shall be brought exclusively in the courts located in [Fake Jurisdiction].

9. Termination
We reserve the right to terminate or suspend your access to the Service at our sole discretion, without notice, for conduct that we believe violates these Terms or is harmful to other users of the Service, us, or third parties, or for any other reason.

10. Modifications to the Service
Fake Company Inc. reserves the right to modify or discontinue, temporarily or permanently, the Service (or any part thereof) with or without notice. You agree that we will not be liable for any modification, suspension, or discontinuance of the Service.

11. Contact Information
If you have any questions about these Terms, please contact us at:

Email: support@fakecompany.com
Address: 1234 Fictional Road, Imaginary City, Country

"""
]

# Run inference on each sample
for idx, text in enumerate(sample_texts):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    result = "accepted" if prediction == 1 else "Denied"
    print(f"Sample {idx+1} Prediction: {result}")