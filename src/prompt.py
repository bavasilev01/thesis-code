prompt_mimic = """
Analyze the provided chest X-ray and generate a valid radiology report in professional clinical language. The report must include two sections:

Findings: A visual description of the anatomical structures and any abnormalities visible on the image. Use objective, medically appropriate terminology. Do not interpret beyond what is visible.

Impression: A brief summary emphasizing clinically significant findings.

Base the report solely on the image provided.
"""
prompt_mimic_impression = """
Analyze the provided chest X-ray and generate a valid radiology report in professional clinical language. The report must include one section:

Impression: A brief summary emphasizing clinically significant findings.

The report must be concise and focus on the most relevant findings. Use objective, medically appropriate terminology. Do not interpret beyond what is visible.

Base the report solely on the image provided.
"""
