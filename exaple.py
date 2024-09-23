from transformers import pipeline

# Load the spelling correction model
fix_spelling = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")

# Sample text with spelling mistakes
# text_with_errors = "I luk foward to receving your reply"
text_with_errors = "hi howw r u , aeroplaine and it iss nhot fling helo im goood "

# Correct the spelling
clean_text = fix_spelling(text_with_errors, max_length=2048)

# Output the corrected text
print(clean_text[0]['generated_text'])
