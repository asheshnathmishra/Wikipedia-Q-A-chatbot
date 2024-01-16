from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import re

# Load pre-trained RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')
model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

def clean_text(text):
    """
    Cleans the Wikipedia text by removing HTML tags, hyper references, 
    paragraph numbers, etc.
    """
    text = re.sub('<.*?>', '', text)      # Remove HTML tags
    text = re.sub(r'\[\d+\]', '', text)   # Remove reference numbers like [1], [2], etc.
    text = text.replace('\n', ' ')        # Replace newline characters with space
    text = text.replace('<s>', '').replace('</s>', '')  # Remove <s> tags
    return text

def answer_question(model, tokenizer, question, context, max_length=512):
    # Tokenize the question to understand its length
    question_tokens = tokenizer.encode(question, add_special_tokens=True)

    # Initialize variables to store processed segments
    segments = []

    # Tokenize the context and split into segments
    remaining_tokens = tokenizer.encode(context, add_special_tokens=False, add_prefix_space=True)
    while remaining_tokens:
        segment_tokens = []
        while remaining_tokens and len(question_tokens) + len(segment_tokens) < max_length:
            segment_tokens.append(remaining_tokens.pop(0))
        segments.append(tokenizer.decode(segment_tokens))

    # Iterate over each segment and look for the answer
    for segment in segments:
        inputs = tokenizer.encode_plus(question, segment, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = inputs['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        with torch.no_grad():
            outputs = model(input_ids)

        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Check if a valid answer is found in this segment
        if answer_start < answer_end:
            return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))

    return "Sorry, I could not find an answer in the text."

    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
