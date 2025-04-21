import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the fine-tuned model and tokenizer from the saved directory
model_dir = "./phobert_feedback_finetuned"  # Update if your model is stored elsewhere
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

def answer_question(question: str) -> str:
    """
    Given an input text (question or customer feedback), this function tokenizes the input,
    uses the fine-tuned PhoBERT classification model to predict its label, and then
    returns a corresponding canned response based on the predicted category.

    Parameters:
        question (str): The input text (question or feedback).

    Returns:
        str: A response message based on the predicted category.
    """
    # Tokenize the input text with the same preprocessing as during training
    inputs = tokenizer(question, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    # Ensure the model is in evaluation mode to disable dropout and speed up inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label from the model's output logits
    prediction = int(outputs.logits.argmax(dim=-1))
    
    # Define responses for each category based on your label mapping:
    # Label mapping (from your JSON):
    # {"BATTERY": 0, "CAMERA": 1, "DESIGN": 2, "FEATURES": 3, "GENERAL": 4,
    #  "Others": 5, "PERFORMANCE": 6, "PRICE": 7, "SCREEN": 8, "SER&ACC": 9, "STORAGE": 10}
    responses = {
        0: "Cảm ơn bạn về phản hồi liên quan đến pin. Chúng tôi sẽ cố gắng cải thiện chất lượng pin sản phẩm.",
        1: "Cảm ơn bạn về ý kiến liên quan đến camera. Chúng tôi sẽ ghi nhận và cải thiện chất lượng camera.",
        2: "Cảm ơn bạn về phản hồi về thiết kế. Ý kiến của bạn rất quý báu đối với chúng tôi.",
        3: "Cảm ơn bạn về phản hồi về tính năng sản phẩm. Chúng tôi sẽ xem xét kỹ các đề xuất để cải thiện.",
        4: "Cảm ơn bạn đã chia sẻ về trải nghiệm tổng quan của sản phẩm. Chúng tôi trân trọng ý kiến của bạn.",
        5: "Cảm ơn bạn đã cung cấp phản hồi. Chúng tôi luôn mong nhận được những góp ý của bạn.",
        6: "Cảm ơn bạn về phản hồi liên quan đến hiệu năng sản phẩm. Chúng tôi sẽ nỗ lực cải thiện hiệu suất làm việc.",
        7: "Cảm ơn bạn về phản hồi liên quan đến giá cả. Chúng tôi sẽ cân nhắc điều chỉnh để phù hợp hơn với khách hàng.",
        8: "Cảm ơn bạn về phản hồi liên quan đến màn hình. Chúng tôi luôn nỗ lực nâng cao chất lượng màn hình sản phẩm.",
        9: "Cảm ơn bạn về trải nghiệm dịch vụ và hỗ trợ. Chúng tôi luôn chú trọng cải thiện chất lượng dịch vụ.",
        10: "Cảm ơn bạn về phản hồi liên quan đến bộ nhớ sản phẩm. Chúng tôi sẽ xem xét và cải thiện thông số bộ nhớ."
    }
    
    # Return the response corresponding to the predicted label, or a default response if not found
    return responses.get(prediction, "Xin cảm ơn bạn đã đóng góp ý kiến!")

# Example usage:
if __name__ == "__main__":
    # Example feedback or question
    sample_question = "Pin tốt, giá cả hợp lý"
    
    # Get the answer from the fine-tuned model
    response = answer_question(sample_question)
    
    # Print the input and the corresponding response
    print("Input:", sample_question)
    print("Response:", response)