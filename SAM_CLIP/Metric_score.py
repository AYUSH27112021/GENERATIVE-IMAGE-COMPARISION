def adjust_score(images_scores, threshold=0.2, initial_score=0.5):
    """
    Adjusts the score based on the given image scores and a threshold for halucinating.

    Parameters:
    - images_scores: List of scores for images.
    - threshold: Hard set value for comparision 0.3 in this case
    - initial_score: Starting score.

    Returns:
    - final_score: Score within the range 0-1.
    """
    score = initial_score

    for image_score in images_scores:
        if image_score > threshold:
            # Quadratic increase for scores above the threshold
            score += ((image_score - threshold) ** 2) / len(images_scores)
        else:
            # Quadratic penalty for scores below the threshold
            score -= ((threshold - image_score) ** 2) / len(images_scores)
    
    # Normalize the score to be within 0-1 range
    final_score = min(max(score, 0), 1)
    
    return final_score

# # Example usage
# images_scores = [0.908691 , 0.091370]
# final_score = adjust_score(images_scores, threshold=0.3, initial_score=0.5)
# print(f"Final Score: {final_score}")
