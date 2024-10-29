import facebook
import requests
from textblob import TextBlob

# Replace with your values
ACCESS_TOKEN = 'YOUR_PAGE_ACCESS_TOKEN'
PAGE_ID = 'YOUR_PAGE_ID'
COMMENT_ID_TO_DELETE = None  # Will be filled with the comment ID of the negative comment
FACEBOOK_API_VERSION = 'v17.0'  # Example API version, adjust accordingly

# Initialize the Facebook Graph API
graph = facebook.GraphAPI(access_token=ACCESS_TOKEN, version=FACEBOOK_API_VERSION)

def get_comments(post_id):
    # Get all comments from a specific post
    comments = graph.get_connections(id=post_id, connection_name='comments')
    return comments['data']

def analyze_sentiment(comment_text):
    # Use TextBlob for sentiment analysis
    analysis = TextBlob(comment_text)
    # Return sentiment polarity (-1.0 is very negative, 1.0 is very positive)
    return analysis.sentiment.polarity

def delete_comment(comment_id):
    # Delete a comment using Facebook Graph API
    response = graph.delete_object(id=comment_id)
    return response

def send_message(user_id, message_text):
    # Send a message using Facebook Graph API
    url = f"https://graph.facebook.com/{FACEBOOK_API_VERSION}/me/messages"
    payload = {
        "recipient": {"id": user_id},
        "message": {"text": message_text},
        "access_token": ACCESS_TOKEN
    }
    response = requests.post(url, json=payload)
    return response.json()

def handle_comment(post_id, comment):
    comment_id = comment['id']
    message = comment['message']
    user_id = comment['from']['id']

    # Analyze the sentiment of the comment
    sentiment = analyze_sentiment(message)
    print(f"Sentiment for '{message}': {sentiment}")

    if sentiment < -0.2:  # If sentiment is negative, adjust threshold as needed
        # Delete the negative comment
        delete_comment(comment_id)
        # Send a DM (if allowed)
        send_message(user_id, "Your comment has been flagged as inappropriate. Please follow our community guidelines.")
        print(f"Deleted comment: {message} and sent DM to User ID: {user_id}")
    else:
        print(f"Comment is positive/neutral: {message}")

# Example function to handle comments for a particular post
def monitor_post_comments(post_id):
    comments = get_comments(post_id)
    for comment in comments:
        handle_comment(post_id, comment)

# Replace 'YOUR_POST_ID' with the ID of the post you want to monitor
monitor_post_comments('YOUR_POST_ID')
