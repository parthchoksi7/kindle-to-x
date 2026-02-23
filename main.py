import os
import re
import json
import requests
import anthropic
import tweepy
from datetime import datetime, timedelta, timezone

# --- Config ---
READWISE_TOKEN = os.environ["READWISE_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
X_API_KEY = os.environ["X_API_KEY"]
X_API_SECRET = os.environ["X_API_SECRET"]
X_ACCESS_TOKEN = os.environ["X_ACCESS_TOKEN"]
X_ACCESS_SECRET = os.environ["X_ACCESS_SECRET"]

# --- Fetch new highlights from Readwise ---
def get_new_highlights():
    # Get highlights updated in the last 8 days to avoid missing any
    since = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()

    headers = {"Authorization": f"Token {READWISE_TOKEN}"}
    params = {
        "updated__gt": since,
        "page_size": 100
    }

    response = requests.get(
        "https://readwise.io/api/v2/highlights/",
        headers=headers,
        params=params
    )

    if response.status_code != 200:
        raise Exception(f"Readwise API error: {response.status_code} - {response.text}")

    data = response.json()
    results = data.get("results", [])

    if not results:
        print("No new highlights found this week.")
        return None, None

    # Get the most recent book
    # Group highlights by book
    books = {}
    for h in results:
        book_id = h.get("book_id")
        if book_id not in books:
            books[book_id] = {
                "highlights": [],
                "latest": h.get("updated")
            }
        books[book_id]["highlights"].append(h.get("text", "").strip())
        if h.get("updated") > books[book_id]["latest"]:
            books[book_id]["latest"] = h.get("updated")

    # Pick most recently updated book
    most_recent_book_id = max(books, key=lambda b: books[b]["latest"])
    highlights = books[most_recent_book_id]["highlights"]

    # Get book title
    book_response = requests.get(
        f"https://readwise.io/api/v2/books/{most_recent_book_id}/",
        headers=headers
    )
    book_title = "Unknown Book"
    if book_response.status_code == 200:
        book_title = book_response.json().get("title", "Unknown Book")

    print(f"Book: {book_title}")
    print(f"New highlights: {len(highlights)}")
    return book_title, highlights

# --- Clean posts ---
def clean_post(post):
    post = post.replace("—", "-").replace("–", "-")
    post = re.sub(r'[^\x00-\x7F]+', '', post)
    post = re.sub(r' +', ' ', post).strip()
    return post

# --- Generate posts with Claude ---
def generate_posts(highlights, book_title):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    highlights_text = "\n".join(f"- {h}" for h in highlights)

    prompt = f"""You are a book insight assistant. Based on these Kindle highlights from '{book_title}', write 3-5 tweets (max 280 chars each) sharing the most valuable learnings. Make them insightful and standalone. Use plain text only - no emojis, no icons, no em dashes. Return only the tweets, numbered 1. 2. 3. etc.

Highlights:
{highlights_text}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text
    posts = re.findall(r'\d+\.\s(.+?)(?=\n\d+\.|$)', raw, re.DOTALL)
    return [clean_post(p.strip()) for p in posts if p.strip()]

# --- Post to X ---
def post_to_x(posts, book_title):
    client = tweepy.Client(
        consumer_key=X_API_KEY,
        consumer_secret=X_API_SECRET,
        access_token=X_ACCESS_TOKEN,
        access_token_secret=X_ACCESS_SECRET
    )

    print(f"\nPosting {len(posts)} tweets about '{book_title}'...")
    for i, post in enumerate(posts):
        try:
            client.create_tweet(text=post)
            print(f"Posted {i+1}/{len(posts)}: {post[:60]}...")
        except Exception as e:
            print(f"Failed to post tweet {i+1}: {e}")
            raise

# --- Main ---
def main():
    print(f"Running at {datetime.now(timezone.utc).isoformat()}")

    # Step 1: Get highlights
    try:
        book_title, highlights = get_new_highlights()
    except Exception as e:
        print(f"Error fetching highlights: {e}")
        raise

    if not highlights:
        print("Nothing to post this week.")
        return

    # Step 2: Generate posts
    try:
        posts = generate_posts(highlights, book_title)
        print(f"\nGenerated {len(posts)} posts:")
        for i, p in enumerate(posts):
            print(f"{i+1}. {p}")
    except Exception as e:
        print(f"Error generating posts with Claude: {e}")
        raise

    # Step 3: Post to X
    try:
        post_to_x(posts, book_title)
        print("\nAll done!")
    except Exception as e:
        print(f"Error posting to X: {e}")
        raise

if __name__ == "__main__":
    main()
