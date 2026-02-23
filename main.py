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

STATE_FILE = "state.json"

# --- State Management ---
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "seen_highlights": [],
        "book_threads": {}
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# --- Fetch new highlights from Readwise ---
def get_new_highlights(seen_highlights):
    headers = {"Authorization": f"Token {READWISE_TOKEN}"}
    params = {
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
        print("No highlights found in Readwise.")
        return None, None, None, None

    # Filter out already seen highlights
    seen_set = set(seen_highlights)
    new_results = [h for h in results if h.get("text", "").strip() not in seen_set]

    if not new_results:
        print("No new highlights found since last run.")
        return None, None, None, None

    # Group new highlights by book
    books = {}
    for h in new_results:
        book_id = str(h.get("book_id"))
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

    # Get book title and author
    book_response = requests.get(
        f"https://readwise.io/api/v2/books/{most_recent_book_id}/",
        headers=headers
    )
    book_title = "Unknown Book"
    book_author = "Unknown Author"
    if book_response.status_code == 200:
        book_data = book_response.json()
        book_title = book_data.get("title", "Unknown Book")
        book_author = book_data.get("author", "Unknown Author")

    print(f"Book: {book_title} by {book_author}")
    print(f"New highlights: {len(highlights)}")
    return most_recent_book_id, book_title, book_author, highlights

# --- Clean posts ---
def clean_post(post):
    post = post.replace("—", "-").replace("–", "-")
    post = re.sub(r'[^\x00-\x7F]+', '', post)
    post = re.sub(r' +', ' ', post).strip()
    return post

# --- Generate posts with Claude ---
def generate_posts(highlights, book_title, book_author, is_continuing_thread):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    highlights_text = "\n".join(f"- {h}" for h in highlights)

    if is_continuing_thread:
        intro_rule = "This is a continuation of an existing thread about this book. Do NOT introduce the book again. Jump straight into the next insights."
    else:
        intro_rule = "The first tweet must introduce the book: include the full title and author, and set up what the thread is about."

    prompt = f"""You are a book insight assistant helping a product manager share learnings on X (Twitter).

Based on these Kindle highlights from '{book_title}' by {book_author}, write a Twitter thread of 3-5 tweets.

Rules:
- {intro_rule}
- Each insight tweet must do two things:
    1. Ground the insight in the original book context - reference the specific idea, character, situation, or concept from the book that the highlight refers to, so that readers who have read the book immediately recognize it
    2. Connect that reference to a practical product management lesson - think prioritization, user research, team dynamics, strategy, decision making, stakeholder management, or building products
- The connection between the book reference and the PM lesson should feel natural, not forced
- Each tweet must be max 280 characters
- Plain text only - no emojis, no icons, no em dashes
- End the last tweet with a summary or call to action for PMs reading

Example structure for each tweet (not a template, just to illustrate):
"In [book reference], [author] shows [original idea]. For PMs, this means [practical PM lesson]."

Return only the tweets, numbered 1. 2. 3. etc.

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

# --- Post as a thread to X ---
def post_thread_to_x(posts, book_title, last_tweet_id=None):
    client = tweepy.Client(
        consumer_key=X_API_KEY,
        consumer_secret=X_API_SECRET,
        access_token=X_ACCESS_TOKEN,
        access_token_secret=X_ACCESS_SECRET
    )

    if last_tweet_id:
        print(f"Continuing existing thread for '{book_title}'...")
    else:
        print(f"Starting new thread for '{book_title}'...")

    reply_to_id = last_tweet_id
    first_tweet_id = None

    for i, post in enumerate(posts):
        try:
            if reply_to_id is None:
                response = client.create_tweet(text=post)
            else:
                response = client.create_tweet(
                    text=post,
                    in_reply_to_tweet_id=reply_to_id
                )
            tweet_id = response.data["id"]

            if first_tweet_id is None:
                first_tweet_id = tweet_id

            reply_to_id = tweet_id
            print(f"Posted {i+1}/{len(posts)}: {post[:60]}...")
        except Exception as e:
            print(f"Failed to post tweet {i+1}: {e}")
            raise

    return first_tweet_id, reply_to_id

# --- Main ---
def main():
    print(f"Running at {datetime.now(timezone.utc).isoformat()}")

    # Load state
    state = load_state()
    seen_highlights = state.get("seen_highlights", [])
    book_threads = state.get("book_threads", {})

    # Step 1: Get new highlights
    try:
        book_id, book_title, book_author, highlights = get_new_highlights(seen_highlights)
    except Exception as e:
        print(f"Error fetching highlights: {e}")
        raise

    if not highlights:
        print("Nothing to post this week.")
        return

    # Step 2: Check if this book already has a thread
    book_thread = book_threads.get(book_id, {})
    last_tweet_id = book_thread.get("last_tweet_id")
    is_continuing_thread = last_tweet_id is not None

    if is_continuing_thread:
        print(f"Continuing existing thread for '{book_title}' from tweet {last_tweet_id}")
    else:
        print(f"Starting new thread for '{book_title}'")

    # Step 3: Generate posts
    try:
        posts = generate_posts(highlights, book_title, book_author, is_continuing_thread)
        print(f"\nGenerated {len(posts)} posts:")
        for i, p in enumerate(posts):
            print(f"{i+1}. {p}")
    except Exception as e:
        print(f"Error generating posts with Claude: {e}")
        raise

    # Step 4: Post thread to X
    try:
        first_tweet_id, last_tweet_id = post_thread_to_x(
            posts,
            book_title,
            last_tweet_id=last_tweet_id
        )
        print("\nAll posted successfully!")
    except Exception as e:
        print(f"Error posting to X: {e}")
        raise

    # Step 5: Update and save state
    if book_id not in book_threads:
        book_threads[book_id] = {
            "title": book_title,
            "first_tweet_id": first_tweet_id
        }
    book_threads[book_id]["last_tweet_id"] = last_tweet_id

    state["seen_highlights"] = list(set(seen_highlights + highlights))
    state["book_threads"] = book_threads
    save_state(state)
    print("State saved.")

if __name__ == "__main__":
    main()
