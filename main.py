import os
import re
import json
import time
import random
import requests
import anthropic
import tweepy
from datetime import datetime, timezone

# --- Config ---
READWISE_TOKEN = os.environ["READWISE_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
X_API_KEY = os.environ["X_API_KEY"]
X_API_SECRET = os.environ["X_API_SECRET"]
X_ACCESS_TOKEN = os.environ["X_ACCESS_TOKEN"]
X_ACCESS_SECRET = os.environ["X_ACCESS_SECRET"]
MODE = os.environ.get("MODE", "generate")

STATE_FILE = "state.json"

# --- State Management ---
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "seen_highlights": [],
        "book_threads": {},
        "pending_tweets": []
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# --- Fetch all highlights from Readwise (once) ---
def fetch_all_highlights():
    headers = {"Authorization": f"Token {READWISE_TOKEN}"}
    all_results = []
    page = 1

    while True:
        response = requests.get(
            "https://readwise.io/api/v2/highlights/",
            headers=headers,
            params={"page_size": 100, "page": page}
        )

        if response.status_code == 429:
            wait_time = 60
            try:
                detail = response.json().get("detail", "")
                match = re.search(r'\d+', detail)
                if match:
                    wait_time = int(match.group()) + 5
            except Exception:
                pass
            print(f"Rate limited. Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            continue

        if response.status_code != 200:
            raise Exception(f"Readwise API error: {response.status_code} - {response.text}")

        data = response.json()
        results = data.get("results", [])
        all_results.extend(results)

        if not data.get("next"):
            break
        page += 1

    print(f"Fetched {len(all_results)} total highlights from Readwise.")
    return all_results

# --- Fetch book metadata ---
def fetch_book(book_id):
    headers = {"Authorization": f"Token {READWISE_TOKEN}"}
    response = requests.get(
        f"https://readwise.io/api/v2/books/{book_id}/",
        headers=headers
    )
    if response.status_code == 200:
        data = response.json()
        return data.get("title", "Unknown Book"), data.get("author", "Unknown Author")
    return "Unknown Book", "Unknown Author"

# --- Get new highlights from most recent book ---
def get_new_highlights(seen_highlights, all_highlights):
    if not all_highlights:
        print("No highlights found in Readwise.")
        return None, None, None, None, None

    # Filter out already seen highlights
    seen_set = set(seen_highlights)
    new_results = [h for h in all_highlights if h.get("text", "").strip() not in seen_set]

    if not new_results:
        print("No new highlights found since last run.")
        return None, None, None, None, None

    # Group new highlights by book
    books = {}
    for h in new_results:
        book_id = str(h.get("book_id"))
        if book_id not in books:
            books[book_id] = {
                "highlights": [],
                "latest": h.get("updated")
            }
        books[book_id]["highlights"].append({
            "text": h.get("text", "").strip(),
            "location": h.get("location"),
            "note": h.get("note", "")
        })
        if h.get("updated") > books[book_id]["latest"]:
            books[book_id]["latest"] = h.get("updated")

    # Pick most recently updated book
    most_recent_book_id = max(books, key=lambda b: books[b]["latest"])
    highlights = books[most_recent_book_id]["highlights"]

    # Sort highlights by location
    highlights.sort(key=lambda h: h["location"] or 0)

    # Estimate book length from max location across all highlights for this book
    all_book_highlights = [h for h in all_highlights if str(h.get("book_id")) == most_recent_book_id]
    max_location = max((h.get("location") or 0) for h in all_book_highlights)

    book_title, book_author = fetch_book(most_recent_book_id)

    print(f"Book: {book_title} by {book_author}")
    print(f"New highlights: {len(highlights)}")
    print(f"Estimated book length: {max_location} locations")

    return most_recent_book_id, book_title, book_author, highlights, max_location

# --- Get a random highlight from a past book ---
def get_past_highlight(seen_highlights, current_book_id, all_highlights):
    past = [
        h for h in all_highlights
        if str(h.get("book_id")) != str(current_book_id)
        and h.get("text", "").strip()
    ]

    if not past:
        print("No past highlights found.")
        return None, None, None, None, None

    chosen = random.choice(past)
    book_id = str(chosen.get("book_id"))
    book_title, book_author = fetch_book(book_id)

    book_highlights = [h for h in all_highlights if str(h.get("book_id")) == book_id]
    max_location = max((h.get("location") or 0) for h in book_highlights)

    return book_title, book_author, chosen.get("text", "").strip(), chosen.get("location"), max_location

# --- Clean posts ---
def clean_post(post):
    post = post.replace("—", "-").replace("–", "-")
    post = re.sub(r'[^\x00-\x7F]+', '', post)
    post = re.sub(r' +', ' ', post).strip()
    return post

# --- Format location context ---
def format_location_context(location, max_location):
    if not location or not max_location or max_location == 0:
        return "Location unknown"
    pct = round((location / max_location) * 100)
    if pct <= 15:
        stage = "early in the book - likely setup or introduction of core ideas"
    elif pct <= 40:
        stage = "in the first half - ideas are being developed and argued"
    elif pct <= 65:
        stage = "in the middle - arguments are deepening or being tested"
    elif pct <= 85:
        stage = "in the later sections - ideas are being synthesized or applied"
    else:
        stage = "near the end - likely conclusions, implications, or final arguments"
    return f"Location {location} of ~{max_location} ({pct}% through - {stage})"

# --- Generate thread posts from current book ---
def generate_thread_posts(highlights, book_title, book_author, max_location, is_continuing_thread):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    highlights_text = ""
    for h in highlights:
        location_context = format_location_context(h.get("location"), max_location)
        highlights_text += f"- [{location_context}] {h['text']}"
        if h.get("note"):
            highlights_text += f" (Reader note: {h['note']})"
        highlights_text += "\n"

    if is_continuing_thread:
        intro_rule = "This is a continuation of an existing thread about this book. Do NOT introduce the book again. Jump straight into the next insights."
    else:
        intro_rule = "The first tweet must introduce the book with the full title and author. Make it a hook - not a bland introduction. Set up why this book is worth paying attention to."

    prompt = f"""You are helping a product manager share book insights on X (Twitter).

Your writing style:
- Write like a sharp, well-read person who has genuine opinions - not like a book report
- Vary sentence length. Short punchy sentences. Then a longer one that builds on it.
- Have a point of view. Don't just observe - have an opinion or a take.
- Be specific. Use the actual ideas, names, and concepts from the book - not vague summaries.
- Sound human. Imperfect is fine. Conversational asides are fine.
- Never use jargon like "leverage", "actionable", "key takeaway", "delve"

Content rules:
- {intro_rule}
- Each insight tweet should reference the specific idea from the book using the location context to understand where this idea sits in the author's overall argument
- Use your judgment on whether to connect to product management. If the insight is powerful enough on its own, just share it. Don't force a PM angle.
- When you do connect to PM, make it feel natural - not like a lesson plan
- Each tweet must be max 280 characters
- Plain text only - no emojis, no icons, no em dashes
- Write exactly 3 tweets, numbered 1. 2. 3.

Book: '{book_title}' by {book_author}

Highlights with location context:
{highlights_text}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text
    posts = re.findall(r'\d+\.\s(.+?)(?=\n\d+\.|$)', raw, re.DOTALL)
    return [clean_post(p.strip()) for p in posts if p.strip()]

# --- Generate single post from past book ---
def generate_single_post(highlight_text, book_title, book_author, location, max_location):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    location_context = format_location_context(location, max_location)

    prompt = f"""You are helping a product manager share book insights on X (Twitter).

Your writing style:
- Write like a sharp, well-read person who has genuine opinions - not like a book report
- Vary sentence length. Short punchy sentences. Then a longer one that builds on it.
- Have a point of view. Don't just observe - have an opinion or a take.
- Be specific. Use the actual ideas, names, and concepts from the book.
- Sound human. Imperfect is fine. Conversational asides are fine.
- Never use jargon like "leverage", "actionable", "key takeaway", "delve"

Content rules:
- Write exactly 1 tweet based on this highlight
- Use the location context to understand where this idea sits in the book's overall argument
- Use your judgment on whether to connect to product management. If the insight stands alone, just share it powerfully. Don't force a PM angle.
- Include the book title and author naturally somewhere in the tweet
- Max 280 characters
- Plain text only - no emojis, no icons, no em dashes

Book: '{book_title}' by {book_author}
Highlight: [{location_context}] {highlight_text}

Return only the tweet text, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    return clean_post(response.content[0].text.strip())

# --- Post a single tweet to X ---
def post_single_tweet(tweet_text, reply_to_id=None):
    client = tweepy.Client(
        consumer_key=X_API_KEY,
        consumer_secret=X_API_SECRET,
        access_token=X_ACCESS_TOKEN,
        access_token_secret=X_ACCESS_SECRET
    )

    if reply_to_id:
        response = client.create_tweet(
            text=tweet_text,
            in_reply_to_tweet_id=reply_to_id
        )
    else:
        response = client.create_tweet(text=tweet_text)

    return response.data["id"]

# --- GENERATE MODE: Sunday 9am ---
def run_generate():
    print("Mode: GENERATE")

    state = load_state()
    seen_highlights = state.get("seen_highlights", [])
    book_threads = state.get("book_threads", {})

    print("Fetching all highlights from Readwise...")
    all_highlights = fetch_all_highlights()

    result = get_new_highlights(seen_highlights, all_highlights)
    if result[0] is None:
        print("Nothing to generate this week.")
        return

    book_id, book_title, book_author, highlights, max_location = result

    book_thread = book_threads.get(book_id, {})
    last_tweet_id = book_thread.get("last_tweet_id")
    is_continuing_thread = last_tweet_id is not None

    monday_posts = generate_thread_posts(
        highlights, book_title, book_author, max_location, is_continuing_thread
    )

    wed_result = get_past_highlight(seen_highlights, book_id, all_highlights)
    wednesday_post = None
    if wed_result[0]:
        wed_title, wed_author, wed_text, wed_location, wed_max = wed_result
        wednesday_post = generate_single_post(wed_text, wed_title, wed_author, wed_location, wed_max)

    fri_result = get_past_highlight(seen_highlights, book_id, all_highlights)
    friday_post = None
    if fri_result[0]:
        fri_title, fri_author, fri_text, fri_location, fri_max = fri_result
        friday_post = generate_single_post(fri_text, fri_title, fri_author, fri_location, fri_max)

    print(f"\nMonday thread ({len(monday_posts)} tweets):")
    for i, p in enumerate(monday_posts):
        print(f"{i+1}. {p}\n")

    if wednesday_post:
        print(f"Wednesday post:\n{wednesday_post}\n")

    if friday_post:
        print(f"Friday post:\n{friday_post}\n")

    pending = []

    for i, post in enumerate(monday_posts):
        pending.append({
            "order": i,
            "day": "monday",
            "text": post,
            "posted": False,
            "tweet_id": None,
            "book_id": book_id,
            "book_title": book_title
        })

    if wednesday_post:
        pending.append({
            "order": 3,
            "day": "wednesday",
            "text": wednesday_post,
            "posted": False,
            "tweet_id": None,
            "book_id": None,
            "book_title": None
        })

    if friday_post:
        pending.append({
            "order": 4,
            "day": "friday",
            "text": friday_post,
            "posted": False,
            "tweet_id": None,
            "book_id": None,
            "book_title": None
        })

    state["pending_tweets"] = pending
    state["seen_highlights"] = list(set(seen_highlights + [h["text"] for h in highlights]))
    save_state(state)
    print("All posts saved to state.")

# --- POST MODE ---
def run_post():
    print("Mode: POST")

    state = load_state()
    pending = state.get("pending_tweets", [])
    book_threads = state.get("book_threads", {})

    today = datetime.now(timezone.utc).strftime("%A").lower()
    print(f"Today is {today}")

    todays_tweets = [t for t in pending if t.get("day") == today and not t.get("posted")]

    if not todays_tweets:
        print(f"No pending tweets for {today}.")
        return

    tweet = todays_tweets[0]
    book_id = tweet.get("book_id")
    tweet_text = tweet.get("text")
    tweet_order = tweet.get("order")
    day = tweet.get("day")

    reply_to_id = None
    if day == "monday" and book_id:
        book_thread = book_threads.get(book_id, {})
        reply_to_id = book_thread.get("last_tweet_id")

    print(f"Posting {day} tweet (order {tweet_order}):")
    print(tweet_text)

    try:
        tweet_id = post_single_tweet(tweet_text, reply_to_id=reply_to_id)
        print(f"Posted successfully. Tweet ID: {tweet_id}")

        for t in pending:
            if t["order"] == tweet_order:
                t["posted"] = True
                t["tweet_id"] = str(tweet_id)

        if day == "monday" and book_id:
            if book_id not in book_threads:
                book_threads[book_id] = {
                    "title": tweet.get("book_title"),
                    "first_tweet_id": str(tweet_id)
                }
            book_threads[book_id]["last_tweet_id"] = str(tweet_id)

        state["pending_tweets"] = pending
        state["book_threads"] = book_threads
        save_state(state)

    except Exception as e:
        print(f"Failed to post tweet: {e}")
        raise

# --- Main ---
def main():
    print(f"Running at {datetime.now(timezone.utc).isoformat()}")
    print(f"Mode: {MODE}")

    if MODE == "generate":
        run_generate()
    elif MODE == "post":
        run_post()
    else:
        raise Exception(f"Unknown mode: {MODE}")

if __name__ == "__main__":
    main()
