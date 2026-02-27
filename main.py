import os
import re
import json
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
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPO = os.environ["GITHUB_REPO"]  # e.g. "username/repo"
X_BEARER_TOKEN = os.environ["X_BEARER_TOKEN"]
X_USER_ID = os.environ["X_USER_ID"]
MODE = os.environ.get("MODE", "generate")

STATE_FILE = "state.json"
INTERVIEW_ISSUE_LABEL = "weekly-questions"

# --- Voice reference: real answers from the account owner ---
# These grow over time as more answers are collected. Used to calibrate tone.
VOICE_EXAMPLES = [
    {
        "q": "What's the last thing that annoyed you at work - not a big problem, just a small friction you couldn't stop thinking about?",
        "a": "after taking a program manager through an entire flow that was scoped out by me, he set up another meeting with 10 people trying to get answers to things that I had already documented. the doc was never opened.",
        "tweet": "Spent an hour walking a stakeholder through a fully documented flow. They then scheduled a 10-person meeting to answer questions that were in the doc. The doc was never opened. Getting people to read your work is the whole job."
    }
]

# --- Who this person is --- used to generate better questions and more specific tweets
PERSON_CONTEXT = """
Who they are:
- Product manager at a tech company working on complex, multi-stakeholder products
- Trains with a personal trainer - sessions feel like deep work time, a rare space away from meetings and Slack to think through hard problems
- Currently on a short training break - reading has filled the gap naturally, habits trade off against each other
- Dad to a 2.5 year old daughter
- Heavy reader - uses Kindle highlights to capture ideas
- Voice: direct, specific, slightly dry, no fluff. Talks like a person not a content creator.
"""

def get_voice_context(state=None):
    examples = list(VOICE_EXAMPLES)
    # Merge in any high-rated examples saved in state
    if state:
        for ex in state.get("voice_examples", []):
            if ex not in examples:
                examples.append(ex)
    examples_text = ""
    for ex in examples:
        examples_text += f"Raw answer: \"{ex['a']}\"\nTweet written from it: \"{ex['tweet']}\"\n\n"
    return f"""Here are real examples of how this person talks and how their answers get turned into tweets.
Use these to match their voice - direct, specific, no fluff, slightly dry:

{examples_text}
{PERSON_CONTEXT}"""

# --- State Management ---
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            print("Warning: state.json corrupted or empty, resetting.")
    return {
        "seen_highlights": [],
        "book_threads": {},
        "pending_tweets": [],
        "interview": {
            "issue_number": None,
            "questions": [],
            "sent_at": None,
            "pending_tweets": []
        }
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# --- GitHub API helpers ---
def github_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

def create_github_issue(title, body):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    payload = {
        "title": title,
        "body": body,
        "labels": [INTERVIEW_ISSUE_LABEL]
    }
    response = requests.post(url, headers=github_headers(), json=payload)
    if response.status_code != 201:
        raise Exception(f"GitHub API error creating issue: {response.status_code} - {response.text}")
    return response.json()["number"]

def close_github_issue(issue_number):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{issue_number}"
    requests.patch(url, headers=github_headers(), json={"state": "closed"})

def get_issue_comments(issue_number):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{issue_number}/comments"
    response = requests.get(url, headers=github_headers())
    if response.status_code != 200:
        return []
    return response.json()

# --- Fetch new highlights from Readwise ---
def get_new_highlights(seen_highlights):
    headers = {"Authorization": f"Token {READWISE_TOKEN}"}
    params = {"page_size": 100}

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

    seen_set = set(seen_highlights)
    new_results = [h for h in results if h.get("text", "").strip() not in seen_set]

    if not new_results:
        print("No new highlights found since last run.")
        return None, None, None, None

    books = {}
    for h in new_results:
        book_id = str(h.get("book_id"))
        # Use highlighted_at (when you actually highlighted in Kindle) not updated (Readwise metadata)
        highlighted_at = h.get("highlighted_at") or h.get("updated") or ""
        if book_id not in books:
            books[book_id] = {"highlights": [], "latest": highlighted_at}
        books[book_id]["highlights"].append(h.get("text", "").strip())
        if highlighted_at > books[book_id]["latest"]:
            books[book_id]["latest"] = highlighted_at

    most_recent_book_id = max(books, key=lambda b: books[b]["latest"])
    highlights = books[most_recent_book_id]["highlights"]

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
    post = post.replace("\u2014", "-").replace("\u2013", "-")
    post = re.sub(r'[^\x00-\x7F]+', '', post)
    post = re.sub(r' +', ' ', post).strip()
    return post

# --- Generate book posts with Claude ---
def generate_posts(highlights, book_title, book_author, is_continuing_thread):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    highlights_text = "\n".join(f"- {h}" for h in highlights)

    if is_continuing_thread:
        intro_rule = "This is a continuation of an existing thread about this book. Do NOT introduce the book again. Jump straight into the next insights. Tweet 1 must lead with the most counterintuitive or surprising idea from the highlights."
    else:
        intro_rule = """Tweet 1 must lead with the single most counterintuitive or surprising idea from the highlights - that is the hook.
Weave in the book title and author naturally within tweet 1 (e.g. "Thiel argues in Zero to One that..." or "In Zero to One, Thiel makes a point that...") - do NOT open with "I am reading..." or "Here are lessons from...".
The reader should be hooked by the idea first, then realize what book it came from."""

    prompt = f"""You are a book insight assistant helping a product manager share learnings on X (Twitter).

Based on these Kindle highlights from '{book_title}' by {book_author}, write a Twitter thread of exactly 3 tweets.

Rules:
- {intro_rule}
- Each tweet must do two things:
    1. Ground the insight in the original book context - reference the specific idea, character, situation, or concept from the book
    2. Connect that reference to a practical product management lesson - think prioritization, user research, team dynamics, strategy, decision making, stakeholder management, or building products
- The connection between the book reference and the PM lesson should feel natural, not forced
- Each tweet must be max 280 characters
- Plain text only - no emojis, no icons, no em dashes
- Tweet 3 ends with a punchy takeaway or open question for PMs - not a generic call to action

Return only the tweets, numbered 1. 2. 3.

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

# --- Generate interview questions with Claude ---
def generate_interview_questions():
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    voice_context = get_voice_context()

    # Load recent question history to avoid repeats
    state = load_state()
    recent_questions = state.get("question_history", [])[-8:]
    history_text = ""
    if recent_questions:
        history_text = "Questions asked in recent weeks (DO NOT repeat these angles):\n" + "\n".join(f"- {q}" for q in recent_questions) + "\n\n"

    prompt = f"""You help a PM who runs @marginnotespm on X build their brand through short, specific posts.

{voice_context}

{history_text}Generate 2 questions to ask them this week to draw out something worth posting.

The questions that work best are ones that:
- Ask about a specific recent moment, not a general opinion
- Feel like a therapist or a curious friend asking, not a journalist
- Make them think but don't require a polished answer

Good question angles to rotate between:
- A small frustration or friction at work they couldn't stop thinking about
- Something their daughter did or said that reframed something for them
- The last time they felt like a really good PM and what they were actually doing
- What they think about during training sessions - what problem keeps coming back
- How habits trade off against each other (e.g. when one drops, another fills the space)
- An idea from a book they want to push back on
- Something they changed their mind about recently
- A moment where they noticed a gap between what they said and what they did

Keep each question to one casual sentence. Make them feel unexpected, not like a survey.

Return only the 2 questions, numbered 1. and 2."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text
    questions = re.findall(r'\d+\.\s(.+?)(?=\n\d+\.|$)', raw, re.DOTALL)
    return [q.strip() for q in questions if q.strip()]

# --- Generate standalone Thursday tweet from highlights ---
def generate_standalone_tweet(highlights, book_title, book_author):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    highlights_text = "\n".join(f"- {h}" for h in highlights)

    prompt = f"""You help a PM who runs @marginnotespm on X.

From these Kindle highlights from '{book_title}' by {book_author}, extract the single most striking or counterintuitive idea and write ONE standalone tweet about it.

Rules:
- This is a standalone tweet, not part of a thread - it must work completely on its own
- Do not introduce the book in a generic way - either reference it naturally or lead with the idea itself
- Make it feel like a real observation, not a summary
- Plain text only - no emojis, no em dashes, no hashtags
- Max 240 characters (leave room for engagement)
- Direct and specific - no throat-clearing

Return only the tweet, no numbering or labels.

Highlights:
{highlights_text}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    return clean_post(response.content[0].text.strip())

# --- Accounts to monitor for reply opportunities ---
REPLY_ACCOUNTS = [
    "shreyas",        # Ex-Stripe PM, human side of product
    "lennysan",       # Ex-Airbnb, huge PM following
    "ttorres",        # Teresa Torres, continuous discovery
    "sahilbloom",     # Writer, books + life lessons
    "johncutlefish",  # John Cutler, contrarian PM takes
    "jasonfried",     # Basecamp, anti-conventional wisdom
    "nirandfar",      # Nir Eyal, books + product overlap
]

# --- Fetch recent tweets from a user via X API ---
def get_user_tweets(username, max_results=3):
    # First get user ID
    user_url = f"https://api.twitter.com/2/users/by/username/{username}"
    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    user_resp = requests.get(user_url, headers=headers)
    if user_resp.status_code != 200:
        print(f"Could not fetch user {username}: {user_resp.status_code}")
        return []
    user_id = user_resp.json().get("data", {}).get("id")
    if not user_id:
        return []

    # Then get their recent tweets
    tweets_url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    params = {
        "max_results": max_results,
        "tweet.fields": "public_metrics,created_at",
        "exclude": "retweets,replies"
    }
    tweets_resp = requests.get(tweets_url, headers=headers, params=params)
    if tweets_resp.status_code != 200:
        print(f"Could not fetch tweets for {username}: {tweets_resp.status_code}")
        return []

    tweets = tweets_resp.json().get("data", [])
    return [{"handle": username, "text": t["text"], "id": t["id"],
             "likes": t.get("public_metrics", {}).get("like_count", 0)} for t in tweets]

# --- Generate reply suggestions from real tweets ---
def generate_reply_suggestions(issue_number):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    voice_context = get_voice_context()

    # Fetch real tweets from tracked accounts
    all_tweets = []
    for account in REPLY_ACCOUNTS:
        tweets = get_user_tweets(account, max_results=3)
        all_tweets.extend(tweets)

    if not all_tweets:
        print("Could not fetch any tweets. Skipping reply suggestions.")
        return

    # Sort by likes and take top 6 as candidates
    all_tweets.sort(key=lambda t: t["likes"], reverse=True)
    candidates = all_tweets[:6]
    candidates_text = "\n\n".join(
        f"@{t['handle']}: {t['text']} (likes: {t['likes']})"
        for t in candidates
    )

    prompt = f"""You help a PM who runs @marginnotespm on X grow their following by replying to relevant tweets.

{voice_context}

Here are recent tweets from PM and books accounts they follow. Pick the 3 best ones to reply to and write the reply.

Choose tweets where this person would have something specific and genuine to say - a pushback, a real example from their experience, or an unexpected angle.

Tweets to choose from:
{candidates_text}

For each of the 3 you pick, format as:
ACCOUNT: @handle
TWEET: [the tweet text]
REPLY: [their reply - under 240 chars, plain text, direct, slightly dry, no emojis]

Pick 3."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    comment_body = f"""## Reply suggestions for this week

3 real tweets from accounts you follow - worth replying to. Copy the reply, tweak if needed, post manually. Takes 5 min.

{raw}

---
*Replying to accounts your target audience follows is the fastest way to grow at your stage.*"""

    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{issue_number}/comments"
    requests.post(url, headers=github_headers(), json={"body": comment_body})
    print("Reply suggestions posted to GitHub Issue.")

# --- Generate interview tweets from answers ---
def generate_interview_tweets(questions, answer_text):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    voice_context = get_voice_context()

    prompt = f"""You help a PM who runs @marginnotespm on X craft tweets from their raw thoughts.

{voice_context}

They were asked these questions:
Q1: {questions[0]}
Q2: {questions[1] if len(questions) > 1 else ''}

Their answer (raw, unedited):
{answer_text}

Write a 2-tweet thread based on their answer. Rules:
- Match the voice from the examples above - direct, no fluff, slightly dry
- Be specific - use the actual details, friction, and texture from their answer
- Start tweet 1 with the concrete situation or moment, not a generic setup line
- Tweet 2 lands the insight or reframe - leave it slightly open, not wrapped up too neatly
- Plain text only - no emojis, no em dashes, no hashtags
- Each tweet max 280 characters
- Do not moralize or over-explain

Return only the 2 tweets, numbered 1. and 2."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text
    posts = re.findall(r'\d+\.\s(.+?)(?=\n\d+\.|$)', raw, re.DOTALL)
    return [clean_post(p.strip()) for p in posts if p.strip()]

# --- Post a single tweet to X ---
def post_single_tweet(tweet_text, reply_to_id=None):
    client = tweepy.Client(
        consumer_key=X_API_KEY,
        consumer_secret=X_API_SECRET,
        access_token=X_ACCESS_TOKEN,
        access_token_secret=X_ACCESS_SECRET
    )

    if reply_to_id:
        response = client.create_tweet(text=tweet_text, in_reply_to_tweet_id=reply_to_id)
    else:
        response = client.create_tweet(text=tweet_text)

    return response.data["id"]

# --- INTERVIEW_ASK MODE: Run Friday 9am ---
# --- Fetch analytics for last 20 tweets ---
def fetch_tweet_analytics():
    """Fetch last 20 tweets with public metrics via X API v2 using Bearer Token."""
    tw = tweepy.Client(bearer_token=X_BEARER_TOKEN)

    # Fetch last 20 tweets with public_metrics
    response = tw.get_users_tweets(
        id=X_USER_ID,
        max_results=20,
        tweet_fields=["public_metrics", "created_at", "text"],
        exclude=["retweets", "replies"]
    )

    if not response.data:
        print("No tweets found for analytics.")
        return []

    tweets = []
    for t in response.data:
        m = t.public_metrics
        tweets.append({
            "id": t.id,
            "text": t.text[:100] + "..." if len(t.text) > 100 else t.text,
            "created_at": str(t.created_at),
            "likes": m["like_count"],
            "retweets": m["retweet_count"],
            "replies": m["reply_count"],
            "impressions": m.get("impression_count", 0),
        })

    return tweets

# --- Generate analytics digest and post to GitHub Issue ---
def post_analytics_digest(issue_number):
    print("Fetching tweet analytics...")
    tweets = fetch_tweet_analytics()

    if not tweets:
        print("No analytics data available. Skipping digest.")
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    tweets_text = ""
    for t in tweets:
        line = f"- {t['text']} | likes: {t['likes']} | retweets: {t['retweets']} | replies: {t['replies']} | impressions: {t['impressions']}"
        tweets_text += line + "\n"

    prompt = f"""You are analyzing tweet performance for @marginnotespm, an anonymous PM account focused on books and product management.

Here are their last 20 tweets with metrics:
{tweets_text}

Write a brief analytics digest with:
1. Top 3 tweets by engagement (likes + retweets + replies combined)
2. What patterns you notice - which topics, formats, or angles performed best
3. One clear recommendation for what to post more of based on the data

Keep it concise and actionable. Plain text, no emojis."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    digest = response.content[0].text.strip()

    # Build full metrics table
    metrics_table = "| Tweet | Likes | RTs | Replies | Impressions |\n|---|---|---|---|---|\n"
    for t in tweets:
        short = t["text"][:60].replace("|", "-")
        metrics_table += f"| {short}... | {t['likes']} | {t['retweets']} | {t['replies']} | {t['impressions']} |\n"

    comment_body = f"""## Weekly analytics digest

{digest}

---

### Raw numbers (last 20 tweets)

{metrics_table}"""

    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{issue_number}/comments"
    requests.post(url, headers=github_headers(), json={"body": comment_body})
    print("Analytics digest posted to GitHub Issue.")

def run_interview_ask():
    print("Mode: INTERVIEW_ASK")

    state = load_state()

    # Close any open interview issue from last week
    old_issue = state.get("interview", {}).get("issue_number")
    if old_issue:
        close_github_issue(old_issue)
        print(f"Closed old issue #{old_issue}")

    questions = generate_interview_questions()
    print(f"Generated questions: {questions}")

    body = """Drop your answers as a single comment below. As casual as you like - rambling is fine.

**Q1:** {}

**Q2:** {}

---
*Your comment will be picked up automatically on Sunday and turned into a tweet posted Wednesday.*
""".format(questions[0], questions[1] if len(questions) > 1 else "")

    issue_number = create_github_issue(
        title=f"Weekly questions - {datetime.now(timezone.utc).strftime('%b %d')}",
        body=body
    )
    print(f"Created issue #{issue_number}")

    # Save questions to history
    question_history = state.get("question_history", [])
    question_history.extend(questions)
    state["question_history"] = question_history[-16:]  # Keep last 16 (8 weeks)

    state["interview"] = {
        "issue_number": issue_number,
        "questions": questions,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "pending_tweets": []
    }
    save_state(state)

    # Post analytics digest to the new issue (requires X API Basic tier)
    try:
        print("Generating analytics digest...")
        post_analytics_digest(issue_number)
    except Exception as e:
        print(f"Analytics skipped (requires X API Basic tier): {e}")

    print("Done. Check your GitHub Issues.")

# --- GENERATE MODE: Run Sunday 9am ---
def run_generate():
    print("Mode: GENERATE")

    state = load_state()
    seen_highlights = state.get("seen_highlights", [])
    book_threads = state.get("book_threads", {})

    # --- Book tweets ---
    book_id, book_title, book_author, highlights = get_new_highlights(seen_highlights)

    if highlights:
        book_thread = book_threads.get(book_id, {})
        last_tweet_id = book_thread.get("last_tweet_id")
        is_continuing_thread = last_tweet_id is not None

        posts = generate_posts(highlights, book_title, book_author, is_continuing_thread)
        print(f"\nGenerated {len(posts)} book posts:")
        for i, p in enumerate(posts):
            print(f"{i+1}. {p}")

        standalone = generate_standalone_tweet(highlights, book_title, book_author)
        print(f"\nGenerated standalone tweet:\n{standalone}")

        state["pending_tweets"] = [
            {
                "order": i,
                "text": post,
                "posted": False,
                "tweet_id": None,
                "book_id": book_id,
                "book_title": book_title,
                "type": "book"
            }
            for i, post in enumerate(posts)
        ] + [
            {
                "order": len(posts),
                "text": standalone,
                "posted": False,
                "tweet_id": None,
                "book_id": book_id,
                "book_title": book_title,
                "type": "standalone"
            }
        ]
        state["seen_highlights"] = list(set(seen_highlights + highlights))
    else:
        print("No new book highlights this week.")

    # --- Interview tweets ---
    interview = state.get("interview", {})
    issue_number = interview.get("issue_number")
    questions = interview.get("questions", [])

    if issue_number and questions:
        comments = get_issue_comments(issue_number)
        user_comments = [c for c in comments if c.get("user", {}).get("type") != "Bot"]

        # Check for rating comment and update voice examples in state
        rating_comments = [c for c in user_comments if "RATING:" in c.get("body", "")]
        if rating_comments:
            rating_body = rating_comments[-1]["body"]
            rating_match = re.search(r"RATING:\s*([123])", rating_body)
            if rating_match:
                rating = int(rating_match.group(1))
                notes_match = re.search(r"NOTES:\s*(.+)", rating_body)
                notes = notes_match.group(1).strip() if notes_match else ""
                print(f"Found rating: {rating}/3. Notes: {notes}")

                # Store rating against last week's interview tweets
                rated_tweets = interview.get("pending_tweets", [])
                interview["last_rating"] = {"rating": rating, "notes": notes, "tweets": rated_tweets}

                # If rated 3, add the answer to VOICE_EXAMPLES in state for future use
                if rating == 3:
                    saved_examples = state.get("voice_examples", [])
                    last_answer = interview.get("last_answer", "")
                    last_tweet = rated_tweets[0]["text"] if rated_tweets else ""
                    if last_answer and last_tweet:
                        saved_examples.append({"a": last_answer, "tweet": last_tweet})
                        state["voice_examples"] = saved_examples[-5:]  # Keep best 5
                        print("High-rated answer saved to voice examples.")

        # Filter out rating comments to find actual answers
        answer_comments = [c for c in user_comments if "RATING:" not in c.get("body", "")]

        if answer_comments:
            answer_text = answer_comments[-1]["body"].strip()
            interview["last_answer"] = answer_text  # Save for potential voice example
            print(f"\nFound interview answer ({len(answer_text)} chars)")

            interview_tweets = generate_interview_tweets(questions, answer_text)
            print(f"Generated {len(interview_tweets)} interview tweets:")
            for i, t in enumerate(interview_tweets):
                print(f"{i+1}. {t}")

            interview["pending_tweets"] = [
                {
                    "order": i,
                    "text": tweet,
                    "posted": False,
                    "tweet_id": None,
                    "type": "interview"
                }
                for i, tweet in enumerate(interview_tweets)
            ]
        else:
            print("No answer found on interview issue this week. Skipping interview tweets.")

        print("\nGenerating reply suggestions...")
        generate_reply_suggestions(issue_number)

    else:
        print("No open interview issue. Skipping interview tweets.")

    state["interview"] = interview
    save_state(state)
    print("\nState saved. Book tweets post Monday. Interview tweets post Wednesday. Standalone tweet posts Thursday.")

# --- POST MODE: Run 3x on Monday (book tweets only) ---
def run_post():
    print("Mode: POST")

    state = load_state()
    pending = state.get("pending_tweets", [])
    book_threads = state.get("book_threads", {})

    unposted = [t for t in pending if not t.get("posted") and t.get("type") == "book"]

    if not unposted:
        print("No pending book tweets to post.")
        return

    tweet = unposted[0]
    book_id = tweet.get("book_id")
    book_title = tweet.get("book_title")
    tweet_text = tweet.get("text")
    tweet_order = tweet.get("order")

    book_thread = book_threads.get(book_id, {})
    last_tweet_id = book_thread.get("last_tweet_id")

    print(f"Posting book tweet {tweet_order + 1} for '{book_title}':")
    print(tweet_text)

    try:
        tweet_id = post_single_tweet(tweet_text, reply_to_id=last_tweet_id)
        print(f"Posted successfully. Tweet ID: {tweet_id}")

        for t in pending:
            if t["order"] == tweet_order:
                t["posted"] = True
                t["tweet_id"] = str(tweet_id)

        if book_id not in book_threads:
            book_threads[book_id] = {"title": book_title, "first_tweet_id": str(tweet_id)}
        book_threads[book_id]["last_tweet_id"] = str(tweet_id)

        state["pending_tweets"] = pending
        state["book_threads"] = book_threads
        save_state(state)

    except Exception as e:
        print(f"Failed to post tweet: {e}")
        raise

# --- POST_INTERVIEW MODE: Run 2x on Wednesday ---
def run_post_interview():
    print("Mode: POST_INTERVIEW")

    state = load_state()
    interview = state.get("interview", {})
    pending = interview.get("pending_tweets", [])

    unposted = [t for t in pending if not t.get("posted")]

    if not unposted:
        print("No pending interview tweets to post.")
        return

    tweet = unposted[0]
    tweet_text = tweet.get("text")
    tweet_order = tweet.get("order")

    reply_to_id = None
    if tweet_order > 0:
        prev = next((t for t in pending if t["order"] == tweet_order - 1), None)
        if prev:
            reply_to_id = prev.get("tweet_id")

    print(f"Posting interview tweet {tweet_order + 1}:")
    print(tweet_text)

    try:
        tweet_id = post_single_tweet(tweet_text, reply_to_id=reply_to_id)
        print(f"Posted successfully. Tweet ID: {tweet_id}")

        for t in pending:
            if t["order"] == tweet_order:
                t["posted"] = True
                t["tweet_id"] = str(tweet_id)

        interview["pending_tweets"] = pending
        state["interview"] = interview
        save_state(state)

        # After last interview tweet posts, add rating request to GitHub Issue
        remaining = [t for t in pending if not t.get("posted")]
        if not remaining:
            issue_number = interview.get("issue_number")
            if issue_number:
                rating_body = """## How were this week's tweets?

The 2 interview tweets just posted. Rate them so the writing improves over time:

**Reply with a single comment in this format:**
RATING: [1, 2, or 3]
NOTES: [optional - what worked or didn't]

1 = Missed the mark
2 = Good enough
3 = Nailed it

*Your rating gets used to calibrate future tweets to sound more like you.*"""
                url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{issue_number}/comments"
                requests.post(url, headers=github_headers(), json={"body": rating_body})
                print("Rating request posted to GitHub Issue.")

    except Exception as e:
        print(f"Failed to post interview tweet: {e}")
        raise

# --- POST_STANDALONE MODE: Run once Thursday 9am ---
def run_post_standalone():
    print("Mode: POST_STANDALONE")

    state = load_state()
    pending = state.get("pending_tweets", [])

    unposted = [t for t in pending if not t.get("posted") and t.get("type") == "standalone"]

    if not unposted:
        print("No standalone tweet to post this week.")
        return

    tweet = unposted[0]
    tweet_text = tweet.get("text")
    tweet_order = tweet.get("order")

    print(f"Posting standalone tweet:\n{tweet_text}")

    try:
        tweet_id = post_single_tweet(tweet_text)
        print(f"Posted successfully. Tweet ID: {tweet_id}")

        for t in pending:
            if t["order"] == tweet_order and t.get("type") == "standalone":
                t["posted"] = True
                t["tweet_id"] = str(tweet_id)

        state["pending_tweets"] = pending
        save_state(state)

    except Exception as e:
        print(f"Failed to post standalone tweet: {e}")
        raise

# --- Main ---
def main():
    print(f"Running at {datetime.now(timezone.utc).isoformat()}")
    print(f"Mode: {MODE}")

    if MODE == "generate":
        run_generate()
    elif MODE == "post":
        run_post()
    elif MODE == "interview_ask":
        run_interview_ask()
    elif MODE == "post_interview":
        run_post_interview()
    elif MODE == "post_standalone":
        run_post_standalone()
    else:
        raise Exception(f"Unknown mode: {MODE}")

if __name__ == "__main__":
    main()
