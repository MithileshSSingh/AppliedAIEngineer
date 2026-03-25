"""Generate synthetic product review dataset for sentiment analysis.

Creates reviews with clear sentiment signals for training, plus
challenging edge cases (sarcasm, mixed sentiment) for error analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_reviews(n_reviews: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic product reviews with sentiment labels."""
    np.random.seed(seed)

    # Templates for each sentiment
    positive_templates = [
        "Absolutely love this {product}! {reason}. Would definitely recommend.",
        "Great {product}, exactly what I needed. {reason}. Five stars!",
        "Best {product} I've ever bought. {reason}. Worth every penny.",
        "This {product} exceeded my expectations. {reason}.",
        "Fantastic quality {product}. {reason}. Very happy with my purchase.",
        "I'm so glad I bought this {product}. {reason}. Highly recommended!",
        "Amazing {product}! {reason}. My whole family loves it.",
        "Perfect {product} for the price. {reason}. Will buy again.",
        "This {product} is a game changer. {reason}. Can't imagine going back.",
        "Outstanding {product}. {reason}. Shipped fast too!",
    ]

    negative_templates = [
        "Terrible {product}. {reason}. Complete waste of money.",
        "Very disappointed with this {product}. {reason}. Returning it.",
        "Do NOT buy this {product}. {reason}. Wish I could give zero stars.",
        "This {product} broke after a week. {reason}. Horrible quality.",
        "Worst {product} I've ever purchased. {reason}. Total garbage.",
        "Save your money. This {product} is awful. {reason}.",
        "I regret buying this {product}. {reason}. Very frustrating.",
        "Cheap {product} that doesn't work as advertised. {reason}.",
        "This {product} is a scam. {reason}. Don't fall for it.",
        "Extremely poor quality {product}. {reason}. Very unhappy.",
    ]

    neutral_templates = [
        "The {product} is okay. {reason}. Nothing special but does the job.",
        "Decent {product} for the price. {reason}. It's alright.",
        "This {product} is average. {reason}. Not great, not terrible.",
        "The {product} works as expected. {reason}. No complaints but no excitement.",
        "It's a fine {product}. {reason}. Would consider other options next time.",
        "Mixed feelings about this {product}. {reason}. It has pros and cons.",
        "The {product} is satisfactory. {reason}. Gets the job done.",
        "An okay {product}. {reason}. Meets basic expectations.",
    ]

    products = ['headphones', 'laptop', 'phone case', 'backpack', 'water bottle',
                'keyboard', 'monitor', 'chair', 'desk lamp', 'tablet', 'speaker',
                'charger', 'mouse', 'webcam', 'microphone']

    positive_reasons = [
        "The quality is exceptional", "Works perfectly every time",
        "Great value for the price", "Beautiful design and sturdy build",
        "Super comfortable and easy to use", "Battery life is incredible",
        "Sound quality is crystal clear", "Setup was quick and simple",
        "Customer service was helpful", "Arrived earlier than expected",
    ]

    negative_reasons = [
        "The quality is terrible", "Stopped working after a few days",
        "Way overpriced for what you get", "Looks nothing like the pictures",
        "Extremely uncomfortable", "Battery dies in an hour",
        "Sound quality is awful", "Instructions were impossible to follow",
        "Customer service was useless", "Arrived damaged",
    ]

    neutral_reasons = [
        "The quality is acceptable", "Works most of the time",
        "Fair price for what it is", "Looks decent enough",
        "Comfortable enough for short use", "Battery is average",
        "Sound is passable", "Setup took some effort",
        "Haven't needed customer service", "Arrived on time",
    ]

    categories = ['Electronics', 'Accessories', 'Furniture', 'Audio', 'Computing']

    reviews = []
    sentiments = []
    product_list = []
    category_list = []

    # Distribution: 40% positive, 35% negative, 25% neutral
    sentiment_choices = np.random.choice(
        ['positive', 'negative', 'neutral'],
        n_reviews,
        p=[0.40, 0.35, 0.25]
    )

    for sentiment in sentiment_choices:
        product = np.random.choice(products)
        category = np.random.choice(categories)

        if sentiment == 'positive':
            template = np.random.choice(positive_templates)
            reason = np.random.choice(positive_reasons)
        elif sentiment == 'negative':
            template = np.random.choice(negative_templates)
            reason = np.random.choice(negative_reasons)
        else:
            template = np.random.choice(neutral_templates)
            reason = np.random.choice(neutral_reasons)

        review = template.format(product=product, reason=reason)

        # Add some noise
        if np.random.random() < 0.1:
            review = review.upper()  # SHOUTING
        if np.random.random() < 0.05:
            review = review.replace('.', '!!!').replace(',', '!!')  # Excessive punctuation
        if np.random.random() < 0.05:
            words = review.split()
            if len(words) > 3:
                idx = np.random.randint(1, len(words))
                words[idx] = words[idx][::-1]  # Typo: reversed word
                review = ' '.join(words)

        reviews.append(review)
        sentiments.append(sentiment)
        product_list.append(product)
        category_list.append(category)

    df = pd.DataFrame({
        'review_id': range(1, n_reviews + 1),
        'review': reviews,
        'sentiment': sentiments,
        'product': product_list,
        'category': category_list,
        'review_length': [len(r.split()) for r in reviews],
    })

    return df


def main():
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    df = generate_reviews()
    df.to_csv(data_dir / "reviews.csv", index=False)

    print(f"Generated {len(df)} reviews")
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    print(f"\nSaved to {data_dir / 'reviews.csv'}")


if __name__ == "__main__":
    main()
