import re
import html

def clean_text(text):

    # 1. Decode HTML entities
    text = html.unescape(text)

    # 2. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 3. Convert to lowercase
    text = text.lower()

    # 4. Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":

    sample = """
    THIS Movie is AMAZING!!! I loved it.
    <br /><br /> It was wonderful!
    """

    print("ORIGINAL TEXT:")
    print(sample)

    print("\nCLEANED TEXT:")
    print(clean_text(sample))
