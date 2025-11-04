import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.preprocessing import clean_text
def test_clean_text():
    assert clean_text("Hello!!! http://example.com") == "hello"
test_clean_text()
print("Test passed!")
