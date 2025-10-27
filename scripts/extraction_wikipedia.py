import wikipedia
from pathlib import Path

def extract_wikipedia_pages(page_titles):
    """
    Extracts Wikipedia pages and stores them in a dictionary.

    Args:
        page_titles: A list of Wikipedia page titles to extract.

    Returns:
        A dictionary containing the text of each Wikipedia page.
    """

    page_data = {}
    for title in page_titles:
        try:
            page = wikipedia.page(title)
            content = page.content.strip()
            page_data[page.title] = content
        except wikipedia.exceptions.PageError:
            print(f"Page '{title}' not found.")
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error for '{title}': {e.options}")

    return page_data

def main():
    # Define output directory
    output_dir = Path(__file__).resolve().parents[1] / "data" / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    page_titles = [
            "Albert Einstein",
            "Owen Willans Richardson",
            "Otto Sackur",
            "Ludvig Lorenz",
            "Klaus von Klitzing",
            "Henri Victor Regnault",
            "Erwin Madelung",
            "Gustav Kirchhoff",
            "Eric Allin Cornell",
            "Wolfgang Ketterle",
            "Raymond Davis Jr.",
            "Masatoshi Koshiba",
            "Riccardo Giacconi",
            "Alexei Alexeyevich Abrikosov",
            "Vitaly Ginzburg",
            "Anthony James Leggett",
            "David Gross",
            "Hugh David Politzer",
            "Frank Wilczek",
            "Roy J. Glauber",
            "John L. Hall",
            "Theodor W. Hänsch",
            "John C. Mather",
            "George Smoot",
            "Albert Fert",
            "Peter Grünberg",
            "Makoto Kobayashi"
        ]
    wikipedia_data = extract_wikipedia_pages(page_titles)
    
    # Save each page as a text file
    for title, content in wikipedia_data.items():
        # Create a safe filename from the title
        safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
        safe_filename = safe_filename.replace(' ', '_') + '.txt'
        
        output_path = output_dir / safe_filename
        output_path.write_text(content, encoding='utf-8')
        print(f"✅ Saved: {safe_filename}")
    
    print(f"\nDone. Saved {len(wikipedia_data)} Wikipedia pages in {output_dir}")
    
if __name__ == "__main__":
    main()

