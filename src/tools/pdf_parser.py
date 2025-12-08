import fitz  # PyMuPDF
import requests
from openpyxl import Workbook


def download_pdf(url: str):
    """Download a PDF from a URL and return a PyMuPDF document."""
    response = requests.get(url)
    response.raise_for_status()
    return fitz.open(stream=response.content, filetype="pdf")


def extract_sections_by_toc_with_hierarchy(doc):
    """
    Builds section hierarchy and returns list of:
    {
      "level": int,
      "title": str,
      "hier_title": str,
      "start_page": int,
      "end_page": int,
      "text": str,
      "description": dict-as-string,
    }
    """

    toc = doc.get_toc(simple=False)
    sections = []

    # Track hierarchy titles
    hierarchy = {}

    for i, entry in enumerate(toc):
        level, title, start_page = entry[:3]
        start_page -= 1  # Convert to 0-based index

        # Update hierarchy levels
        hierarchy[level] = title
        keys_to_delete = [k for k in hierarchy if k > level]
        for k in keys_to_delete:
            del hierarchy[k]

        # Build hierarchical title
        full_title = " → ".join(hierarchy[k] for k in sorted(hierarchy.keys()))

        # Compute end page of heirarchial block
        if i + 1 < len(toc):
            next_start = toc[i + 1][2] - 1
            end_page = next_start - 1
        else:
            end_page = doc.page_count - 1

        if end_page < start_page:
            end_page = start_page

        # Extract text from page range
        text = ""
        for p in range(start_page, end_page + 1):
            text += doc.load_page(p).get_text()

        # Clean text: single-line format like your scraped content
        clean_text = " ".join(text.split()).strip()

        description_value = {
            'title': full_title,
            'meta_description': full_title,
            'content': clean_text
        }

        # Append result
        sections.append({
            "level": level,
            "title": title,
            "hier_title": full_title,
            "start_page": start_page,
            "end_page": end_page,
            "status_code": 200,
            "error": "",
            "text": clean_text,
            "description": str(description_value)  # RAW python dict string
        })

    return sections


def pdf_to_excel(pdf_url: str, output_path: str):
    """Create Excel with URL, hierarchical title, pages, and description dict."""
    doc = download_pdf(pdf_url)
    sections = extract_sections_by_toc_with_hierarchy(doc)

    wb = Workbook()
    ws = wb.active
    ws.title = "PDF_Sections"

    # Excel header
    ws.append([
        "pdf_url",
        "hier_title",
        "title",
        "status_code",
        "error",
        "description",
        "start_page",
        "end_page"
    ])

    # Fill rows
    for sec in sections:
        ws.append([
            pdf_url,
            sec["hier_title"],
            sec["title"],
            sec["status_code"],
            sec["error"],
            sec["description"],
            sec["start_page"],
            sec["end_page"]
        ])

    wb.save(output_path)
    print(f"Excel successfully saved to {output_path}")


# Example usage
if __name__ == "__main__":
    pdf_url = "https://www.mb.studium.fau.de/files/2025/09/SF_MB_2025ws.pdf"
    output_excel = "pdf_sections_hierarchical.xlsx"
    pdf_to_excel(pdf_url, output_excel)
