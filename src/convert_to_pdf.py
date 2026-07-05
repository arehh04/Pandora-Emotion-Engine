import os
from markdown_pdf import MarkdownPdf
from markdown_pdf import Section

def convert_to_pdf():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = r"C:\Users\HP\.gemini\antigravity\brain\655fb615-fa2c-4fe8-b088-a5b416868079"
    
    chapter3_path = os.path.join(artifacts_dir, "Chapter3_Methodology.md")
    chapter4_path = os.path.join(artifacts_dir, "Chapter4_Results_Interpretability.md")
    
    out_dir = os.path.join(base_dir, "thesis_output")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, "FYP_Chapters_3_and_4.pdf")
    
    print("Reading Markdown files...")
    with open(chapter3_path, 'r', encoding='utf-8') as f:
        ch3_text = f.read()
        
    with open(chapter4_path, 'r', encoding='utf-8') as f:
        ch4_text = f.read()
        
    # Combine the text
    full_text = ch3_text + "\n\n<div style='page-break-after: always;'></div>\n\n" + ch4_text
    
    print("Generating PDF (this might take a few seconds)...")
    try:
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(full_text, toc=False))
        pdf.save(out_pdf)
        print(f"Success! PDF saved to: {out_pdf}")
    except Exception as e:
        print(f"Failed to generate PDF: {e}")
        print("Note: Markdown-to-PDF scripts often struggle with complex LaTeX math formulas.")
        print("If it failed or looks messy, open the .md files in VS Code or Chrome and click 'Print to PDF' for perfect academic formatting!")

if __name__ == "__main__":
    convert_to_pdf()
