"""Convert paper.tex -> paper.docx for collaborative prose/tone editing
(e.g. in Google Docs with a co-author).

The LaTeX stays the single source of truth for submission; this is a
working copy for a revision pass. Regenerate it whenever paper.tex
changes, and fold accepted edits back into paper.tex by hand (or send
the revised .docx to be reconciled) -- do not treat the .docx as canonical.

Run:
    uv run --with pypandoc_binary python paper/to_docx.py
"""
import os
import pypandoc

os.chdir(os.path.dirname(os.path.abspath(__file__)))

pypandoc.convert_file(
    "paper.tex",
    "docx",
    outputfile="paper.docx",
    extra_args=[
        "--citeproc",                  # \citep -> (Author Year) + a References list
        "--bibliography=references.bib",
        "--resource-path=figures",     # embed \includegraphics from figures/
        "--metadata=link-citations=true",
    ],
)
print("wrote paper/paper.docx")
