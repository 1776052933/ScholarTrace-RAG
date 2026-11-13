"""Keep ambiguous table values attached to the cited table row."""

import pymupdf as fitz

from app.papers.pdf_preview import _matching_rectangles


def test_table_number_is_highlighted_in_its_row_not_matching_prose():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Transformer (big) 28.4 41.8")
    page.insert_text((72, 120), "Another paragraph also mentions 28.4 as a score")

    rects = _matching_rectangles(page, "| Model | EN-DE | EN-FR |\n"
                                      "| Transformer (big) | 28.4 | 41.8 |")
    number_positions = page.search_for("28.4")

    assert len(number_positions) == 2
    assert any(abs(rect.y0 - number_positions[0].y0) < 1 for rect in rects)
    assert not any(abs(rect.y0 - number_positions[1].y0) < 1 for rect in rects)
    document.close()
