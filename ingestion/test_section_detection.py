import unittest

from langchain_core.documents import Document

from ingestion.section_detector import (
    assign_sections,
    detect_section,
)


class TestSectionDetector(unittest.TestCase):

    def test_detect_numbered_section(self):
        result = detect_section(
            "1. Introduction"
        )

        self.assertEqual(
            result,
            "1. Introduction",
        )

    def test_detect_nested_section(self):
        result = detect_section(
            "1.2. Purpose and Scope"
        )

        self.assertEqual(
            result,
            "1.2. Purpose and Scope",
        )

    def test_non_section(self):
        result = detect_section(
            "This is ordinary paragraph text."
        )

        self.assertIsNone(result)

    def test_empty_line(self):
        self.assertIsNone(
            detect_section("")
        )

    def test_section_propagation(self):
        documents = [
            Document(
                page_content="1. Introduction\n"
                             "Some introductory text.",
                metadata={"page": 1},
            ),
            Document(
                page_content="More introductory text.",
                metadata={"page": 2},
            ),
            Document(
                page_content="1.1. Scope\n"
                             "Scope information.",
                metadata={"page": 3},
            ),
            Document(
                page_content="More scope information.",
                metadata={"page": 4},
            ),
        ]

        result = assign_sections(
            documents
        )

        self.assertEqual(
            result[0].metadata["section"],
            "1. Introduction",
        )

        self.assertEqual(
            result[1].metadata["section"],
            "1. Introduction",
        )

        self.assertEqual(
            result[2].metadata["section"],
            "1.1. Scope",
        )

        self.assertEqual(
            result[3].metadata["section"],
            "1.1. Scope",
        )

    def test_sections_on_page(self):
        document = Document(
            page_content=(
                "1. Introduction\n"
                "Some text.\n"
                "1.1. Scope\n"
                "More text."
            ),
            metadata={"page": 1},
        )

        result = assign_sections(
            [document]
        )

        self.assertEqual(
            result[0].metadata[
                "sections_on_page"
            ],
            [
                "1. Introduction",
                "1.1. Scope",
            ],
        )


if __name__ == "__main__":
    unittest.main()