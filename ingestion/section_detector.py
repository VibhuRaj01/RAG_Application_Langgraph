import re


SECTION_PATTERNS = [
    # Example: "1. Introduction"
    re.compile(
        r"^\s*(\d+(?:\.\d+)*)\.\s+"
        r"([A-Z][A-Za-z0-9 ,&()/'\-:]+)\s*$"
    ),

    # Example: "1.1 Purpose and Scope"
    re.compile(
        r"^\s*(\d+(?:\.\d+)+)\s+"
        r"([A-Z][A-Za-z0-9 ,&()/'\-:]+)\s*$"
    ),
]


def detect_section(text: str) -> str | None:
    """
    Detect a section heading from a line of text.

    Returns the complete heading when detected,
    otherwise None.
    """

    line = text.strip()

    if not line:
        return None

    for pattern in SECTION_PATTERNS:
        match = pattern.match(line)

        if match:
            return line

    return None


def assign_sections(documents):
    """
    Assign the most recently detected section to each document.

    Documents are expected to be in page order.
    """

    current_section = None

    for document in documents:
        lines = document.page_content.splitlines()

        detected_sections = []

        for line in lines:
            section = detect_section(line)

            if section:
                current_section = section
                detected_sections.append(section)

        document.metadata["section"] = (
            current_section
        )

        if detected_sections:
            document.metadata[
                "sections_on_page"
            ] = detected_sections

    return documents