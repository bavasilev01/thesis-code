import re
from collections import OrderedDict


class MIMIC_CXR_Processor:
    def __init__(self, medical_entities=None):
        self.medical_entities = medical_entities if medical_entities else {}
        self.section_headers = [
            "examination", "clinical history", "indication", "technique", 
            "comparison", "findings", "impression", "recommendation",
            "notification", "history" # Add other common headers if needed
        ]
        self.chexpert_columns = [
            "Atelectasis","Cardiomegaly","Consolidation","Edema",
            "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
            "Lung Opacity","No Finding","Pleural Effusion",
            "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
        ]
        # Allows for optional leading/trailing whitespace and variations in capitalization
        self.header_pattern = re.compile(r"^\s*(" + "|".join(self.section_headers) + r")\s*:", re.IGNORECASE)

    def _clean_section_text(self, text):
        """Cleans the text content of a single section."""
        # Remove leading/trailing whitespace
        text = text.strip()
        # Normalize whitespace (replace multiple spaces/newlines with a single space)
        text = re.sub(r'[ \t]+', ' ', text) # Replace multiple spaces/tabs with single space
        text = re.sub(r'\n+', ' ', text)
        return text

    def _normalize_header(self, header):
        """Normalizes a section header string."""
        normalized = header.upper().replace(" ", "_")
        # Handle potential synonyms (e.g., HISTORY -> INDICATION)
        if normalized == "HISTORY":
            return "INDICATION"
        return normalized

    def parse_report(self, text):
        """
        Parses the report text into sections based on identified headers.
        Filters sections to include only those plausible for VLM generation from an image.
        Returns an OrderedDict to maintain section order and the original text.
        """
        # Minimal initial cleaning: remove "FINAL REPORT" and surrounding whitespace
        original_text = text # Keep original for comparison
        text = re.sub(r'^\s*FINAL REPORT\s*', '', text, flags=re.IGNORECASE).strip()
        
        lines = text.splitlines()
        parsed_sections = OrderedDict()
        current_section_header = "PREAMBLE" # Content before the first recognized header
        current_section_content = []

        for line in lines:
            stripped_line_for_match = line.strip()
            if not stripped_line_for_match: # Skip empty lines between sections
                 # Add a newline marker if content exists, helps preserve paragraph breaks
                if current_section_content and current_section_content[-1] != '\n':
                    current_section_content.append('\n')
                continue

            match = self.header_pattern.match(stripped_line_for_match)
            if match:
                raw_header = match.group(1) # Get the matched header text
                # Found a new section header, store the previous section's content
                if current_section_content:
                    section_text = self._clean_section_text("\n".join(current_section_content))
                    if section_text:
                        # Use the normalized header of the previous section
                        parsed_sections[current_section_header] = section_text

                # Start the new section using the normalized header
                current_section_header = self._normalize_header(raw_header)
                # Capture content after the colon on the same line, if any
                content_after_colon = stripped_line_for_match[match.end():].strip()
                current_section_content = [content_after_colon] if content_after_colon else []
            else:
                # Line is part of the current section - use the original line
                current_section_content.append(line.strip())

        # Add the last section found
        if current_section_content:
            section_text = self._clean_section_text("\n".join(current_section_content))
            if section_text:
                 # Use the normalized header for the last section
                parsed_sections[current_section_header] = section_text
        
        # Filter out empty preamble if it exists and is truly empty after cleaning
        if "PREAMBLE" in parsed_sections and not parsed_sections["PREAMBLE"].strip():
            del parsed_sections["PREAMBLE"]

        # --- Filter for VLM-plausible sections ---
        # Choose findings and impression sections or only impression
        vlm_plausible_sections = {"FINDINGS", "IMPRESSION"}
        impression_section = {"IMPRESSION"}
        filtered_sections = OrderedDict()
        for header, content in parsed_sections.items():
            # Check against the plausible set
            if header in vlm_plausible_sections:
                filtered_sections[header] = content
                
        return filtered_sections, original_text
    
    def get_cleaned_report(self, report_text):
        sections, _ = self.parse_report(report_text)
        # Join the sections back into a cleaned report
        cleaned_report = []
        for header, content in sections.items():
            # Ensure header is a string and format correctly
            header_str = str(header).replace("_", " ").title()
            cleaned_report.append(f"{header_str}: {content}")
        cleaned_report = "\n".join(cleaned_report)
        return cleaned_report

    def get_structured_report_text(self, struct_data):
        """
        Generates a textual summary based on structured medical labels.

        Args:
            struct_data (dict): A dictionary where keys are medical labels (str)
                                and values are 1.0 (positive), 0.0 (negative),
                                -1.0 (uncertain), or NaN/None (missing).

        Returns:
            str: A generated text report summarizing the findings.
        """
        report_parts = []
        # Sort keys for consistent output order
        sorted_labels = sorted(struct_data.keys())

        positive_findings = []
        negative_findings = []
        uncertain_findings = []

        for label in sorted_labels:
            value = struct_data[label]
            # Format label for readability (e.g., 'Lung Opacity' -> 'lung opacity')
            formatted_label = label.replace('_', ' ').lower()

            # Check for Nan or No finding
            if value is None or formatted_label == "no finding":
                continue

            if value == 1.0:
                positive_findings.append(formatted_label)
            elif value == 0.0:
                negative_findings.append(formatted_label)
            elif value == -1.0:
                uncertain_findings.append(formatted_label)

        if positive_findings:
            report_parts.append(f"Evidence of: {', '.join(positive_findings)}.")
        if negative_findings:
            report_parts.append(f"No evidence of {', '.join(negative_findings)}.")
        if uncertain_findings:
            report_parts.append(f"Uncertain regarding {', '.join(uncertain_findings)}.")

        if not report_parts:
            return "No significant findings." # Or return empty string "" if preferred

        return " ".join(report_parts)
    
    def is_simple_report(self, report_text):
        # Check for sufficient length
        sufficient_len = False
        if report_text and report_text.strip() and len(report_text.split()) > 5:
            sufficient_len = True
        # Check for presence of references or depersonalized info markers like '_'
        no_references = True
        if re.search(r'_+', report_text):
            no_references = False
        # Check for presence of references to views
        no_views = True
        if re.search(r'PA|lateral', report_text):
            no_views = False
        
        has_findings_and_impression = any(
            section in report_text.lower() for section in ["findings", "impression"]
        )

        return sufficient_len and no_references and no_views
