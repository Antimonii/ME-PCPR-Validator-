import re
import tempfile
import os
import math
from datetime import datetime, timedelta
import dateparser
from PIL import Image
import pytesseract

import fitz  
import pandas as pd
import streamlit as st
import tabula

import os
import cv2
import fitz  
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image
import re
import warnings
from functools import lru_cache
from typing import Optional, List, Dict, Tuple, Union

# Set JAVA_HOME for Tabula
# Ensure this path is correct for your environment if running locally
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-23.0.2"
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\aantony\Downloads\tesseract.exe"


# ---------- Normalization Function ----------
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace(",", "")  # <- Already removes commas
    return re.sub(r'[^a-z0-9]', '', text.lower())


# ---------- Date Validation Function ----------
def is_date_valid(cert_date_str, insp_date_str):
    """
    Returns True if the inspection date is within 2 years of the certificate date.
    Both dates should be in the format DD/MM/YYYY.
    """
    try:
        cert_date = datetime.strptime(cert_date_str, "%d/%m/%Y")
        insp_date = datetime.strptime(insp_date_str, "%d/%m/%Y")
        two_years_later = cert_date.replace(year=cert_date.year + 2)
        return cert_date <= insp_date <= two_years_later, two_years_later
    except (ValueError, TypeError) as e:
        st.error(f"Date parsing error in validation: {e}. Please ensure dates are in DD/MM/YYYY format.")
        return False, None  # Return False and None for the date if an error occurs

# ---------- Date Extraction from MMT PDF ----------
def extract_cert_date_from_mmt(mmt_file_buffer):
    """
    Extracts a likely 'Certificate Date' or 'Verification Date' from the MMT PDF.
    Uses dateparser for flexible parsing of various date formats, including French.
    """
    try:
        mmt_file_buffer.seek(0)
        pdf_document = fitz.open(stream=mmt_file_buffer.read(), filetype="pdf")
        full_mmt_text = ""
        for page_num in range(min(pdf_document.page_count, 3)):
            full_mmt_text += pdf_document[page_num].get_text()
        #hopefully the datefromat is wihtin these 
        date_patterns = [
            # New pattern for DD-mon-YY (e.g., 1-déc-21, 29 janv 2019)
            r'(\d{1,2}[-/\s]?(?:jan|fév|mar|avr|mai|jun|jul|aoû|sep|oct|nov|déc|janv|févr|sept|octb)\.?-?\d{2,4})',
            r'(?:verification date|Date of the cert|Edition F -)\s*:\s*\n?([\w\s./-]+)',
            r'Date\s*:\s*\n?([\w\s./-]+)', # General Date
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', # DD-MM-YY, DD/MM/YYYY
            r'(\d{1,2}\s+\w+\s+\d{4})' #29 Janvier 2019
        ]

        for pattern in date_patterns:
            match = re.search(pattern, full_mmt_text, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                parsed_date = dateparser.parse(
                    date_str,
                    settings={'DATE_ORDER': 'DMY', 'REQUIRE_PARTS': ['day', 'month', 'year']},
                    languages=['en', 'fr'] 
                )
                if parsed_date:
                    return parsed_date.strftime("%d/%m/%Y")

        st.warning("Certificate date pattern not found in MMT PDF.")
        return None
    except Exception as e:
        st.error(f"Error extracting certificate date from MMT PDF: {e}")
        return None

# ---------- Inspection Date Extraction from MR PDF ----------
def extract_inspection_date_from_MR(MR_file_buffer):
    """
    Extracts the 'Inspection date' from the MR PDF using dateparser for flexibility.
    """
    try:
        MR_file_buffer.seek(0)
        pdf_document = fitz.open(stream=MR_file_buffer.read(), filetype="pdf")
        full_MR_text = ""
        for page_num in range(pdf_document.page_count):
            full_MR_text += pdf_document[page_num].get_text()

        # Pattern DD/MM/YYYY date
        insp_date_pattern = r'Inspection date\s*:\s*\n?(\d{2}/\d{2}/\d{4})'
        match = re.search(insp_date_pattern, full_MR_text, re.IGNORECASE)
        if match:
            date_str = match.group(1).strip()
            parsed_date = dateparser.parse(
                date_str,
                settings={'DATE_ORDER': 'DMY', 'REQUIRE_PARTS': ['day', 'month', 'year']}
            )
            if parsed_date:
                return parsed_date.strftime("%d/%m/%Y")
        st.warning("Inspection date pattern 'Inspection date :' not found in MR PDF.")
        return None
    except Exception as e:
        st.error(f"Error extracting inspection date from MR PDF: {e}")
        return None

# ---------- Full Text Extraction from PDF ----------
def get_full_text_from_pdf(pdf_file_buffer):
    """Extracts all text from a PDF file into a single string for searching."""
    try:
        pdf_file_buffer.seek(0)
        pdf_document = fitz.open(stream=pdf_file_buffer.read(), filetype="pdf")
        full_text = "\n".join([page.get_text() for page in pdf_document])
        return full_text
    except Exception as e:
        st.error(f"Error extracting full text from PDF: {e}")
        return None
# ---------- Extract QA-ID from Image in PDF ----------
# THIS DOES NOT SEEM TO BE WOKRING UNDFORTUENATELY, SO IT IS NOT USED IN THE APP
def extract_qa_id_from_image(pdf_buffer):
    pdf_buffer.seek(0)
    doc = fitz.open(stream=pdf_buffer.read(), filetype="pdf")
    pix = doc[0].get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    ocr_text = pytesseract.image_to_string(img)
    match = re.search(r'QA[-\s]?ID\s*[:：]?\s*([A-Z0-9/]+)', ocr_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# ---------- Header Data Extraction from MR PDF  ----------
def extract_identifiers_from_MR(MR_file_buffer):
    """
    Extracts HTZ, Tool description, ID-Nr, Issue, QA-ID, and Anf from a tabular structure in the MR PDF.
    """
    temp_pdf_path = None
    extracted_data = {}

    try:
        MR_file_buffer.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(MR_file_buffer.read())
            temp_pdf_path = temp_file.name

        tables = tabula.read_pdf(temp_pdf_path, pages="all", multiple_tables=True, encoding="latin1")


        if not tables or tables[0].empty:
            st.warning("No table extracted from MR PDF for identifiers.")
        else:
            df = tables[0]

            for index, row in df.iterrows():
                for col in ['Unnamed: 0', 'Initial inspection', 'X']:
                    if col not in row or pd.isna(row[col]):
                        continue
                    value = str(row[col]) 
                    if 'HTZ' in value:
                        extracted_data['HTZ'] = value.split('HTZ :')[-1].strip()
                    elif 'Tool description' in value:
                        extracted_data['Tool Description'] = value.split('Tool description :')[-1].strip()
                    elif 'Issue' in value:
                        extracted_data['Issue'] = value.split('Issue :')[-1].strip()
                    elif 'QA-ID' in value:
                        extracted_data['QA-ID'] = value.split('QA-ID :')[-1].strip()
                    elif 'Anf.:' in value:
                        parts = value.split('Anf.:')
                        if len(parts) > 1 and parts[-1].strip():
                            extracted_data['Anf.'] = parts[-1].strip()
                    elif 'ID-Nr' in value:
                        extracted_data['ID-Nr'] = value.split('ID-Nr :')[-1].strip()

        # Fallback
        if not extracted_data.get('QA-ID') or not extracted_data.get('Anf.'):
            try:
                MR_file_buffer.seek(0)
                with fitz.open(stream=MR_file_buffer.read(), filetype="pdf") as doc:
                    raw_text = "\n".join([page.get_text() for page in doc])

                if not extracted_data.get('QA-ID'):
                    qa_match = re.search(r'QA[-\s]?ID\s*:?\s*([A-Z0-9/]+)', raw_text, re.IGNORECASE)
                    if qa_match:
                        extracted_data['QA-ID'] = qa_match.group(1).strip()
                    else: 
                        qa_ocr = extract_qa_id_from_image(MR_file_buffer)
                        if qa_ocr:
                            extracted_data['QA-ID'] = qa_ocr

                if not extracted_data.get('Anf.'):
                    anf_match = re.search(r'Anf\.?\s*[:：]?\s*([A-Z0-9]+)', raw_text, re.IGNORECASE)
                    if anf_match:
                        extracted_data['Anf.'] = anf_match.group(1).strip()
            except Exception as e:
                st.warning(f"Fallback text scan failed: {e}")

        return extracted_data

    except Exception as e:
        st.error(f"Error extracting identifiers from MR using Tabula: {e}")
        return {}
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

# ---------- Load and Process Excel Data for Cross-Check ----------
def load_and_process_excel_data(excel_file_buffer):
    """
    Loads Excel/CSV data and applies necessary normalization for cross-checking.
    """
    try:
        excel_file_buffer.seek(0)
        file_extension = os.path.splitext(excel_file_buffer.name)[1].lower()

        if file_extension == '.csv':
            df = pd.read_csv(excel_file_buffer)
        else: 
            df = pd.read_excel(excel_file_buffer)
        
        df.columns = [col.strip() for col in df.columns]

        if 'HTZ neu' in df.columns:
            df['HTZ neu_normalized'] = df['HTZ neu'].astype(str).apply(normalize_text)
        else:
            st.warning("Excel file missing 'HTZ neu' column.")
            df['HTZ neu_normalized'] = ''

        if 'Template Ident-No.' in df.columns:
            df['Template Ident-No._normalized'] = df['Template Ident-No.'].astype(str).apply(lambda x: normalize_text(x).rstrip("0"))
        else:
            st.warning("Excel file missing 'Template Ident-No.' column.")
            df['Template Ident-No._normalized'] = ''

        if 'Anf. (Anfertigung)' in df.columns:
            df['Anf. (Anfertigung)_normalized'] = df['Anf. (Anfertigung)'].astype(str).str.replace('Anf\\.\\s*', '', regex=True).apply(normalize_text)
        else:
            st.warning("Excel file missing 'Anf. (Anfertigung)' column.")
            df['Anf. (Anfertigung)_normalized'] = ''

        if 'QA identification' in df.columns:
            df['QA identification_normalized'] = df['QA identification'].astype(str).apply(normalize_text)
        else:
            st.warning("Excel file missing 'QA identification' column.")
            df['QA identification_normalized'] = ''

        if 'Eindeutigkeit' not in df.columns:
            st.warning("Excel file missing 'Eindeutigkeit' column. Final output might be incomplete.")
            df['Eindeutigkeit'] = pd.NA

        return df
    except Exception as e:
        st.error(f"Error loading or processing Excel/CSV file: {e}")
        return pd.DataFrame()


# ---------- Coordinate and Tolerance Extraction from Diagram PDF ----------
def extract_coordinates_from_pdf(pdf_file):
    try:
        pdf_file.seek(0)
        full_text = get_full_text_from_pdf(pdf_file)
        if not full_text: return None

        coord_pattern = r"A - Coordinate(.*?)B - Coordinate(.*?)F - Coordinate(.*?)Signature"
        match = re.search(coord_pattern, full_text, re.DOTALL)
        if not match: return None

        coordinate_sections = {
            "A - Coordinate": match.group(1).strip(), "B - Coordinate": match.group(2).strip(), "F - Coordinate": match.group(3).strip(),
        }
        coordinates_data = {}
        line_pattern = r"(\d+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)"
        for label, section_text in coordinate_sections.items():
            lines = [line for line in section_text.splitlines() if "Ø" not in line]
            matches = re.findall(line_pattern, "\n".join(lines))
            if matches:
                df = pd.DataFrame(matches, columns=["REF.", "X", "Y", "Z"])
                coordinates_data[label] = df
        
        tolerance_pattern = r"POINTS\s+(A|F)\s*:\s*(Ø\d+\.\d+)"
        tolerance_matches = re.findall(tolerance_pattern, full_text)
        tolerance_dict = {point: value for point, value in tolerance_matches}
        return {"coordinates": coordinates_data, "tolerances": tolerance_dict} if coordinates_data and tolerance_dict else None
    except Exception as e:
        st.error(f"Error extracting coordinates from Diagram PDF: {e}")
        return None

# ---------- Dimension Table Extraction from MR PDF ----------
def extract_dimension_tables_from_MR(MR_file_buffer):
    temp_file_path = None
    try:
        MR_file_buffer.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(MR_file_buffer.read())
            temp_file_path = temp_file.name

        tables = tabula.read_pdf(temp_file_path, pages="all", multiple_tables=True, encoding="latin1")
        def contains_dimensions(table):
            if table.empty: return False
            return (table.columns.str.contains("DIMENSIONS", case=False, na=False).any() or
                            table.apply(lambda col: col.astype(str).str.contains("DIMENSIONS", case=False, na=False)).any().any())
        filtered_tables = [t for t in tables if contains_dimensions(t)]
        extracted_MR_tables = {"A": {}, "B": {}, "F": {}}
        for table in filtered_tables:
            if "DIMENSIONS" not in table.columns:
                if len(table) > 1:
                    table.columns = table.iloc[0]
                    table = table.drop(0).reset_index(drop=True)
                else: continue
            if "DIMENSIONS" not in table.columns: continue
            table_copy = table.copy()
            table_copy["DIMENSIONS"] = table_copy["DIMENSIONS"].bfill()
            table_copy["DIMENSIONS"] = table_copy["DIMENSIONS"].apply(
                lambda x: " ".join([word for word in str(x).split() if any(c in word for c in "ABF")]))
            table_copy = table_copy.loc[:, table_copy.columns.notna()]
            table_copy = table_copy.loc[:, ~table_copy.columns.astype(str).str.match(r"^Unnamed|^\s*$")]
            for label in ["A", "B", "F"]:
                sub_table = table_copy[table_copy["DIMENSIONS"].str.startswith(label)].reset_index(drop=True)
                if not sub_table.empty:
                    dim = str(sub_table["DIMENSIONS"].iloc[0])
                    table_for_ref = dim[1:]
                    extracted_MR_tables[label][table_for_ref] = sub_table
        return extracted_MR_tables
    except Exception as e:
        st.error(f"Error reading MR PDF for tables: {e}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# ---------- Sequential Cross-Check Logic ----------
def perform_sequential_cross_check(excel_df, MR_identifiers):
    """
    Performs sequential cross-checking based on HTZ, ID-Nr, Anf., and QA-ID.
    Returns the Eindeutigkeit value if a unique match is found, along with a status message.
    """
    df_filtered = excel_df.copy()
    result_eindeutigkeit = None
    status_message = "No unique match found for the given criteria chain." # Default failure message

    # --- Step 1: HTZ Cross-Check ---
    st.markdown("##### 1. Checking HTZ Number")
    if 'HTZ' in MR_identifiers and 'HTZ neu_normalized' in df_filtered.columns:
        MR_htz_norm = normalize_text(MR_identifiers['HTZ'])
        
        current_matches = df_filtered[df_filtered['HTZ neu_normalized'] == MR_htz_norm]
        
        if len(current_matches) == 1:
            result_eindeutigkeit = current_matches['Eindeutigkeit'].iloc[0]
            status_message = f"✅ Unique match found on **HTZ**. Eindeutigkeit: `{result_eindeutigkeit}`"
            return result_eindeutigkeit, status_message, current_matches.iloc[0]
        elif len(current_matches) > 1:
            df_filtered = current_matches.copy()
            st.info(f"Multiple HTZ matches ({len(df_filtered)}). Proceeding to next check (ID-Nr.).")
        else:
            status_message = f"❌ No match found for HTZ: `{MR_identifiers['HTZ']}` in Excel."
            return None, status_message, None
    else:
        status_message = "⚠️ HTZ not found in MR data or corresponding column missing in Excel. Skipping HTZ check."
        if 'HTZ' not in MR_identifiers:
            status_message = "❌ HTZ identifier missing from MR data. Cannot perform full sequential cross-check."
            return None, status_message, None
        if 'HTZ neu_normalized' not in df_filtered.columns:
            status_message = "❌ 'HTZ neu' column missing in Excel data. Cannot perform HTZ cross-check."
            return None, status_message, None


    # --- Step 2: Template Ident-No.  ---
    st.markdown("##### 2. Checking Template Ident-No. / ID-Nr")
    if len(df_filtered) > 1 and 'ID-Nr' in MR_identifiers and 'Template Ident-No._normalized' in df_filtered.columns:
        MR_id_nr_norm = normalize_text(MR_identifiers['ID-Nr'])
        
        current_matches = df_filtered[df_filtered['Template Ident-No._normalized'].str.startswith(MR_id_nr_norm)]
        st.write(f"🔍 Normalized ID-Nr from MR: `{MR_id_nr_norm}`")
        if len(current_matches) == 1:
            result_eindeutigkeit = current_matches['Eindeutigkeit'].iloc[0]
            status_message = f"✅ Unique match found on **ID-Nr**. Eindeutigkeit: `{result_eindeutigkeit}`"
            return result_eindeutigkeit, status_message, current_matches.iloc[0]
        elif len(current_matches) > 1:
            df_filtered = current_matches.copy()
            st.info(f"Multiple ID-Nr matches ({len(df_filtered)}). Proceeding to next check (Anf.).")
        else:
            status_message = f"❌ No match found for ID-Nr: `{MR_identifiers['ID-Nr']}` in remaining Excel data."
            return None, status_message,None
    elif len(df_filtered) == 0: 
        return None, status_message, None
    else: 
        if 'ID-Nr' not in MR_identifiers:
            st.warning("ID-Nr not found in MR data. Skipping ID-Nr check.")
        if 'Template Ident-No._normalized' not in df_filtered.columns:
            st.warning("'Template Ident-No.' column missing in Excel data. Skipping ID-Nr cross-check.")
        if len(df_filtered) == 1: 
            result_eindeutigkeit = df_filtered['Eindeutigkeit'].iloc[0]
            status_message = f"✅ Unique match found earlier. Eindeutigkeit: `{result_eindeutigkeit}`"
            return result_eindeutigkeit, status_message, None
        else: 
            status_message = "❌ Cannot proceed with ID-Nr check due to missing data or previous filtering."
            return None, status_message, None


    # --- Step 3: Anf. (Anfertigung) Cross-Check ---
    st.markdown("##### 3. Checking Anf. (Anfertigung)")
    if len(df_filtered) > 1 and 'Anf.' in MR_identifiers and 'Anf. (Anfertigung)_normalized' in df_filtered.columns:
        raw_anf = MR_identifiers.get('Anf.', '').strip()
        if not raw_anf:
            st.warning("⚠️ 'Anf.' is empty or missing from MR. Skipping Anf. check.")
            return None, "❌ Anf. value missing in MR identifiers."
        MR_anf_norm = normalize_text(raw_anf)

        
        current_matches = df_filtered[df_filtered['Anf. (Anfertigung)_normalized'] == MR_anf_norm]

        if len(current_matches) == 1:
            result_eindeutigkeit = current_matches['Eindeutigkeit'].iloc[0]
            status_message = f"✅ Unique match found on **Anf.**. Eindeutigkeit: `{result_eindeutigkeit}`"
            return result_eindeutigkeit, status_message, current_matches.iloc[0]
        elif len(current_matches) > 1:
            df_filtered = current_matches.copy()
            st.info(f"Multiple Anf. matches ({len(df_filtered)}). Proceeding to next check (QA-ID).")
        else:
            status_message = f"❌ No match found for Anf.: `{MR_identifiers['Anf.']}` in remaining Excel data."
            return None, status_message,None
    elif len(df_filtered) == 0: 
        return None, status_message,None
    else: 
        if 'Anf.' not in MR_identifiers:
            st.warning("Anf. not found in MR data. Skipping Anf. check.")
        if 'Anf. (Anfertigung)_normalized' not in df_filtered.columns:
            st.warning("'Anf. (Anfertigung)' column missing in Excel data. Skipping Anf. cross-check.")
        if len(df_filtered) == 1:
            result_eindeutigkeit = df_filtered['Eindeutigkeit'].iloc[0]
            status_message = f"✅ Unique match found earlier. Eindeutigkeit: `{result_eindeutigkeit}`"
            return result_eindeutigkeit, status_message,df_filtered.iloc[0]
        else:
            status_message = "❌ Cannot proceed with Anf. check due to missing data or previous filtering."
            return None, status_message,None

    # --- Step 4: QA identification / QA-ID Cross-Check (Final Step) ---
    st.markdown("##### 4. Checking QA identification / QA-ID")
    if len(df_filtered) > 0 and 'QA-ID' in MR_identifiers and 'QA identification_normalized' in df_filtered.columns:
        MR_qa_id_norm = normalize_text(MR_identifiers['QA-ID'])
        
        current_matches = df_filtered[df_filtered['QA identification_normalized'] == MR_qa_id_norm]

        if len(current_matches) == 1:
            result_eindeutigkeit = current_matches['Eindeutigkeit'].iloc[0]
            status_message = f"✅ Unique match found on **QA-ID**. Eindeutigkeit: `{result_eindeutigkeit}`"
            return result_eindeutigkeit, status_message,current_matches.iloc[0]
        elif len(current_matches) > 1:
            status_message = f"⚠️ Ambiguous match: Multiple records ({len(current_matches)}) found even after all criteria."
            result_eindeutigkeit = current_matches['Eindeutigkeit'].iloc[0]
            status_message += f" Returning Eindeutigkeit from first match: `{result_eindeutigkeit}`"
            return result_eindeutigkeit, status_message, current_matches.iloc[0]
        else:
            status_message = f"❌ No match found for QA-ID: `{MR_identifiers['QA-ID']}` in remaining Excel data."
            return None, status_message,None
    elif len(df_filtered) == 0: 
        return None, status_message,None
    else: 
        if 'QA-ID' not in MR_identifiers:
            st.warning("QA-ID not found in MR data. Skipping QA-ID check.")
        if 'QA identification_normalized' not in df_filtered.columns:
            st.warning("'QA identification' column missing in Excel data. Skipping QA-ID cross-check.")
        if len(df_filtered) == 1:
            result_eindeutigkeit = df_filtered['Eindeutigkeit'].iloc[0]
            status_message = f"✅ Unique match found earlier. Eindeutigkeit: `{result_eindeutigkeit}`"
            return result_eindeutigkeit, status_message, df_filtered.iloc[0]
        else: 
            status_message = "❌ Cannot complete QA-ID check due to missing data or previous filtering."
            return None, status_message,None

    return None, status_message

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

keras_ocr = None

TESSERACT_MAX_DIM = 32000
KERAS_OCR_MAX_DIM = 2048
DEFAULT_DPI = 300
OCR_CONFIG = r'--oem 3 --psm 6'

@lru_cache(maxsize=1)
def get_metadata_patterns() -> Dict[str, re.Pattern]:
    """Get compiled regex patterns for metadata extraction."""
    patterns = {
        "ST": r"ST\s*[:\-]?\s*([A-Z0-9 ./-]+)",
        "ID-N°": r"ID[-\s]*N[°O0]?\s*[:\-]?\s*([A-Z0-9]+)",
        "QA-ID": r"QA[-\s]*ID\s*[:\-]?\s*([A-Z0-9/]+)",
        "HTZ": r"HTZ\s*[:\-]?\s*([A-Z0-9\-]+)",
        "ISSUE": r"ISSUE\s*[:\-]?\s*([A-Z])",
        "ANF": r"ANF\s*N[°O0]?\s*[:\-]?\s*([A-Z0-9]*)",
        "EQ-N°": r"EQ[-\s]*N[°O0]?\s*[:\-]?\s*([A-Z0-9]*)"
    }
    return {key: re.compile(pattern, re.IGNORECASE) for key, pattern in patterns.items()}

def load_keras_ocr() -> Union[object, bool]:
    """
    Lazy loading of keras-ocr with compatibility fixes.
    Maintains original function name from provided code.
    """
    global keras_ocr
    if keras_ocr is None:
        try:
            import keras_ocr as ko
            keras_ocr = ko
            print("  📚 keras-ocr loaded successfully")
            return keras_ocr
        except ImportError as e:
            print(f"  ⚠️ keras-ocr not available: {e}")
            print("  ⚠️ Line removal will be skipped.")
            keras_ocr = False
            return False
        except Exception as e:
            print(f"  ⚠️ keras-ocr compatibility issue: {e}")
            print("  ⚠️ Falling back to standard OCR processing")
            keras_ocr = False
            return False
    return keras_ocr

def resize_image_if_needed(image: np.ndarray, max_dim: int, 
                          interpolation: int = cv2.INTER_LANCZOS4) -> Tuple[np.ndarray, bool]:
    """
    Resize image if it exceeds maximum dimensions.
    
    Returns:
        Tuple of (resized_image, was_resized)
    """
    height, width = image.shape[:2]
    
    if height <= max_dim and width <= max_dim:
        return image, False
    
    if height > width:
        new_height = max_dim
        new_width = int(width * (max_dim / height))
    else:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    print(f"  ℹ️ Resized image: {width}x{height} -> {new_width}x{new_height}")
    return resized, True

def enhance_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Apply basic image enhancements for better OCR performance.
    Optimized version with fewer operations.
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return cleaned
        
    except Exception as e:
        print(f"  ⚠️ Image enhancement failed: {e}")
        return image

def enhanced_opencv_line_removal(binary_image: np.ndarray) -> np.ndarray:
    """
    Enhanced OpenCV-based line removal with optimized parameters.
    """
    try:
        result = binary_image.copy()
        
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, h_kernel, iterations=2)
        
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, v_kernel, iterations=2)
        
        detected_lines = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0.0)
        result_no_lines = cv2.subtract(result, detected_lines)
        
        cleanup_kernel = np.ones((2, 2), np.uint8)
        result_clean = cv2.morphologyEx(result_no_lines, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)
        
        print("  ✓ Enhanced OpenCV line removal completed")
        return result_clean
        
    except Exception as e:
        print(f"  ⚠️ Enhanced OpenCV processing failed: {e}")
        return binary_image

def remove_lines_with_keras_ocr(image: np.ndarray, keras_ocr_lib) -> np.ndarray:
    """
    Remove lines from image using keras-ocr with improved error handling.
    Maintains original function name from provided code.
    """
    try:
        if not keras_ocr_lib or keras_ocr_lib is False:
            raise ValueError("keras-ocr library not properly loaded")
        
        pipeline = keras_ocr_lib.pipeline.Pipeline()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        rgb_image, was_resized = resize_image_if_needed(rgb_image, KERAS_OCR_MAX_DIM, cv2.INTER_AREA)
        original_shape = image.shape[:2]       
        prediction_groups = pipeline.recognize([rgb_image])        
        mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        for predictions in prediction_groups:
            for word, box in predictions:
                poly = np.array(box, np.int32)
                cv2.fillPoly(mask, [poly], 255)
        
        result = cv2.inpaint(rgb_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        if was_resized and result.shape[:2] != original_shape:
            result = cv2.resize(result, (original_shape[1], original_shape[0]), 
                              interpolation=cv2.INTER_LANCZOS4)
        
        print("  ✓ keras-ocr line removal completed successfully")
        return result
        
    except Exception as e:
        print(f"  ⚠️ keras-ocr line removal failed: {e}")
        return image
def should_use_keras_ocr(image: np.ndarray) -> bool:
    """
    Determine if keras-ocr should be used based on line analysis.
    Optimized with faster edge detection.
    """
    try:
        keras_ocr_lib = load_keras_ocr()
        if not keras_ocr_lib or keras_ocr_lib is False:
            return False
        
        small_image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
        gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        
        line_count = len(lines) if lines is not None else 0
        use_keras = line_count > 15  
        
        analysis_result = 'keras-ocr' if use_keras else 'standard OCR'
        print(f"  📊 Line analysis: {line_count} lines detected - using {analysis_result}")
        return use_keras
        
    except Exception as e:
        print(f"  ⚠️ Line analysis failed: {e} - using standard OCR")
        return False

def preprocess_image_for_ocr(image: np.ndarray, use_keras_ocr: bool = False) -> np.ndarray:
    """
    Advanced image preprocessing with optional keras-ocr line removal.
    Maintains original function name from provided code.
    """
    try:
        result_image = image.copy()       
        enhanced = enhance_image_for_ocr(result_image)
        
        if use_keras_ocr:
            keras_ocr_lib = load_keras_ocr()
            if keras_ocr_lib and keras_ocr_lib is not False:
                try:
                    result = remove_lines_with_keras_ocr(result_image, keras_ocr_lib)
                    if len(result.shape) == 3:
                        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    else:
                        result = enhanced
                except Exception as e:
                    print(f"  ⚠️ keras-ocr failed, using standard processing: {e}")
                    result = enhanced_opencv_line_removal(enhanced)
            else:
                result = enhanced_opencv_line_removal(enhanced)
        else:
            result = enhanced_opencv_line_removal(enhanced)
        if len(result.shape) == 2:
            result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            result_bgr = result
        
        return result_bgr
        
    except Exception as e:
        print(f"  ⚠️ Preprocessing failed: {e}")
        return image

def extract_metadata_from_image_basic(image: np.ndarray) -> Optional[Dict[str, str]]:
    """
    Extract metadata from image using optimized OCR processing.
    """
    try:
        patterns = get_metadata_patterns()        
        processed_image, _ = resize_image_if_needed(image, TESSERACT_MAX_DIM)        
        enhanced = enhance_image_for_ocr(processed_image)
        
        try:
            text = pytesseract.image_to_string(enhanced, config=OCR_CONFIG)
        except Exception as e:
            print(f"  ⚠️ Basic OCR failed: {e}")
            return None
        
        text = re.sub(r'[\n\x0c]', ' ', text).strip()
        text = re.sub(r'\s{2,}', ' ', text)
        
        parsed_data = {}
        for key, pattern in patterns.items():
            match = pattern.search(text)
            parsed_data[key] = match.group(1).strip() if match else None
        
        excluded_keys = {"ST", "ANF", "EQ-N°"}
        parsed_data = {k: v for k, v in parsed_data.items() if k not in excluded_keys}
        
        if parsed_data.get("ID-N°"):
            parsed_data["ID-N°"] = parsed_data["ID-N°"].replace("QA", "")
        
        if parsed_data.get("QA-ID"):
            parsed_data["QA-ID"] = parsed_data["QA-ID"].replace("7", "/").replace("ISSUE", "").strip()

        if parsed_data.get("HTZ"):
            parsed_data["HTZ"] = parsed_data["HTZ"].replace("ISSUE", "")
        
        return parsed_data if any(v for v in parsed_data.values() if v) else None
            
    except Exception as e:
        print(f"  ⚠️ Metadata extraction failed: {e}")
        return None

def save_images_to_pdf(image_data_list: List[Dict], output_pdf_path: str) -> bool:
    """
    Save images to PDF with better error handling.
    """
    if not image_data_list:
        print("Warning: No images provided to save to PDF.")
        return False
    
    try:
        pil_images = []
        
        for item in image_data_list:
            img_np = item['image']
            
            # RGB
            if len(img_np.shape) == 2:   
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else:  
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            pil_images.append(Image.fromarray(img_rgb))
        
        if not pil_images:
            print("Warning: No valid images to save to PDF.")
            return False
        
        print(f"\nSaving {len(pil_images)} images to {output_pdf_path}...")
        
        # Save PDF
        pil_images[0].save(
            output_pdf_path, "PDF", resolution=100.0,
            save_all=True, append_images=pil_images[1:]
        )
        
        print(f"  ✅ PDF saved successfully: {output_pdf_path}")
        return True
        
    except Exception as e:
        print(f"  ⚠️ Failed to save PDF: {e}")
        return False

def filter_images_with_metadata(image_list: List[Dict]) -> List[Dict]:
    """
    Filter images to only include those containing metadata.
    """
    filtered_images = []
    
    print("\n=== Filtering images based on metadata presence ===")
    
    for item in image_list:
        title = item['title']
        image = item['image']
        
        print(f"\nChecking metadata for: {title}")
        metadata = extract_metadata_from_image_basic(image)
        
        if metadata:
            print(f"  ✓ Metadata found in {title}:")
            for k, v in metadata.items():
                if v:
                    print(f"    - {k}: {v}")
            filtered_images.append(item)
        else:
            print(f"  ✗ No metadata found in {title} - excluding from PDF")
    
    print(f"\n📊 Summary: {len(filtered_images)} out of {len(image_list)} images contain metadata")
    return filtered_images

def extract_diagram_by_engrave(pdf_path: str) -> Optional[np.ndarray]:
    """
    Extract diagram enclosed by 'ENGRAVE' markers from PDF.
    """
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=DEFAULT_DPI)
                img_data = pix.tobytes("png")
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                try:
                    data = pytesseract.image_to_data(image, output_type=Output.DICT)
                except Exception as e:
                    print(f"  ⚠️ OCR failed on page {page_num+1}: {e}")
                    continue
                
                # Find ENGRAVE markers
                engrave_boxes = []
                for i, word in enumerate(data['text']):
                    if word and "ENGRAVE" in word.upper():
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        engrave_boxes.append((x, y, w, h))
                
                if len(engrave_boxes) >= 2:
                    # Calculate bounding box
                    x_vals = [x for x, _, w, _ in engrave_boxes] + [x + w for x, _, w, _ in engrave_boxes]
                    y_vals = [y for _, y, _, h in engrave_boxes] + [y + h for _, y, _, h in engrave_boxes]
                    
                    padding = 100
                    x1 = max(min(x_vals) - padding, 0)
                    x2 = min(max(x_vals) + padding, image.shape[1])
                    y1 = max(min(y_vals) - padding, 0)
                    y2 = min(max(y_vals) + padding, image.shape[0])
                    
                    cropped = image[y1:y2, x1:x2]
                    print(f"Extracted diagram from page {page_num+1} using ENGRAVE markers.")
                    return cropped
                else:
                    print(f"ℹ️ Page {page_num+1}: Found {len(engrave_boxes)} ENGRAVE markers (need ≥2).")
        
        print("Warning: No suitable diagram found in PDF.")
        return None
        
    except Exception as e:
        print(f"Error extracting diagram: {e}")
        return None

def split_technical_drawing(image: np.ndarray) -> List[Dict]:
    """Split technical drawing using predefined regions."""
    if image is None:
        return []
    
    try:
        height, width = image.shape[:2]
        regions = {
            "section_view_a_a": (0, 0, width//4, height//2),
            "main_central_diagram": (width//4, height//6, width//2, 2*height//3),
            "section_view_c_c": (3*width//4, 0, width//4, height//2),
            "bottom_left_detail": (0, 2*height//3, width//3, height//3),
            "bottom_center_details": (width//3, 2*height//3, width//3, height//3),
            "bottom_right_detail": (2*width//3, 2*height//3, width//3, height//3),
            "top_detail_views": (width//4, 0, width//2, height//6)
        }
        
        extracted_diagrams = []
        for name, (x, y, w, h) in regions.items():
            if w > 50 and h > 50:
                diagram = image[y:y+h, x:x+w]
                extracted_diagrams.append({'title': name, 'image': diagram})
                print(f"✓ Extracted (predefined): {name}")
        
        return extracted_diagrams
        
    except Exception as e:
        print(f"Error in split_technical_drawing: {e}")
        return []

def split_with_adaptive_regions(image: np.ndarray) -> List[Dict]:
    """Split drawing using contour analysis."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        diagram_regions = []
        min_area = image.shape[0] * image.shape[1] * 0.01 
        
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                diagram_regions.append(image[y1:y2, x1:x2])
        
        diagram_regions.sort(key=lambda img: img.shape[0] * img.shape[1], reverse=True)
        
        final_images = []
        for i, diagram in enumerate(diagram_regions[:6]):
            title = f"adaptive_diagram_{i+1}"
            final_images.append({'title': title, 'image': diagram})
            print(f"✓ Extracted (adaptive): {title}")
        
        return final_images
        
    except Exception as e:
        print(f"Error in split_with_adaptive_regions: {e}")
        return []

def extract_specific_components(image: np.ndarray) -> List[Dict]:
    """Extract components using percentage-based coordinates."""
    try:
        height, width = image.shape[:2]
        specific_regions = {
            "left_section_view": (int(0.02*width), int(0.1*height), int(0.22*width), int(0.45*height)),
            "main_numbered_diagram": (int(0.25*width), int(0.12*height), int(0.48*width), int(0.55*height)),
            "right_section_view": (int(0.75*width), int(0.1*height), int(0.23*width), int(0.45*height)),
            "bottom_left_component": (int(0.02*width), int(0.7*height), int(0.22*width), int(0.28*height)),
            "engrave_specifications": (int(0.25*width), int(0.7*height), int(0.48*width), int(0.28*height)),
            "bottom_right_component": (int(0.75*width), int(0.7*height), int(0.23*width), int(0.28*height))
        }
        
        extracted_components = []
        for name, (x, y, w, h) in specific_regions.items():
            if x + w <= width and y + h <= height and w > 0 and h > 0:
                component = image[y:y+h, x:x+w]
                extracted_components.append({'title': name, 'image': component})
                print(f"✓ Extracted (specific): {name}")
        
        return extracted_components
        
    except Exception as e:
        print(f"Error in extract_specific_components: {e}")
        return []

def process_pdf_and_split_diagrams(pdf_path: str, output_pdf: str = "output_diagrams.pdf") -> bool:
    """
    Complete workflow: extract, split, filter, and save diagrams.
    """
    try:
        print("Step 1: Extracting main diagram using ENGRAVE markers...")
        main_diagram_image = extract_diagram_by_engrave(pdf_path)
        
        if main_diagram_image is None:
            print("Error: Failed to extract diagram from PDF.")
            return False
        
        all_images_for_pdf = [{'title': 'Main Extracted Diagram', 'image': main_diagram_image}]
        
        print("\n=== Processing extracted diagram with multiple methods ===")
        
        all_images_for_pdf.extend(split_technical_drawing(main_diagram_image))
        all_images_for_pdf.extend(split_with_adaptive_regions(main_diagram_image))
        all_images_for_pdf.extend(extract_specific_components(main_diagram_image))
        
        filtered_images = filter_images_with_metadata(all_images_for_pdf)
        
        if not filtered_images:
            print("\n❌ No images contain metadata. No PDF created.")
            return False
        
        success = save_images_to_pdf(filtered_images, output_pdf)
        
        if success:
            print(f"\n✅ Processing completed! PDF with {len(filtered_images)} images saved to: {output_pdf}")
        
        return success
        
    except Exception as e:
        print(f"Error in process_pdf_and_split_diagrams: {e}")
        return False

def remove_arrows_selectively_with_keras_ocr(image: np.ndarray, keras_ocr_lib) -> np.ndarray:
    """
    Remove only arrows and arrowheads using keras-ocr while preserving text metadata.
    """
    try:
        if not keras_ocr_lib or keras_ocr_lib is False:
            raise ValueError("keras-ocr library not properly loaded")
        
        pipeline = keras_ocr_lib.pipeline.Pipeline()
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        rgb_image, was_resized = resize_image_if_needed(rgb_image, KERAS_OCR_MAX_DIM, cv2.INTER_AREA)
        original_shape = image.shape[:2]
        
        prediction_groups = pipeline.recognize([rgb_image])
        
        mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        
        arrow_patterns = [
            r'^[→←↑↓▲▼◄►≫≪⟶⟵⟰⟱↗↘↙↖⇄⇅⇆⇇⇈⇉⇊⇋⇌⇍⇎⇏]$',  # Arrow symbols
            r'^[─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬]$',  # Box drawing
            r'^[°′″‴⁰¹²³⁴⁵⁶⁷⁸⁹]$',  # Degree and superscript symbols
            r'^[▪▫■□●○◦•‣⁃]$',  # Bullet points and shapes
            r'^[∅∆∇∈∉∋∌∏∑∞∫∮∝∴∵∶∷∸∹∺∻∼∽∾∿≀≁≂≃]$',  # Mathematical symbols
        ]
        
        compiled_patterns = [re.compile(pattern) for pattern in arrow_patterns]
        
        for predictions in prediction_groups:
            for word, box in predictions:
                is_arrow_symbol = any(pattern.match(word.strip()) for pattern in compiled_patterns)
                
                is_short_symbol = len(word.strip()) <= 2 and not word.strip().isalnum()
                
                if is_arrow_symbol or is_short_symbol:
                    poly = np.array(box, np.int32)
                    cv2.fillPoly(mask, [poly], 255)
                    print(f"  🎯 Removing symbol: '{word}' (arrow/symbol)")
                else:
                    print(f"  ✓ Preserving text: '{word}' (metadata)")
        
      
        if np.sum(mask) > 0:  
            result = cv2.inpaint(rgb_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            print("  ✓ Selective arrow removal completed")
        else:
            result = rgb_image
            print("  ℹ️ No arrows detected to remove")
        
    
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
       
        if was_resized and result.shape[:2] != original_shape:
            result = cv2.resize(result, (original_shape[1], original_shape[0]), 
                              interpolation=cv2.INTER_LANCZOS4)
        
        return result
        
    except Exception as e:
        print(f"  ⚠️ Selective arrow removal failed: {e}")
        return image

def detect_arrows_with_opencv(image: np.ndarray) -> np.ndarray:
    """
    Detect and remove arrows using OpenCV morphological operations.
    This targets arrow shapes specifically.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        h_arrow_kernel = np.array([
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1]
        ], dtype=np.uint8)
        
        # Vertical arrows
        v_arrow_kernel = np.array([
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]
        ], dtype=np.uint8)
        
        h_arrows = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_arrow_kernel)
        v_arrows = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_arrow_kernel)
        
        arrows_mask = cv2.bitwise_or(h_arrows, v_arrows)
        
        dilate_kernel = np.ones((3, 3), np.uint8)
        arrows_mask = cv2.dilate(arrows_mask, dilate_kernel, iterations=2)
        
        result = cv2.inpaint(image, arrows_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        print("  ✓ OpenCV arrow detection and removal completed")
        return result
        
    except Exception as e:
        print(f"  ⚠️ OpenCV arrow removal failed: {e}")
        return image

def preprocess_image_for_ocr_selective(image: np.ndarray, use_keras_ocr: bool = False) -> np.ndarray:
    """
    Modified preprocessing that selectively removes arrows while preserving text.
    """
    try:
        result_image = image.copy()
        
        enhanced = enhance_image_for_ocr(result_image)
        
        if use_keras_ocr:
            keras_ocr_lib = load_keras_ocr()
            if keras_ocr_lib and keras_ocr_lib is not False:
                try:
                    # Use selective arrow removal instead of general line removal
                    result = remove_arrows_selectively_with_keras_ocr(result_image, keras_ocr_lib)
                    print("  ✓ Used selective keras-ocr arrow removal")
                except Exception as e:
                    print(f"  ⚠️ keras-ocr failed, trying OpenCV arrow removal: {e}")
                    result = detect_arrows_with_opencv(result_image)
            else:
                result = detect_arrows_with_opencv(result_image)
        else:
            result = detect_arrows_with_opencv(result_image)
        
        if len(result.shape) == 3:
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            result_gray = result
        result_clean = cv2.medianBlur(result_gray, 3)        
        if len(result_clean.shape) == 2:
            result_bgr = cv2.cvtColor(result_clean, cv2.COLOR_GRAY2BGR)
        else:
            result_bgr = result_clean
        
        return result_bgr
        
    except Exception as e:
        print(f"  ⚠️ Selective preprocessing failed: {e}")
        return image

def extract_metadata_from_pdf_with_selective_keras_ocr(pdf_path: str) -> None:
    """
    Extract metadata from PDF using selective arrow removal with keras-ocr.
    """
    print(f"\n=== Starting Selective Arrow Removal and Metadata Extraction from: {pdf_path} ===")
    
    patterns = get_metadata_patterns()
    
    try:
        with fitz.open(pdf_path) as doc:
            final_parsed_data = {}
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                print(f"\n--- Processing Page {page_num + 1} with Selective Arrow Removal ---")
                
                pix = page.get_pixmap(dpi=DEFAULT_DPI)
                img_data = pix.tobytes("png")
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                image, was_resized = resize_image_if_needed(image, TESSERACT_MAX_DIM)
                if was_resized:
                    print(f"  ⚠️ Resized image for OCR processing")
                
                use_keras = should_use_keras_ocr(image)
                print("  📋 Applying selective arrow removal preprocessing...")
                preprocessed_image = preprocess_image_for_ocr_selective(image, use_keras_ocr=use_keras)
                
                if len(preprocessed_image.shape) == 3:
                    gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = preprocessed_image
                
                # Extract text using enhanced OCR settings
                enhanced_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:/-°'
                text = pytesseract.image_to_string(gray, config=enhanced_config)
                
               
                text = re.sub(r'[\n\x0c]', ' ', text).strip()
                text = re.sub(r'\s{2,}', ' ', text)
                
                print(f"  📄 Extracted text preview: {text[:200]}...")
                
                parsed_data = {}
                for key, pattern in patterns.items():
                    match = pattern.search(text)
                    parsed_data[key] = match.group(1).strip() if match else None
                
                excluded_keys = {"ST", "ANF", "EQ-N°"}
                parsed_data = {k: v for k, v in parsed_data.items() if k not in excluded_keys}
                
                if parsed_data.get("ID-N°"):
                    parsed_data["ID-N°"] = parsed_data["ID-N°"].replace("QA", "")

                if parsed_data.get("HTZ"):
                    parsed_data["HTZ"] = parsed_data["HTZ"].replace("LH", "")
                
                if parsed_data.get("QA-ID"):
                    parsed_data["QA-ID"] = parsed_data["QA-ID"].replace("7", "/").replace("ISSUE", "").strip()
                
                found_metadata = False
                for k, v in parsed_data.items():
                    if v:  
                        print(f"  ✓ Found {k}: {v}")
                        found_metadata = True
                        
                        if k in final_parsed_data:
                            final_parsed_data[k].append(v)  
                        else:
                            final_parsed_data[k] = [v]  

                if not found_metadata:
                    print("  ℹ️ No metadata patterns found on this page.")
                print(f"\n📋 Final parsed data summary:")
                for key, values in final_parsed_data.items():
                    print(f"  {key}: {values} (count: {len(values)})")
        return final_parsed_data
    
    except Exception as e:
        print(f"Error in selective metadata extraction: {e}")

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    output_pdf_path = tmp_file.name

# ''''    
# if not os.path.exists(pdf_path):
#         print(f"❌ Error: PDF file not found at {pdf_path}")
#         print("Please update the pdf_path variable with the correct path to your PDF file.")
#         exit(1)
    
# print("🚀 Starting Enhanced PDF Processing")
# print("=" * 70)
# print("📋 Phase 1: Extract and filter diagrams")
# print("📋 Phase 2: Advanced metadata extraction with keras-ocr")
# print("=" * 70)
    
# success = process_pdf_and_split_diagrams(pdf_path, output_pdf=output_pdf_path)
    
# if success and os.path.exists(output_pdf_path):
#     print("\n" + "=" * 50)
#     print("🎯 Starting Selective Arrow Removal and Metadata Extraction")
#     print("=" * 50)
#     parsed_data = extract_metadata_from_pdf_with_selective_keras_ocr(output_pdf_path)
# elif not success:
#     print(f"\nPhase 1 failed - skipping advanced metadata extraction.")
# else:
#     print(f"\nOutput PDF '{output_pdf_path} was not created. Skipping metadata extraction.")
        
#     print("\n" + "=" * 60)
#     print("✅ Enhanced PDF processing completed!")
#     print("📋 Summary:")
#     print("  - Used standard OCR for initial metadata filtering")
#     print("  - Used keras-ocr selectively for final data extraction")
#     print("  - Optimized performance by avoiding unnecessary keras-ocr usage")
# '''
# ---------- Streamlit App ----------
st.set_page_config(page_title="ME PCPR AI Powered Non Conformity", layout="wide")
st.title("📐 ME PCPR AI Powered Non Conformity Checker")
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0
if st.button("🔄 Clear All Uploaded Files"):
    st.session_state.reset_counter += 1
    st.rerun()
diagram_key = f"diagram_{st.session_state.reset_counter}"
MR_key = f"MR_{st.session_state.reset_counter}"
mmt_key = f"mmt_{st.session_state.reset_counter}"
excel_key = f"excel_{st.session_state.reset_counter}"

with st.sidebar:
    st.header("📁 Upload Files")
    diagram_file = st.file_uploader("Upload Diagram PDF", type="pdf", key=diagram_key)
    MR_file = st.file_uploader("Upload MR Report PDF", type="pdf", key=MR_key)
    mmt_file = st.file_uploader("Upload MMT Report PDF", type="pdf", key=mmt_key)
    excel_file = st.file_uploader("Upload Data For Cross-Check", type=["xlsx", "xls", "csv"], key=excel_key)

    st.markdown("---")
    st.caption("Use the 'Clear All' button at the top to reset file uploads.")

    if diagram_file is not None:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(diagram_file.read())
            temp_pdf_path = tmp_file.name  


all_files_uploaded = diagram_file is not None and \
                    MR_file is not None and \
                    mmt_file is not None and \
                    excel_file is not None

if all_files_uploaded:
    with st.status("🔍 Processing files...", expanded=True) as status:
        st.write("Extracting coordinates from Diagram PDF...")
        result = extract_coordinates_from_pdf(diagram_file)
        if not result:
            status.update(label="❌ Processing failed: Could not extract coordinates from Diagram PDF", state="error"); st.stop()
        coordinates, tolerances = result["coordinates"], result["tolerances"]

        st.write("Extracting dimension tables from MR PDF...")
        MR_tables = extract_dimension_tables_from_MR(MR_file)
        if not MR_tables:
            status.update(label="❌ Processing failed: Could not extract dimension tables from MR PDF", state="error"); st.stop()

        st.write("Extracting key identifiers from MR Report...")
        MR_identifiers = extract_identifiers_from_MR(MR_file)
        

        st.write("Extracting dates for validation...")
        cert_date_mmt = extract_cert_date_from_mmt(mmt_file)
        insp_date_MR = extract_inspection_date_from_MR(MR_file)
        
        st.write("Loading and processing data file for cross-checks...")
        processed_excel_df = load_and_process_excel_data(excel_file)
        if processed_excel_df.empty:
            st.warning("Failed to load or process data file. Cross-checks may be limited.")

        status.update(label="✅ Processing complete", state="complete")

    #TAB ORDER
    tab_titles = ["📊 Conformance Checks", "🗓️ Date Validation", "⚙️ Data Cross-Check",]
    tabs = st.tabs(tab_titles)
    conformance_tab, date_validation_tab, data_cross_check_tab = tabs

    with conformance_tab: # Conformance Checks
        with st.expander("✔️ Conformance Check 1 – Nominal Match with Diagram", expanded=True):
            for letter, refs in MR_tables.items():
                for ref, table in refs.items():
                    try:
                        nominal = {axis: float(table[table["U"] == axis]["Nominale"].iloc[0].replace(",", ".")) for axis in "XYZ"}
                        diagram_table = coordinates[f"{letter} - Coordinate"]
                        diagram = {axis: float(diagram_table[diagram_table["REF."] == ref.split("-")[0]][axis].iloc[0]) for axis in "XYZ"}
                        if all(math.isclose(nominal[ax], diagram[ax], rel_tol=1e-6, abs_tol=1e-4) for ax in "XYZ"):
                            st.success(f"✅ Nominal Match Passed for {letter}{ref}")
                        else:
                            for axis in "XYZ":
                                if not math.isclose(nominal[axis], diagram[axis], rel_tol=1e-6, abs_tol=1e-4):
                                    st.error(f"❌ {axis} mismatch for {letter}{ref}: Nominal={nominal[axis]} | Diagram={diagram[axis]}")
                    except Exception as e: st.warning(f"⚠️ Skipping {letter}{ref} due to error in Conformance Check 1: {e}")
        with st.expander("✔️ Conformance Check 2 – Measured Value Within Tolerance", expanded=True):
            for letter, refs in MR_tables.items():
                if letter == "B": continue
                for ref, table in refs.items():
                    try:
                        if letter not in tolerances:
                            st.warning(f"⚠️ No tolerance for point {letter}. Skipping check for {letter}{ref}."); continue
                        tolerance_ext = float(tolerances[letter].replace('Ø', ''))
                        all_within = True
                        for axis in "XYZ":
                            deviation = float(table[table["U"] == axis]["Measured"].iloc[0].replace(",", ".")) - float(table[table["U"] == axis]["Nominale"].iloc[0].replace(",", "."))
                            if not -tolerance_ext / 2 <= deviation <= tolerance_ext / 2:
                                st.error(f"❌ {axis} for {letter}{ref} OUT OF tolerance: Δ={deviation:.4f}"); all_within = False
                        if all_within: st.success(f"✅ Tolerance Passed for {letter}{ref}")
                    except Exception as e: st.warning(f"⚠️ Skipping tolerance check for {letter}{ref} due to error in Conformance Check 2: {e}")

    with date_validation_tab:
        st.subheader("🗓️ Automated Date Validation")
        if cert_date_mmt and insp_date_MR:
            st.write(f"**Certificate Date (from MMT):** `{cert_date_mmt}`")
            st.write(f"**Inspection Date (from MR):** `{insp_date_MR}`")
            
            is_valid, valid_until_date = is_date_valid(cert_date_mmt, insp_date_MR)
            
            if is_valid:
                if valid_until_date:
                    st.success(f"✅ Dates are valid. The inspection date is within two years of the certificate issue date. Validity extends until **{valid_until_date.strftime('%d/%m/%Y')}**.")
                else:
                    st.success(f"✅ Dates are valid.")
            else:
                st.error(f"❌ Inspection Date is NOT within 2 years of Certificate Date.")
        else:
            st.warning("Could not extract both dates for validation. Please check the PDF content.")
    
    # Identifier Cross-Check
    with data_cross_check_tab: 
        st.subheader("📋 Sequential Data Cross-Check with Tool Description")
        
        if excel_file and not processed_excel_df.empty:
            st.write("Performing sequential cross-check using MR identifiers and Tool:")
            eindeutigkeit_value, check_status_message, output_row = perform_sequential_cross_check(processed_excel_df, MR_identifiers)
            
            # Process PDF and extract metadata
            success = process_pdf_and_split_diagrams(temp_pdf_path, output_pdf=output_pdf_path)

            parsed_data = None
            if success and os.path.exists(output_pdf_path):
                st.write("🎯 Extracting metadata from processed PDF...")
                parsed_data = extract_metadata_from_pdf_with_selective_keras_ocr(output_pdf_path)
            elif not success:
                st.warning("⚠️ Phase 1 failed - skipping advanced metadata extraction.")

            if "✅" in check_status_message:
                st.success(check_status_message)
            elif "❌" in check_status_message:
                st.error(check_status_message)
            elif "⚠️" in check_status_message:
                st.warning(check_status_message)
            else:
                st.info(check_status_message) 

            if eindeutigkeit_value is not None:
                st.markdown(f"**Final Eindeutigkeit Value:** `{eindeutigkeit_value}`")
            else:
                st.info("No 'Eindeutigkeit' value could be determined from the cross-check.")

            # Cross-check parsed metadata with Excel data row
            if parsed_data and output_row is not None:
                st.write("---")
                st.write("🔍 **Metadata Cross-Check Results:**")
                
                # Convert pandas Series to dictionary if needed
                if hasattr(output_row, 'to_dict'):
                    excel_data = output_row.to_dict()
                elif isinstance(output_row, dict):
                    excel_data = output_row
                else:
                    st.error("❌ Output row format not supported for comparison.")
                    excel_data = None
                
                if excel_data:
                    matches = []
                    mismatches = []
                    
                    # Debug helpers
                    #st.write("**Debug Info:**")
                    #st.write(f"PDF DATA DEBUG HELPER FN: {list(parsed_data.keys())}")
                    #st.write(f"EXCEL DATA DEBUG HELPER FN {list(excel_data.keys())}")
                    
                    # Only check HTZ, ID-N°, and QA-ID (removed Anf./ISSUE as it's not in Excel)
                    key_mappings = {
                        'HTZ': ['HTZ', 'HTZ neu', 'HTZ_neu', 'HTZ neu_normalized'],
                        'ID-N°': ['ID-Nr', 'ID-N°', 'Template Ident-No.', 'Template Ident-No._normalized'],
                        'QA-ID': ['QA-ID', 'QA identification', 'QA identification_normalized']
                    }
                    
                    for pdf_key, pdf_values in parsed_data.items():
                        if pdf_key not in key_mappings:
                            continue
                            
                        if not pdf_values:  
                            continue
                            
                        excel_match_found = False
                        excel_value = None
                        matched_excel_key = None
                        
                        possible_keys = key_mappings.get(pdf_key, [pdf_key])
                        
                        for excel_key in possible_keys:
                            if excel_key in excel_data and excel_data[excel_key] is not None:
                                excel_value = str(excel_data[excel_key]).strip()
                                matched_excel_key = excel_key
                                break
                        
                        if excel_value:
                            pdf_match_found = False
                            for pdf_val in pdf_values:
                                if pdf_val:
                                    pdf_val_clean = str(pdf_val).strip()
                                    if pdf_val_clean.lower() == excel_value.lower():
                                        matches.append(f"**{pdf_key}**: PDF = `{pdf_val_clean}`  Excel = `{excel_value}` ✅")
                                        pdf_match_found = True
                                        break
                                    elif pdf_key in ['ID-N°', 'HTZ'] and (
                                        pdf_val_clean.lower() in excel_value.lower() or 
                                        excel_value.lower() in pdf_val_clean.lower()
                                    ):
                                        matches.append(f"**{pdf_key}**: PDF = `{pdf_val_clean}`  Excel = `{excel_value}` ✅ ")
                                        pdf_match_found = True
                                        break
                            
                            if not pdf_match_found:
                                pdf_vals_str = ", ".join([f"`{v}`" for v in pdf_values if v])
                                mismatches.append(f"**{pdf_key}**: PDF = [{pdf_vals_str}]  Excel = `{excel_value}` ({matched_excel_key}) ❌")
                        else:
                            pdf_vals_str = ", ".join([f"`{v}`" for v in pdf_values if v])
                            mismatches.append(f"**{pdf_key}**: PDF = [{pdf_vals_str}]  Excel = `Not Found` ⚠️")
                    
                    total_matches = len(matches)
                    total_comparisons = len(matches) + len(mismatches)
                    
                    if total_matches >= 2:
                        st.success(f"🎉 **Cross-Check PASSED** ")
                    elif total_matches > 0:
                        st.warning(f"⚠️ **Cross-Check PARTIAL** ")
                    else:
                        st.error(f"❌ **Cross-Check FAILED** ")
                    
                    if matches:
                        st.write("**✅ Matching Identifiers:**")
                        for match in matches:
                            st.write(f"- {match}")
                    
                    if mismatches:
                        st.write("**❌ Non-Matching Identifiers:**")
                        for mismatch in mismatches:
                            st.write(f"- {mismatch}")
            
            elif parsed_data is None or not parsed_data:
                st.info("📄 No metadata extracted from PDF for comparison.")
            elif output_row is None:
                st.info("📊 No matching row found in Excel data for comparison.")
            else:
                st.warning("⚠️ Unable to perform metadata cross-check. Check data formats.")

        elif not excel_file:
            st.info("Upload the Data Cross-Check file to perform detailed sequential cross-checks.")
        elif processed_excel_df.empty:
            st.warning("Data file was uploaded but could not be processed. Cannot perform cross-checks.")

else:
    st.info("📥 Please upload **all four files** to begin the conformity checks.")
    if diagram_file is None:
        st.write("- Diagram PDF: Missing")
    if MR_file is None:
        st.write("- MR Report PDF: Missing")
    if mmt_file is None:
        st.write("- MMT Report PDF: Missing (Required for date validation)")
    if excel_file is None:
        st.write("- Data For Cross-Check: Missing (Required for identifier cross-checks)")