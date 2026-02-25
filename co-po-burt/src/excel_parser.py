"""
Excel Parser for Compiler Assessment Format
Handles multi-tab Excel files with Internal, ESE, and Indirect assessment data.
"""

import pandas as pd
from typing import Dict, Optional, Tuple
import io


def parse_compiler_excel(
    excel_file: io.BytesIO,
    internal_sheet: str = "Internal",
    ese_sheet: str = "ESE",
    indirect_sheet: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Parse the compiler.xlsx multi-tab format.
    
    Expected tabs:
    - Internal: Student internal assessment scores
    - ESE: End Semester Exam scores  
    - Indirect (optional): Indirect assessment scores
    
    Args:
        excel_file: Uploaded Excel file (BytesIO)
        internal_sheet: Name of internal assessment tab
        ese_sheet: Name of ESE tab
        indirect_sheet: Name of indirect assessment tab (None if not present)
    
    Returns:
        Dict with keys 'internal', 'ese', 'indirect' (if present)
    """
    result = {}
    
    # Get all sheet names
    xl = pd.ExcelFile(excel_file)
    available_sheets = xl.sheet_names
    
    # Parse Internal sheet
    if internal_sheet in available_sheets:
        result['internal'] = pd.read_excel(excel_file, sheet_name=internal_sheet)
    else:
        # Try to find a sheet with "internal" in the name
        internal_candidates = [s for s in available_sheets if 'internal' in s.lower()]
        if internal_candidates:
            result['internal'] = pd.read_excel(excel_file, sheet_name=internal_candidates[0])
    
    # Parse ESE sheet
    if ese_sheet in available_sheets:
        result['ese'] = pd.read_excel(excel_file, sheet_name=ese_sheet)
    else:
        # Try to find a sheet with "ese" or "external" in the name
        ese_candidates = [s for s in available_sheets if 'ese' in s.lower() or 'external' in s.lower()]
        if ese_candidates:
            result['ese'] = pd.read_excel(excel_file, sheet_name=ese_candidates[0])
    
    # Parse Indirect sheet (optional)
    if indirect_sheet and indirect_sheet in available_sheets:
        result['indirect'] = pd.read_excel(excel_file, sheet_name=indirect_sheet)
    elif indirect_sheet:
        # Try to find a sheet with "indirect" in the name
        indirect_candidates = [s for s in available_sheets if 'indirect' in s.lower()]
        if indirect_candidates:
            result['indirect'] = pd.read_excel(excel_file, sheet_name=indirect_candidates[0])
    
    return result


def detect_column_types(df: pd.DataFrame) -> Dict[str, list]:
    """
    Auto-detect column types in assessment data.
    
    Returns dict with:
    - student_id_cols: Columns that look like student IDs
    - name_cols: Columns that look like names
    - score_cols: Numeric columns that look like scores
    - co_cols: Columns mapped to specific COs
    """
    result = {
        'student_id_cols': [],
        'name_cols': [],
        'score_cols': [],
        'co_cols': []
    }
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Detect student ID columns
        if any(keyword in col_lower for keyword in ['reg', 'roll', 'id', 'enroll', 'student']):
            result['student_id_cols'].append(col)
        # Detect name columns
        elif any(keyword in col_lower for keyword in ['name', 'student_name']):
            result['name_cols'].append(col)
        # Detect CO-mapped columns
        elif any(keyword in col_lower for keyword in ['co1', 'co2', 'co3', 'co4', 'co5', 'co6']):
            result['co_cols'].append(col)
        # Detect numeric score columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            result['score_cols'].append(col)
    
    return result


def parse_co_mapped_excel(
    excel_file: io.BytesIO,
    course_code: str = "CS601",
    course_name: str = "Compiler Design",
    year: str = "2024"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Parse Excel with CO-mapped columns and return structured data.
    
    Expected format:
    - Columns: RegNo, Name, CO1_Q1, CO1_Q2, CO2_Q1, etc.
    - Or: RegNo, Name, Mid_CO1, Mid_CO2, AS_SecA, etc.
    
    Returns:
        (internal_df, ese_df, indirect_df) - ese and indirect may be None
    """
    # Read all sheets
    xl = pd.ExcelFile(excel_file)
    
    internal_df = None
    ese_df = None
    indirect_df = None
    
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        sheet_lower = sheet_name.lower()
        
        if 'internal' in sheet_lower or sheet_lower == 'ia':
            internal_df = df
        elif 'ese' in sheet_lower or 'external' in sheet_lower:
            ese_df = df
        elif 'indirect' in sheet_lower:
            indirect_df = df
    
    # If no sheets detected by name, assume first sheet is internal
    if internal_df is None and xl.sheet_names:
        internal_df = pd.read_excel(excel_file, sheet_name=xl.sheet_names[0])
    
    return internal_df, ese_df, indirect_df


def extract_co_columns(df: pd.DataFrame) -> Dict[str, list]:
    """
    Extract columns grouped by CO from dataframe headers.
    
    Example:
        Input columns: ['RegNo', 'Name', 'CO1_Mid', 'CO1_Assign', 'CO2_Mid', 'CO2_Assign']
        Output: {'CO1': ['CO1_Mid', 'CO1_Assign'], 'CO2': ['CO2_Mid', 'CO2_Assign']}
    """
    co_mapping = {}
    
    for col in df.columns:
        col_str = str(col).upper()
        
        # Look for CO patterns: CO1, CO2, etc.
        for i in range(1, 10):
            if f'CO{i}' in col_str or f'CO_{i}' in col_str:
                co_key = f'CO{i}'
                if co_key not in co_mapping:
                    co_mapping[co_key] = []
                co_mapping[co_key].append(col)
                break
    
    return co_mapping
