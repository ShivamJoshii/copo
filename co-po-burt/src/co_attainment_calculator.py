"""
CO Attainment Calculator Module
Calculates Course Outcome (CO) attainment from Internal Assessment and ESE scores.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def calculate_co_attainment_from_internal(
    internal_df: pd.DataFrame,
    co_mapping: Dict[str, List[str]],
    max_marks: Dict[str, float]
) -> pd.DataFrame:
    """
    Calculate CO attainment from Internal Assessment scores.
    
    Args:
        internal_df: DataFrame with student scores
        co_mapping: Dict mapping CO to list of question/section columns
                   e.g., {'CO1': ['QA1(a)', 'QA1(b)', 'QB2'], ...}
        max_marks: Dict mapping question to max marks
                  e.g., {'QA1(a)': 1, 'QA1(b)': 1, ...}
    
    Returns:
        DataFrame with CO attainment per student and course-level attainment
    """
    results = []
    
    for co, questions in co_mapping.items():
        # Calculate max possible for this CO
        co_max = sum(max_marks.get(q, 0) for q in questions)
        
        if co_max == 0:
            continue
            
        # Calculate student scores for this CO
        internal_df[f'{co}_score'] = internal_df[questions].sum(axis=1)
        internal_df[f'{co}_pct'] = (internal_df[f'{co}_score'] / co_max) * 100
        
        # Count students achieving >= 50%
        students_above_50 = (internal_df[f'{co}_pct'] >= 50).sum()
        total_students = len(internal_df)
        attainment_pct = (students_above_50 / total_students) * 100
        
        # Determine level
        if attainment_pct >= 70:
            level = 3
        elif attainment_pct >= 60:
            level = 2
        elif attainment_pct >= 50:
            level = 1
        else:
            level = 0
            
        results.append({
            'co': co,
            'students_above_50': students_above_50,
            'total_students': total_students,
            'attainment_pct': round(attainment_pct, 2),
            'attainment_level': level,
            'max_marks': co_max
        })
    
    return pd.DataFrame(results)


def calculate_weighted_co_attainment(
    internal_attainment: pd.DataFrame,
    ese_attainment: pd.DataFrame,
    internal_weight: float = 0.4,
    ese_weight: float = 0.6,
    indirect_weight: float = 0.0,
    indirect_scores: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculate weighted CO attainment combining Internal and ESE.
    
    Direct Attainment = (Internal × weight) + (ESE × weight)
    Final Attainment = Direct × (1 - indirect_weight) + Indirect × indirect_weight
    
    Args:
        internal_attainment: DataFrame with columns [co, attainment_pct, ...]
        ese_attainment: DataFrame with columns [co, attainment_pct, ...]
        internal_weight: Weight for internal assessment (default 0.4)
        ese_weight: Weight for ESE (default 0.6)
        indirect_weight: Weight for indirect assessment (default 0.0)
        indirect_scores: Optional DataFrame with indirect attainment per CO
    
    Returns:
        DataFrame with combined attainment per CO
    """
    # Merge internal and ESE
    merged = internal_attainment[['co', 'attainment_pct']].merge(
        ese_attainment[['co', 'attainment_pct']],
        on='co',
        suffixes=('_internal', '_ese')
    )
    
    # Calculate direct attainment
    merged['direct_attainment'] = (
        merged['attainment_pct_internal'] * internal_weight +
        merged['attainment_pct_ese'] * ese_weight
    )
    
    # Add indirect if provided
    if indirect_scores is not None and indirect_weight > 0:
        merged = merged.merge(
            indirect_scores[['co', 'attainment_pct']].rename(
                columns={'attainment_pct': 'indirect_attainment'}
            ),
            on='co',
            how='left'
        )
        merged['indirect_attainment'] = merged['indirect_attainment'].fillna(0)
        merged['final_attainment'] = (
            merged['direct_attainment'] * (1 - indirect_weight) +
            merged['indirect_attainment'] * indirect_weight
        )
    else:
        merged['indirect_attainment'] = 0
        merged['final_attainment'] = merged['direct_attainment']
    
    # Convert to value (0-1 scale) for PO calculation
    merged['attainment_value'] = merged['final_attainment'] / 100.0
    
    return merged[['co', 'attainment_pct_internal', 'attainment_pct_ese', 
                   'direct_attainment', 'indirect_attainment', 'final_attainment', 
                   'attainment_value']]


def parse_internal_assessment_pdf_format(
    df: pd.DataFrame,
    course_code: str,
    course_name: str,
    year: str = "2024"
) -> pd.DataFrame:
    """
    Parse the Internal Assessment format from the PDF.
    Expected columns: RegNo, Name, plus question columns mapped to COs.
    
    For Compiler Design example:
    - CO1: Mid Sem questions mapped to CO1
    - CO2: Assignment sections
    - CO3: Presentation/Quiz
    - CO4: MOOC/Attendance
    """
    # Standard format from PDF
    co_mapping = {
        'CO1': ['Mid_CO1'],  # Aggregated mid-sem CO1 score
        'CO2': ['Mid_CO2', 'AS_SecA', 'AS_SecB'],  # Mid + Assignment
        'CO3': ['Mid_CO3', 'PPT_Quiz'],  # Mid + Presentation
        'CO4': ['Mid_CO4', 'MOOC', 'Attendance']  # Mid + MOOC + Attendance
    }
    
    max_marks = {
        'Mid_CO1': 20, 'Mid_CO2': 20, 'Mid_CO3': 20, 'Mid_CO4': 20,
        'AS_SecA': 2.5, 'AS_SecB': 2.5,
        'PPT_Quiz': 5,
        'MOOC': 5, 'Attendance': 5
    }
    
    # Calculate CO-wise attainment
    co_results = calculate_co_attainment_from_internal(df, co_mapping, max_marks)
    
    # Add metadata
    co_results['course'] = course_code
    co_results['course_name'] = course_name
    co_results['year'] = year
    co_results['attainment_type'] = 'INTERNAL'
    co_results['value'] = co_results['attainment_pct'] / 100.0
    
    return co_results[['year', 'course', 'co', 'attainment_type', 'attainment_pct', 
                       'attainment_level', 'value']]


def generate_co_attainment_csv(
    internal_df: pd.DataFrame,
    ese_df: Optional[pd.DataFrame] = None,
    indirect_df: Optional[pd.DataFrame] = None,
    course_code: str = "CS601",
    course_name: str = "Compiler Design",
    year: str = "2024"
) -> pd.DataFrame:
    """
    Main function to generate CO attainment CSV for PO calculation.
    
    Returns DataFrame in format:
    year,course,co,attainment_type,value
    """
    records = []
    
    # Parse internal assessment
    internal_co = parse_internal_assessment_pdf_format(
        internal_df, course_code, course_name, year
    )
    
    for _, row in internal_co.iterrows():
        records.append({
            'year': row['year'],
            'course': row['course'],
            'co': row['co'],
            'attainment_type': 'INTERNAL',
            'value': round(row['value'], 4)
        })
    
    # If ESE provided, calculate and add DIRECT + FINAL
    if ese_df is not None:
        # Calculate ESE CO attainment (similar logic)
        ese_co = parse_internal_assessment_pdf_format(
            ese_df, course_code, course_name, year
        )
        ese_co['attainment_type'] = 'ESE'
        
        for _, row in ese_co.iterrows():
            records.append({
                'year': row['year'],
                'course': row['course'],
                'co': row['co'],
                'attainment_type': 'ESE',
                'value': round(row['value'], 4)
            })
        
        # Calculate weighted attainment
        weighted = calculate_weighted_co_attainment(
            internal_co.rename(columns={'attainment_pct': 'attainment_pct'}),
            ese_co.rename(columns={'attainment_pct': 'attainment_pct'}),
            indirect_scores=indirect_df
        )
        
        # Add DIRECT attainment
        for _, row in weighted.iterrows():
            records.append({
                'year': year,
                'course': course_code,
                'co': row['co'],
                'attainment_type': 'DIRECT',
                'value': round(row['direct_attainment'] / 100, 4)
            })
            
            # Add FINAL attainment
            records.append({
                'year': year,
                'course': course_code,
                'co': row['co'],
                'attainment_type': 'FINAL',
                'value': round(row['final_attainment'] / 100, 4)
            })
    
    return pd.DataFrame(records)
