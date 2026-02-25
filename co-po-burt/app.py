import streamlit as st

import pandas as pd

from src.io_utils import load_thresholds, load_targets

from src.nba_math import compute_po_attainment_nba

from src.burt import compute_burt_adjustments_from_students

from src.nlp_mapping import generate_co_po_mapping, generate_co_to_single_outcome_mapping
from src.co_attainment_calculator import generate_co_attainment_csv, calculate_co_attainment_from_internal, calculate_weighted_co_attainment
from src.excel_parser import parse_compiler_excel, extract_co_columns



st.set_page_config(page_title="CO–PO Attainment System", layout="wide")



st.title("CO–PO / PSO Attainment Dashboard")



st.sidebar.header("Mode")



mode = st.sidebar.radio(

    "Select Mode",

    ["NLP CO–PO Mapping", "CO Attainment Calculator", "PO/PSO Attainment Calculation", "End-to-End (Excel → CO → PO)", "Full Pipeline (Map → Attain → PO)"]

)



# --------------------

# Upload section

# --------------------

st.sidebar.header("Upload Files")



if mode == "NLP CO–PO Mapping":

    co_text_file = st.sidebar.file_uploader("CO Statements CSV", type=["csv"])

    po_text_file = st.sidebar.file_uploader("PO / PSO Statements CSV", type=["csv"])

    # NLP Mapping Controls
    st.sidebar.subheader("NLP Mapping Settings")

    st.sidebar.markdown("**Preprocessing**")
    preprocess_mode = st.sidebar.selectbox(
        "Preprocessing Strength",
        ["minimal", "aggressive_with_fallback", "aggressive", "light"],
        index=0,
        help="minimal: keeps full sentences (human-like, recommended); aggressive: removes stopwords + generic verbs; light: stopwords only; fallback: tries aggressive, uses light if too short"
    )
    dept = st.sidebar.selectbox(
        "Department",
        ["general", "engineering", "business", "cs"],
        index=0,
        help="Department-specific filtering for generic verbs"
    )

    st.sidebar.info("ℹ️ Weight bins are fixed:\n- 0.00-0.10 ⇒ 0\n- 0.10-0.25 ⇒ 1\n- 0.25-0.50 ⇒ 2\n- 0.50+ ⇒ 3")

    show_comparison = st.sidebar.checkbox("Show raw vs processed similarity", value=False,
        help="Compare similarity scores with and without preprocessing")

    if not co_text_file:
        st.info("⬅️ Upload CO statement CSV to begin")
        st.stop()

    if not po_text_file:
        st.info("⬅️ Upload PO / PSO statement CSV to continue")
        st.stop()



elif mode == "PO/PSO Attainment Calculation":

    co_file = st.sidebar.file_uploader("CO Attainment CSV", type=["csv"])

    map_file = st.sidebar.file_uploader("CO → PO / PSO Mapping CSV", type=["csv"])

    threshold_file = st.sidebar.file_uploader("Thresholds CSV", type=["csv"])

    target_file = st.sidebar.file_uploader("Targets CSV", type=["csv"])



    if not all([co_file, map_file, threshold_file, target_file]):

        st.info("⬅️ Upload all required CSV files for attainment calculation")

        st.stop()



# --------------------

# NLP Mapping Mode

# --------------------

if mode == "NLP CO–PO Mapping":

    co_text_df = pd.read_csv(co_text_file, encoding="latin1")

    po_text_df = pd.read_csv(po_text_file, encoding="latin1")

    # Detect columns
    from src.nlp_mapping import detect_id_column, detect_text_column

    po_id_col = detect_id_column(po_text_df, ["po", "pso", "outcome"])
    po_text_col = detect_text_column(po_text_df, po_id_col)

    # Clean PO/PSO data
    po_text_df = po_text_df.dropna(subset=[po_text_col])
    po_text_df[po_id_col] = po_text_df[po_id_col].astype(str).str.strip()
    po_text_df[po_text_col] = po_text_df[po_text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Create dropdown options with ID + text preview
    po_options = []
    for _, row in po_text_df.iterrows():
        outcome_id = row[po_id_col]
        outcome_text = row[po_text_col]
        # Truncate text for dropdown display
        text_preview = outcome_text[:80] + "..." if len(outcome_text) > 80 else outcome_text
        po_options.append(f"{outcome_id}: {text_preview}")

    # Dropdown to select ONE outcome
    st.subheader("Select PO / PSO for Mapping")
    selected_option = st.selectbox(
        "Choose exactly ONE outcome to map against ALL COs:",
        po_options,
        help="Select the Program Outcome or Program Specific Outcome to analyze"
    )

    # Extract selected outcome ID
    selected_outcome_id = selected_option.split(":")[0].strip()
    selected_row = po_text_df[po_text_df[po_id_col] == selected_outcome_id].iloc[0]
    selected_outcome_text = selected_row[po_text_col]

    # Display selected outcome clearly
    st.info(f"**Selected Outcome:** {selected_outcome_id}\n\n{selected_outcome_text}")

    # Run mapping
    with st.spinner("Computing NLP mapping..."):
        mapping_df = generate_co_to_single_outcome_mapping(
            co_text_df,
            outcome_id=selected_outcome_id,
            outcome_text=selected_outcome_text,
            dept=dept,
            preprocess_mode=preprocess_mode
        )
        
        # Optionally compute raw (unprocessed) similarity for comparison
        if show_comparison:
            raw_mapping_df = generate_co_to_single_outcome_mapping(
                co_text_df,
                outcome_id=selected_outcome_id,
                outcome_text=selected_outcome_text,
                dept=dept,
                preprocess_mode="minimal"
            )
            # Merge raw similarity into main dataframe
            mapping_df['raw_similarity'] = raw_mapping_df['similarity']
            mapping_df['similarity_diff'] = mapping_df['raw_similarity'] - mapping_df['similarity']

    # Display results
    st.subheader(f"CO Mapping to {selected_outcome_id}")

    # Show top-N preview
    # Fix: Handle empty dataframe case
    if mapping_df.empty:
        st.warning("No mapping results available. The dataframe is empty.")
        top_n = 5
    else:
        max_slider = max(5, len(mapping_df))
        default_val = min(10, len(mapping_df)) if len(mapping_df) > 0 else 5
        top_n = st.slider("Show top N results", min_value=5, max_value=max_slider, value=default_val, step=1)
    st.markdown(f"**Top {top_n} Results (sorted by similarity)**")
    st.dataframe(mapping_df.head(top_n), use_container_width=True)

    # Show full table in expander
    with st.expander("View Full Results"):
        st.dataframe(mapping_df, use_container_width=True)

    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total COs", len(mapping_df))
    col2.metric("Weight 3", len(mapping_df[mapping_df["weight"] == 3]))
    col3.metric("Weight 2", len(mapping_df[mapping_df["weight"] == 2]))
    col4.metric("Weight 1", len(mapping_df[mapping_df["weight"] == 1]))

    # Download buttons
    st.subheader("Download Results")

    # CSV download
    csv = mapping_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"co_mapping_{selected_outcome_id}.csv",
        mime="text/csv",
    )

    # PDF download (optional)
    try:
        from io import BytesIO
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch

        def generate_pdf(df, outcome_id, outcome_text, settings):
            """Generate PDF report of CO-PO mapping."""
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            elements = []
            styles = getSampleStyleSheet()

            # Title
            title = Paragraph(f"<b>CO Mapping to {outcome_id}</b>", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 0.2*inch))

            # Outcome text
            outcome_para = Paragraph(f"<b>Outcome:</b> {outcome_text}", styles['Normal'])
            elements.append(outcome_para)
            elements.append(Spacer(1, 0.2*inch))

            # Settings
            settings_text = f"<b>Settings:</b> dept={settings['dept']}, mode={settings['mode']}, weight_bins={settings['weight_bins']}"
            settings_para = Paragraph(settings_text, styles['Normal'])
            elements.append(settings_para)
            elements.append(Spacer(1, 0.3*inch))

            # Table
            # Limit table to top 30 rows for PDF readability
            table_data = [["CO", "Similarity", "Weight"]]
            for _, row in df.head(30).iterrows():
                table_data.append([row["co"], f"{row['similarity']:.4f}", str(row["weight"])])

            table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)

            # Note if truncated
            if len(df) > 30:
                elements.append(Spacer(1, 0.2*inch))
                note = Paragraph(f"<i>Note: Showing top 30 of {len(df)} results. Download CSV for complete data.</i>", styles['Normal'])
                elements.append(note)

            # Build PDF
            doc.build(elements)
            buffer.seek(0)
            return buffer

        # Generate and download PDF
        pdf_buffer = generate_pdf(
            mapping_df,
            selected_outcome_id,
            selected_outcome_text,
            {
                'dept': dept,
                'mode': preprocess_mode,
                'weight_bins': 'Fixed: 0.00-0.25=>0, 0.25-0.50=>1, 0.50-0.75=>2, 0.75-1.00=>3'
            }
        )

        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name=f"co_mapping_{selected_outcome_id}.pdf",
            mime="application/pdf",
        )

    except ImportError:
        st.warning("PDF export requires reportlab. Install with: pip install reportlab")



# --------------------

# PO/PSO Attainment Calculation Mode

# --------------------

elif mode == "PO/PSO Attainment Calculation":

    # --------------------

    # Load data

    # --------------------

    co_df = pd.read_csv(co_file)

    map_df = pd.read_csv(map_file)

    thresholds = load_thresholds(threshold_file)

    targets = load_targets(target_file)



    # Filters

    years = sorted(co_df["year"].unique())

    courses = sorted(co_df["course"].unique())

    att_types = sorted(co_df["attainment_type"].unique())



    col1, col2, col3 = st.columns(3)

    year = col1.selectbox("Year", years)

    course = col2.selectbox("Course", courses)

    att_type = col3.selectbox("Attainment Type", att_types)



    co_df = co_df[

        (co_df["year"] == year) &

        (co_df["course"] == course) &

        (co_df["attainment_type"] == att_type)

    ]

    map_df = map_df[map_df["course"] == course]



    # --------------------

    # Burt (optional)

    # --------------------

    use_burt = st.sidebar.checkbox("Use Burt Adjustment", value=False)

    student_file = None

    if use_burt:

        student_file = st.sidebar.file_uploader("Student CO Scores CSV", type=["csv"])

        if student_file is None:

            st.error("Student CO Scores CSV required for Burt mode")

            st.stop()

        stu_df = pd.read_csv(student_file)

        stu_df = stu_df[

            (stu_df["year"] == year) &

            (stu_df["course"] == course)

        ]

        assoc = compute_burt_adjustments_from_students(stu_df, thresholds)

    else:

        assoc = None



    # --------------------

    # Compute

    # --------------------

    results = compute_po_attainment_nba(

        co_attainment=co_df,

        mapping=map_df,

        thresholds=thresholds,

        targets=targets,

        attainment_type=att_type,

        assoc=assoc,

    )



    # --------------------

    # Display

    # --------------------

    st.subheader("CO Attainment (Used)")

    st.dataframe(results["co_attainment_used"], use_container_width=True)



    st.subheader("CO Attainment Levels")

    st.dataframe(results["co_report"], use_container_width=True)



    st.subheader("PO / PSO Attainment (Scale of 3)")

    st.dataframe(results["po_matrix_scale"], use_container_width=True)



    st.subheader("Target Achievement (≥ 1.4)")

    st.dataframe(results["po_matrix_target"], use_container_width=True)



    st.subheader("PO / PSO Attainment (%)")

    st.dataframe(results["po_matrix_pct"], use_container_width=True)



    st.success("✅ Computation complete")


# --------------------
# CO Attainment Calculator Mode
# --------------------

elif mode == "CO Attainment Calculator":
    st.header("CO Attainment Calculator")
    st.markdown("""
    Calculate Course Outcome (CO) attainment from Internal Assessment and ESE scores.
    
    **Input Format Expected:**
    - Internal Assessment CSV: student_id, name, plus CO-mapped question scores
    - ESE CSV (optional): student_id, name, CO-mapped ESE scores
    
    **Output:** CO attainment CSV ready for PO calculation
    """
    )
    
    # Course metadata
    col1, col2, col3 = st.columns(3)
    course_code = col1.text_input("Course Code", value="CS601")
    course_name = col2.text_input("Course Name", value="Compiler Design")
    year = col3.text_input("Year", value="2024")
    
    # File uploads
    st.subheader("Upload Assessment Data")
    
    internal_file = st.file_uploader("Internal Assessment CSV", type=["csv"])
    ese_file = st.file_uploader("ESE (External) CSV (optional)", type=["csv"])
    
    # Weights
    st.subheader("Attainment Weights")
    col1, col2 = st.columns(2)
    internal_weight = col1.slider("Internal Weight", 0.0, 1.0, 0.4, 0.1)
    ese_weight = col2.slider("ESE Weight", 0.0, 1.0, 0.6, 0.1)
    
    if internal_file:
        internal_df = pd.read_csv(internal_file)
        st.subheader("Internal Assessment Preview")
        st.dataframe(internal_df.head(), use_container_width=True)
        
        # Column mapping
        st.subheader("Map Columns to COs")
        st.markdown("Select which columns map to each Course Outcome (CO)")
        
        available_cols = internal_df.columns.tolist()
        
        co1_cols = st.multiselect("CO1 Columns", available_cols, default=[])
        co2_cols = st.multiselect("CO2 Columns", available_cols, default=[])
        co3_cols = st.multiselect("CO3 Columns", available_cols, default=[])
        co4_cols = st.multiselect("CO4 Columns", available_cols, default=[])
        
        if st.button("Calculate CO Attainment"):
            if not any([co1_cols, co2_cols, co3_cols, co4_cols]):
                st.error("Please select at least one column for a CO")
                st.stop()
            
            # Build CO mapping
            co_mapping = {}
            max_marks = {}
            
            if co1_cols:
                co_mapping['CO1'] = co1_cols
                for col in co1_cols:
                    max_marks[col] = internal_df[col].max()
            if co2_cols:
                co_mapping['CO2'] = co2_cols
                for col in co2_cols:
                    max_marks[col] = internal_df[col].max()
            if co3_cols:
                co_mapping['CO3'] = co3_cols
                for col in co3_cols:
                    max_marks[col] = internal_df[col].max()
            if co4_cols:
                co_mapping['CO4'] = co4_cols
                for col in co4_cols:
                    max_marks[col] = internal_df[col].max()
            
            # Calculate attainment
            from src.co_attainment_calculator import calculate_co_attainment_from_internal
            
            results_df = calculate_co_attainment_from_internal(
                internal_df, co_mapping, max_marks
            )
            
            # Add metadata
            results_df['course'] = course_code
            results_df['course_name'] = course_name
            results_df['year'] = year
            results_df['attainment_type'] = 'INTERNAL'
            results_df['value'] = results_df['attainment_pct'] / 100.0
            
            st.subheader("CO Attainment Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            output_df = results_df[['year', 'course', 'co', 'attainment_type', 'value']]
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CO Attainment CSV",
                data=csv,
                file_name=f"co_attainment_{course_code}_{year}.csv",
                mime="text/csv"
            )
            
            st.success("✅ CO Attainment calculated. Download and use in PO Attainment Calculation mode.")


# --------------------
# End-to-End Mode (Excel → CO → PO)
# --------------------

elif mode == "End-to-End (Excel → CO → PO)":
    st.header("End-to-End Attainment Calculator")
    st.markdown("""
    **Complete pipeline:** Upload Excel → Calculate CO Attainment → Compute PO/PSO Attainment
    
    **Expected Excel Format:**
    - **Internal** tab: Student internal assessment scores (CO-mapped columns)
    - **ESE** tab: End Semester Exam scores (CO-mapped columns)
    - **Indirect** tab (optional): Indirect assessment data
    - **Mapping** tab: CO → PO/PSO mapping with weights
    
    **Flow:** Internal → ESE → Direct → Final CO → PO/PSO %
    """)
    
    # Course metadata
    col1, col2, col3 = st.columns(3)
    course_code = col1.text_input("Course Code", value="CS601")
    course_name = col2.text_input("Course Name", value="Compiler Design")
    year = col3.text_input("Year", value="2024")
    
    # File uploads
    st.subheader("Upload Data")
    
    excel_file = st.file_uploader("Assessment Excel (Internal/ESE/Indirect tabs)", type=["xlsx", "xls"])
    threshold_file = st.file_uploader("Thresholds CSV", type=["csv"])
    target_file = st.file_uploader("Targets CSV", type=["csv"])
    
    # Weights for attainment calculation
    st.subheader("Attainment Weights")
    col1, col2, col3 = st.columns(3)
    internal_weight = col1.slider("Internal Weight", 0.0, 1.0, 0.4, 0.1)
    ese_weight = col2.slider("ESE Weight", 0.0, 1.0, 0.6, 0.1)
    indirect_weight = col3.slider("Indirect Weight", 0.0, 1.0, 0.0, 0.1)
    
    if excel_file and threshold_file and target_file:
        # Load thresholds and targets
        try:
            thresholds = load_thresholds(threshold_file)
            targets = load_targets(target_file)
        except Exception as e:
            st.error(f"Error loading thresholds/targets: {e}")
            st.stop()
        
        # Parse Excel
        try:
            parsed = parse_compiler_excel(excel_file)
            internal_df = parsed.get('internal')
            ese_df = parsed.get('ese')
            indirect_df = parsed.get('indirect')
        except Exception as e:
            st.error(f"Error parsing Excel: {e}")
            st.stop()
        
        if internal_df is None:
            st.error("Could not find Internal assessment data in Excel. Please ensure a tab named 'Internal' exists.")
            st.stop()
        
        # Display previews
        st.subheader("Data Preview")
        
        with st.expander("Internal Assessment Data"):
            st.dataframe(internal_df.head(10), use_container_width=True)
        
        if ese_df is not None:
            with st.expander("ESE Data"):
                st.dataframe(ese_df.head(10), use_container_width=True)
        
        if indirect_df is not None:
            with st.expander("Indirect Assessment Data"):
                st.dataframe(indirect_df.head(10), use_container_width=True)
        
        # Auto-detect CO columns
        internal_co_mapping = extract_co_columns(internal_df)
        
        st.subheader("Detected CO Column Mapping")
        for co, cols in internal_co_mapping.items():
            st.markdown(f"**{co}:** {', '.join(cols)}")
        
        # Manual override for column mapping
        with st.expander("Override CO Column Mapping (Optional)"):
            available_cols = [c for c in internal_df.columns if c not in ['RegNo', 'Name', 'RollNo', 'StudentID']]
            
            for co in ['CO1', 'CO2', 'CO3', 'CO4', 'CO5', 'CO6']:
                default_cols = internal_co_mapping.get(co, [])
                selected = st.multiselect(f"{co} Columns", available_cols, default=default_cols, key=f"e2e_{co}")
                if selected:
                    internal_co_mapping[co] = selected
        
        # Calculate max marks for each column
        max_marks = {}
        for co, cols in internal_co_mapping.items():
            for col in cols:
                if col in internal_df.columns:
                    max_marks[col] = internal_df[col].max()
        
        # Process button
        if st.button("Calculate Full Attainment Flow", type="primary"):
            with st.spinner("Processing Internal Assessment..."):
                # Step 1: Calculate Internal CO Attainment
                internal_results = calculate_co_attainment_from_internal(
                    internal_df, internal_co_mapping, max_marks
                )
            
            # Step 2: Calculate ESE CO Attainment (if provided)
            if ese_df is not None:
                with st.spinner("Processing ESE..."):
                    ese_co_mapping = extract_co_columns(ese_df)
                    ese_max_marks = {}
                    for co, cols in ese_co_mapping.items():
                        for col in cols:
                            if col in ese_df.columns:
                                ese_max_marks[col] = ese_df[col].max()
                    
                    ese_results = calculate_co_attainment_from_internal(
                        ese_df, ese_co_mapping, ese_max_marks
                    )
            else:
                ese_results = None
            
            # Step 3: Calculate weighted attainment (Internal + ESE + Indirect)
            with st.spinner("Calculating Weighted Attainment..."):
                if ese_results is not None:
                    weighted = calculate_weighted_co_attainment(
                        internal_results.rename(columns={'attainment_pct': 'attainment_pct'}),
                        ese_results.rename(columns={'attainment_pct': 'attainment_pct'}),
                        internal_weight=internal_weight,
                        ese_weight=ese_weight,
                        indirect_weight=indirect_weight,
                        indirect_scores=indirect_df
                    )
                else:
                    # Only internal assessment
                    weighted = internal_results.copy()
                    weighted['direct_attainment'] = weighted['attainment_pct']
                    weighted['final_attainment'] = weighted['attainment_pct']
                    weighted['attainment_value'] = weighted['attainment_pct'] / 100.0
            
            # Display CO-level results
            st.subheader("📊 CO Attainment Results")
            
            co_results_tab = st.tabs(["Internal", "ESE", "Direct", "Final"])
            
            with co_results_tab[0]:
                st.markdown("**Internal Assessment CO Attainment**")
                internal_display = internal_results.copy()
                internal_display['course'] = course_code
                internal_display['year'] = year
                internal_display['attainment_type'] = 'INTERNAL'
                st.dataframe(internal_display, use_container_width=True)
            
            with co_results_tab[1]:
                if ese_results is not None:
                    st.markdown("**ESE CO Attainment**")
                    ese_display = ese_results.copy()
                    ese_display['course'] = course_code
                    ese_display['year'] = year
                    ese_display['attainment_type'] = 'ESE'
                    st.dataframe(ese_display, use_container_width=True)
                else:
                    st.info("No ESE data provided")
            
            with co_results_tab[2]:
                st.markdown("**Direct Attainment (Internal × weight + ESE × weight)**")
                st.dataframe(weighted[['co', 'attainment_pct_internal', 'attainment_pct_ese', 'direct_attainment']], use_container_width=True)
            
            with co_results_tab[3]:
                st.markdown("**Final CO Attainment**")
                final_display = weighted[['co', 'final_attainment']].copy()
                final_display['attainment_level'] = final_display['final_attainment'].apply(
                    lambda x: 3 if x >= 70 else (2 if x >= 60 else (1 if x >= 50 else 0))
                )
                st.dataframe(final_display, use_container_width=True)
            
            # Step 4: Generate CO Attainment CSV for PO calculation
            co_records = []
            
            # Add INTERNAL
            for _, row in internal_results.iterrows():
                co_records.append({
                    'year': year,
                    'course': course_code,
                    'co': row['co'],
                    'attainment_type': 'INTERNAL',
                    'value': round(row['attainment_pct'] / 100, 4)
                })
            
            # Add ESE
            if ese_results is not None:
                for _, row in ese_results.iterrows():
                    co_records.append({
                        'year': year,
                        'course': course_code,
                        'co': row['co'],
                        'attainment_type': 'ESE',
                        'value': round(row['attainment_pct'] / 100, 4)
                    })
            
            # Add DIRECT
            for _, row in weighted.iterrows():
                co_records.append({
                    'year': year,
                    'course': course_code,
                    'co': row['co'],
                    'attainment_type': 'DIRECT',
                    'value': round(row['direct_attainment'] / 100, 4)
                })
            
            # Add FINAL
            for _, row in weighted.iterrows():
                co_records.append({
                    'year': year,
                    'course': course_code,
                    'co': row['co'],
                    'attainment_type': 'FINAL',
                    'value': round(row['final_attainment'] / 100, 4)
                })
            
            co_attainment_df = pd.DataFrame(co_records)
            
            st.subheader("📥 Generated CO Attainment CSV")
            st.dataframe(co_attainment_df, use_container_width=True)
            
            # Download CO CSV
            co_csv = co_attainment_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CO Attainment CSV",
                data=co_csv,
                file_name=f"co_attainment_{course_code}_{year}.csv",
                mime="text/csv"
            )
            
            # Need CO-PO mapping to proceed to PO calculation
            st.subheader("📋 CO → PO/PSO Mapping")
            st.markdown("Upload the CO → PO/PSO mapping file to calculate final PO/PSO attainment")
            
            mapping_file = st.file_uploader("CO → PO/PSO Mapping CSV", type=["csv"], key="e2e_mapping")
            
            if mapping_file:
                try:
                    map_df = load_mapping(mapping_file)
                    map_df = map_df[map_df['course'] == course_code]
                    
                    st.markdown("**Mapping Preview:**")
                    st.dataframe(map_df, use_container_width=True)
                    
                    # Calculate PO/PSO attainment
                    with st.spinner("Calculating PO/PSO Attainment..."):
                        results = compute_po_attainment_nba(
                            co_attainment=co_attainment_df,
                            mapping=map_df,
                            thresholds=thresholds,
                            targets=targets,
                            attainment_type='FINAL'
                        )
                    
                    st.subheader("🎯 Final PO/PSO Attainment Results")
                    
                    po_tabs = st.tabs(["PO Matrix (%)", "PO Scale (3)", "Target Achievement"])
                    
                    with po_tabs[0]:
                        st.markdown("**PO/PSO Attainment (%)**")
                        st.dataframe(results["po_matrix_pct"], use_container_width=True)
                    
                    with po_tabs[1]:
                        st.markdown("**PO/PSO Attainment (Scale of 3)**")
                        st.dataframe(results["po_matrix_scale"], use_container_width=True)
                    
                    with po_tabs[2]:
                        st.markdown("**Target Achievement (≥ 1.4)**")
                        st.dataframe(results["po_matrix_target"], use_container_width=True)
                    
                    st.success("✅ Full attainment flow complete: Internal → ESE → Direct → Final CO → PO/PSO %")
                    
                except Exception as e:
                    st.error(f"Error calculating PO/PSO attainment: {e}")
                    st.stop()


# --------------------
# Full Pipeline Mode (Map → Attain → PO)
# --------------------

elif mode == "Full Pipeline (Map → Attain → PO)":
    st.header("Full Pipeline: CO-PO Mapping → Attainment → Final PO")
    st.markdown("""
    **Complete workflow in one place:**
    1. **Step 1:** Generate CO-PO mapping with NLP
    2. **Step 2:** Review/adjust mapping weights
    3. **Step 3:** Upload assessment data (Internal/ESE)
    4. **Step 4:** Calculate CO attainment flow
    5. **Step 5:** Calculate final PO/PSO attainment
    6. **Step 6:** Export compiler-style summary
    """)
    
    # Initialize session state for pipeline data
    if 'pipeline_mapping' not in st.session_state:
        st.session_state.pipeline_mapping = None
    if 'pipeline_co_attainment' not in st.session_state:
        st.session_state.pipeline_co_attainment = None
    if 'pipeline_step' not in st.session_state:
        st.session_state.pipeline_step = 1
    
    # Progress indicator
    step = st.session_state.pipeline_step
    st.progress(step / 6, text=f"Step {step} of 6")
    
    # Step 1: CO-PO Mapping
    with st.expander("📋 Step 1: CO-PO Mapping", expanded=(step == 1)):
        st.markdown("Upload CO and PO/PSO statements to generate automatic mapping")
        
        co_text_file = st.file_uploader("CO Statements CSV", type=["csv"], key="pipeline_co")
        po_text_file = st.file_uploader("PO / PSO Statements CSV", type=["csv"], key="pipeline_po")
        
        col1, col2 = st.columns(2)
        with col1:
            preprocess_mode = st.selectbox(
                "Preprocessing",
                ["minimal", "aggressive_with_fallback", "aggressive", "light"],
                index=0,
                help="minimal: keeps full sentences (recommended)"
            )
        with col2:
            dept = st.selectbox("Department", ["general", "engineering", "business", "cs"], index=0)
        
        if co_text_file and po_text_file:
            if st.button("Generate Mapping", type="primary"):
                with st.spinner("Computing NLP mapping..."):
                    co_text_df = pd.read_csv(co_text_file, encoding="latin1")
                    po_text_df = pd.read_csv(po_text_file, encoding="latin1")
                    
                    from src.nlp_mapping import detect_id_column, detect_text_column, generate_co_po_mapping
                    
                    # Generate full mapping (all COs to all POs)
                    mapping_df = generate_co_po_mapping(
                        co_text_df, po_text_df,
                        threshold=0.0,  # Include all for manual review
                        dept=dept,
                        preprocess_mode=preprocess_mode
                    )
                    
                    st.session_state.pipeline_mapping = mapping_df
                    st.session_state.pipeline_step = 2
                    st.rerun()
    
    # Step 2: Review Mapping
    if step >= 2 and st.session_state.pipeline_mapping is not None:
        with st.expander("✏️ Step 2: Review & Adjust Mapping", expanded=(step == 2)):
            st.markdown("Review the automatically generated mapping. Adjust weights if needed.")
            
            mapping_df = st.session_state.pipeline_mapping
            
            # Filter to show only non-zero weights by default
            show_all = st.checkbox("Show all mappings (including weight 0)", value=False)
            if not show_all:
                mapping_display = mapping_df[mapping_df['weight'] > 0]
            else:
                mapping_display = mapping_df
            
            st.dataframe(mapping_display, use_container_width=True)
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Mappings", len(mapping_df))
            col2.metric("Weight 3", len(mapping_df[mapping_df['weight'] == 3]))
            col3.metric("Weight 2", len(mapping_df[mapping_df['weight'] == 2]))
            col4.metric("Weight 1", len(mapping_df[mapping_df['weight'] == 1]))
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Step 1"):
                    st.session_state.pipeline_step = 1
                    st.rerun()
            with col2:
                if st.button("Proceed to Step 3 →", type="primary"):
                    st.session_state.pipeline_step = 3
                    st.rerun()
    
    # Step 3: Upload Assessment Data
    if step >= 3:
        with st.expander("📊 Step 3: Upload Assessment Data", expanded=(step == 3)):
            st.markdown("Upload Internal and ESE assessment data")
            
            col1, col2, col3 = st.columns(3)
            course_code = col1.text_input("Course Code", value="CS601")
            course_name = col2.text_input("Course Name", value="Compiler Design")
            year = col3.text_input("Year", value="2024")
            
            internal_file = st.file_uploader("Internal Assessment CSV", type=["csv"], key="pipeline_internal")
            ese_file = st.file_uploader("ESE CSV (optional)", type=["csv"], key="pipeline_ese")
            
            col1, col2, col3 = st.columns(3)
            internal_weight = col1.slider("Internal Weight", 0.0, 1.0, 0.4, 0.1)
            ese_weight = col2.slider("ESE Weight", 0.0, 1.0, 0.6, 0.1)
            indirect_weight = col3.slider("Indirect Weight", 0.0, 1.0, 0.0, 0.1)
            
            if internal_file:
                if st.button("Process Assessments →", type="primary"):
                    with st.spinner("Processing assessments..."):
                        internal_df = pd.read_csv(internal_file)
                        
                        # Auto-detect CO columns
                        internal_co_mapping = extract_co_columns(internal_df)
                        
                        # Calculate max marks
                        max_marks = {}
                        for co, cols in internal_co_mapping.items():
                            for col in cols:
                                if col in internal_df.columns:
                                    max_marks[col] = internal_df[col].max()
                        
                        # Calculate internal CO attainment
                        internal_results = calculate_co_attainment_from_internal(
                            internal_df, internal_co_mapping, max_marks
                        )
                        
                        # Process ESE if provided
                        if ese_file:
                            ese_df = pd.read_csv(ese_file)
                            ese_co_mapping = extract_co_columns(ese_df)
                            ese_max_marks = {}
                            for co, cols in ese_co_mapping.items():
                                for col in cols:
                                    if col in ese_df.columns:
                                        ese_max_marks[col] = ese_df[col].max()
                            
                            ese_results = calculate_co_attainment_from_internal(
                                ese_df, ese_co_mapping, ese_max_marks
                            )
                            
                            # Calculate weighted
                            weighted = calculate_weighted_co_attainment(
                                internal_results.rename(columns={'attainment_pct': 'attainment_pct'}),
                                ese_results.rename(columns={'attainment_pct': 'attainment_pct'}),
                                internal_weight=internal_weight,
                                ese_weight=ese_weight,
                                indirect_weight=indirect_weight
                            )
                        else:
                            weighted = internal_results.copy()
                            weighted['direct_attainment'] = weighted['attainment_pct']
                            weighted['final_attainment'] = weighted['attainment_pct']
                            weighted['attainment_value'] = weighted['attainment_pct'] / 100.0
                        
                        # Build CO attainment records
                        co_records = []
                        for _, row in internal_results.iterrows():
                            co_records.append({
                                'year': year, 'course': course_code, 'co': row['co'],
                                'attainment_type': 'INTERNAL', 'value': round(row['attainment_pct'] / 100, 4)
                            })
                        
                        if ese_file:
                            for _, row in ese_results.iterrows():
                                co_records.append({
                                    'year': year, 'course': course_code, 'co': row['co'],
                                    'attainment_type': 'ESE', 'value': round(row['attainment_pct'] / 100, 4)
                                })
                        
                        for _, row in weighted.iterrows():
                            co_records.append({
                                'year': year, 'course': course_code, 'co': row['co'],
                                'attainment_type': 'DIRECT', 'value': round(row['direct_attainment'] / 100, 4)
                            })
                            co_records.append({
                                'year': year, 'course': course_code, 'co': row['co'],
                                'attainment_type': 'FINAL', 'value': round(row['final_attainment'] / 100, 4)
                            })
                        
                        st.session_state.pipeline_co_attainment = pd.DataFrame(co_records)
                        st.session_state.pipeline_course_code = course_code
                        st.session_state.pipeline_year = year
                        st.session_state.pipeline_step = 4
                        st.rerun()
    
    # Step 4: Review CO Attainment
    if step >= 4 and st.session_state.pipeline_co_attainment is not None:
        with st.expander("📈 Step 4: CO Attainment Results", expanded=(step == 4)):
            co_df = st.session_state.pipeline_co_attainment
            
            # Pivot for display
            co_pivot = co_df.pivot_table(
                index=['year', 'course', 'co'],
                columns='attainment_type',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            st.dataframe(co_pivot, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Step 3"):
                    st.session_state.pipeline_step = 3
                    st.rerun()
            with col2:
                if st.button("Proceed to Step 5 →", type="primary"):
                    st.session_state.pipeline_step = 5
                    st.rerun()
    
    # Step 5: Calculate PO Attainment
    if step >= 5:
        with st.expander("🎯 Step 5: Calculate PO/PSO Attainment", expanded=(step == 5)):
            st.markdown("Using the mapping from Step 2 and CO attainment from Step 4")
            
            threshold_file = st.file_uploader("Thresholds CSV", type=["csv"], key="pipeline_thresholds")
            target_file = st.file_uploader("Targets CSV", type=["csv"], key="pipeline_targets")
            
            if threshold_file and target_file:
                if st.button("Calculate Final PO Attainment →", type="primary"):
                    with st.spinner("Computing PO/PSO attainment..."):
                        thresholds = load_thresholds(threshold_file)
                        targets = load_targets(target_file)
                        
                        mapping_df = st.session_state.pipeline_mapping
                        co_df = st.session_state.pipeline_co_attainment
                        course_code = st.session_state.pipeline_course_code
                        
                        # Filter mapping to this course
                        mapping_df = mapping_df[mapping_df['course'] == course_code] if 'course' in mapping_df.columns else mapping_df
                        
                        results = compute_po_attainment_nba(
                            co_attainment=co_df,
                            mapping=mapping_df,
                            thresholds=thresholds,
                            targets=targets,
                            attainment_type='FINAL'
                        )
                        
                        st.session_state.pipeline_po_results = results
                        st.session_state.pipeline_step = 6
                        st.rerun()
    
    # Step 6: Final Summary / Compiler Tab
    if step >= 6 and st.session_state.pipeline_po_results is not None:
        with st.expander("✅ Step 6: Final Compiler Summary", expanded=True):
            st.markdown("## 📋 Complete Attainment Report")
            
            results = st.session_state.pipeline_po_results
            
            # Create tabs for different views
            summary_tabs = st.tabs(["PO Matrix (%)", "PO Scale (3)", "Target Achievement", "Export All"])
            
            with summary_tabs[0]:
                st.markdown("### PO/PSO Attainment (%)")
                st.dataframe(results["po_matrix_pct"], use_container_width=True)
            
            with summary_tabs[1]:
                st.markdown("### PO/PSO Attainment (Scale of 3)")
                st.dataframe(results["po_matrix_scale"], use_container_width=True)
            
            with summary_tabs[2]:
                st.markdown("### Target Achievement (≥ 1.4)")
                st.dataframe(results["po_matrix_target"], use_container_width=True)
            
            with summary_tabs[3]:
                st.markdown("### Export Complete Dataset")
                
                # Create a zip of all data
                import io
                import zipfile
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # Mapping
                    mapping_csv = st.session_state.pipeline_mapping.to_csv(index=False)
                    zf.writestr('co_po_mapping.csv', mapping_csv)
                    
                    # CO Attainment
                    co_csv = st.session_state.pipeline_co_attainment.to_csv(index=False)
                    zf.writestr('co_attainment.csv', co_csv)
                    
                    # PO Results
                    po_pct_csv = results["po_matrix_pct"].to_csv(index=False)
                    zf.writestr('po_attainment_pct.csv', po_pct_csv)
                    
                    po_scale_csv = results["po_matrix_scale"].to_csv(index=False)
                    zf.writestr('po_attainment_scale.csv', po_scale_csv)
                    
                    po_target_csv = results["po_matrix_target"].to_csv(index=False)
                    zf.writestr('po_target_achievement.csv', po_target_csv)
                
                zip_buffer.seek(0)
                st.download_button(
                    label="📥 Download Complete Report (ZIP)",
                    data=zip_buffer,
                    file_name=f"attainment_report_{st.session_state.pipeline_course_code}_{st.session_state.pipeline_year}.zip",
                    mime="application/zip"
                )
            
            st.success("✅ Full pipeline complete! CO-PO Mapping → Attainment → Final PO Results")
            
            if st.button("🔄 Start New Pipeline"):
                # Clear session state
                for key in ['pipeline_mapping', 'pipeline_co_attainment', 'pipeline_po_results', 
                           'pipeline_course_code', 'pipeline_year', 'pipeline_step']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.pipeline_step = 1
                st.rerun()
