import streamlit as st

import pandas as pd

from src.io_utils import load_thresholds, load_targets

from src.nba_math import compute_po_attainment_nba

from src.burt import compute_burt_adjustments_from_students

from src.nlp_mapping import generate_co_po_mapping, generate_co_to_single_outcome_mapping
from src.co_attainment_calculator import generate_co_attainment_csv



st.set_page_config(page_title="CO–PO Attainment System", layout="wide")



st.title("CO–PO / PSO Attainment Dashboard")



st.sidebar.header("Mode")



mode = st.sidebar.radio(

    "Select Mode",

    ["NLP CO–PO Mapping", "CO Attainment Calculator", "PO/PSO Attainment Calculation"]

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
        ["aggressive_with_fallback", "aggressive", "light"],
        index=0,
        help="aggressive: removes stopwords + generic verbs; light: stopwords only; fallback: tries aggressive, uses light if too short"
    )
    dept = st.sidebar.selectbox(
        "Department",
        ["general", "engineering", "business", "cs"],
        index=0,
        help="Department-specific filtering for generic verbs"
    )

    st.sidebar.info("ℹ️ Weight bins are fixed:\n- 0.00-0.25 ⇒ 0\n- 0.25-0.50 ⇒ 1\n- 0.50-0.75 ⇒ 2\n- 0.75-1.00 ⇒ 3")

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

    # Display results
    st.subheader(f"CO Mapping to {selected_outcome_id}")

    # Show top-N preview
    top_n = st.slider("Show top N results", min_value=5, max_value=len(mapping_df), value=min(10, len(mapping_df)), step=1)
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
