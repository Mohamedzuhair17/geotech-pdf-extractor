import streamlit as st
import tempfile
import os
import pandas as pd

# Import the extraction pipeline functions
from extract_geotech import process_pdf, output_table, OUTPUT_COLUMNS

st.set_page_config(
    page_title="Geotechnical PDF Extractor",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Geotechnical PDF Data Extractor")
st.markdown("""
Upload a Geotechnical Laboratory Factual Report PDF to extract soil test data.
The pipeline extracts Location, Depth, Soil Description, PSD, and Atterberg limits into a standardized table.
""")

# File uploader
uploaded_file = st.file_uploader("Upload PDF Report", type=["pdf"])

if uploaded_file is not None:
    st.info(f"File uploaded: {uploaded_file.name}")
    
    # Run extraction on button click
    if st.button("Extract Data", type="primary"):
        with st.spinner("Processing PDF... This may take a minute depending on the size and LLM fallback."):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_pdf_path = tmp_file.name
                
            try:
                # Run the extraction pipeline
                # process_pdf takes a file path
                rows = process_pdf(tmp_pdf_path)
                
                if not rows:
                    st.warning("No data extracted from the PDF. It may not contain recognizable geotechnical lab tables.")
                else:
                    st.success(f"Successfully extracted {len(rows)} sample rows!")
                    
                    # Format data into a DataFrame
                    display_rows = []
                    for row in rows:
                        display_row = {}
                        for key, col_name in OUTPUT_COLUMNS.items():
                            display_row[col_name] = row.get(key, "-")
                        display_rows.append(display_row)
                    
                    df = pd.DataFrame(display_rows)
                    
                    # Display the dataframe
                    st.subheader("Extracted Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Provide CSV download
                    csv = df.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name="extracted_geotech_results.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"An error occurred during extraction: {e}")
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_pdf_path):
                    os.remove(tmp_pdf_path)
