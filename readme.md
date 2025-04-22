# 🧅 H-2A Job Order Explorer

This project extracts and visualizes structured data from H-2A job order listings hosted on [seasonaljobs.dol.gov](https://seasonaljobs.dol.gov). It includes a custom pipeline for scraping, OCR extraction, geocoding, and rendering interactive dashboards to explore farm employers, retailer networks, and more.

## Features

- 📄 Extracts employer addresses and case numbers from PDFs and screenshots
- 🔍 OCR fallback using Tesseract for failed PDF extractions
- 📍 Geocodes employer addresses for mapping
- 🗺️ Interactive Streamlit dashboard with:
  - Employer map
  - Address table
  - Word clouds of retailer mentions
  - Network graph of farms and retailers
  - Sankey Diagrams
  - Retailer diversity chart

## Link to Dashboard: https://mlaw-699.streamlit.app/
>>>>>>> a3c5ff8 (Add Streamlit config + updated dashboard + readme)
