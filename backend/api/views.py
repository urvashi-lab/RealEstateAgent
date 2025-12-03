from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
import re
import math
from rapidfuzz import process, fuzz
from .llm import extract_intent_with_llm
from django.http import HttpResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
import io
from datetime import datetime

def build_dynamic_column_map(df):
    column_map = {}

    patterns = {
        "price": ["price", "rate", "avg price", "average price", "weighted", "cost"],
        "total_sales": ["total sales", "sales", "revenue", "turnover"],
        "flat_sold": ["flat sold", "flat_sold", "flats sold", "residential"],  # Changed key to singular
        "office_sold": ["office sold", "office_sold", "offices sold", "commercial"],  # Changed key to singular
        "shop_sold": ["shop sold", "shop_sold", "shops sold", "retail"],  # Changed key to singular
        "demand": ["demand", "booking", "interest"]
    }

    for metric_key, keywords in patterns.items():
        for col in df.columns:
            col_l = col.lower()
            if any(k in col_l for k in keywords):
                column_map[metric_key] = col
                # Also add plural versions pointing to same column
                if "_sold" in metric_key:
                    plural_key = metric_key.replace("_sold", "s_sold")
                    column_map[plural_key] = col
                break

    print(f"🗺️ BUILT COLUMN_MAP: {column_map}")
    return column_map


def detect_metric_from_query(query, available_metrics):
    query = query.lower()

    # First, try exact metric name match
    for metric in available_metrics:
        if metric.replace("_", " ") in query:
            return metric

    synonyms = {
        "flat_sold": ["flats sold", "flat sold", "apartment", "residential unit"],
        "office_sold": ["offices sold", "office sold", "commercial office"],
        "shop_sold": ["shops sold", "shop sold", "retail shop", "store"],
        "price": ["price", "cost", "rate", "value"],
        "total_sales": ["sales", "revenue", "turnover"],
        "demand": ["demand", "booking"]
    }

    # Try multi-word phrases first (more specific)
    for metric, words in synonyms.items():
        for w in words:
            if " " in w and w in query:
                # Return the key if it exists in available_metrics
                if metric in available_metrics:
                    return metric
                # Try plural version
                plural_metric = metric.replace("_sold", "s_sold")
                if plural_metric in available_metrics:
                    return plural_metric
    
    # Then try single words
    for metric, words in synonyms.items():
        for w in words:
            if w in query:
                if metric in available_metrics:
                    return metric
                plural_metric = metric.replace("_sold", "s_sold")
                if plural_metric in available_metrics:
                    return plural_metric

    return None


def extract_years_from_text(text):
    if not text:
        return None
    match = re.search(r"(\d+)", text.lower())
    return int(match.group(1)) if match else None

def fuzzy_match_location(search_term, available_locations, threshold=80):
    if not search_term:
        return None
    result = process.extractOne(search_term, available_locations, scorer=fuzz.ratio, score_cutoff=threshold)
    return result[0] if result else None

def detect_core_columns(df):
    location_patterns = ["location", "area", "city", "region", "final location"]
    year_patterns = ["year", "yr", "date"]

    location_col = None
    year_col = None

    for col in df.columns:
        col_l = col.lower()
        if not location_col and any(p in col_l for p in location_patterns):
            location_col = col
        if not year_col and any(p in col_l for p in year_patterns):
            year_col = col

    return location_col, year_col



@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def analyze(request):
    query = request.data.get("query", "")
    uploaded_file = request.FILES.get("file")
    
    # Option 1: Use uploaded file if provided, otherwise use default
    if uploaded_file:
        try:
            # Read the uploaded file based on extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'xlsx' or file_extension == 'xls':
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                return Response({
                    "error": "Unsupported file format", 
                    "details": "Please upload .xlsx, .xls, or .csv files"
                }, status=400)
                
        except Exception as e:
            return Response({
                "error": "Failed to read uploaded file", 
                "details": str(e)
            }, status=500)
    else:
        # Fallback to default file
        try:
            df = pd.read_excel("data.xlsx")
            print(f"📊 ALL COLUMNS: {list(df.columns)}")  # Add this
        except Exception as e:
            return Response({
                "error": "No file uploaded and failed to load default Excel", 
                "details": str(e)
            }, status=500)

    # Validate that the uploaded file has required structure
    location_col, year_col = detect_core_columns(df)

    if not location_col or not year_col:
        return Response({
            "error": "Invalid file structure",
            "details": "Could not auto-detect location or year column",
            "found_columns": list(df.columns)
        }, status=400)


    COLUMN_MAP = build_dynamic_column_map(df)
    print(f"🔍 DEBUG COLUMN_MAP: {COLUMN_MAP}")
    if not COLUMN_MAP:
        return Response({"error": "No metrics detected"}, status=500)

    try:
        intent = extract_intent_with_llm(query)
        print(f"🤖 LLM INTENT: {intent}")
    except Exception as e:
        return Response({"error": "LLM failed", "details": str(e)}, status=500)

   # ... after extracting intent ...
    
    areas = intent.get("areas") or [intent.get("area")]
    metric_key = intent.get("metric")
    
    time_range = intent.get("time_range")
    print(f"📋 EXTRACTED: areas={areas}, metric={metric_key}, time_range={time_range}")

    # IMPROVED METRIC NORMALIZATION
    if metric_key:
        # Normalize the metric key to match COLUMN_MAP
        if metric_key not in COLUMN_MAP:
            # Try multiple normalization strategies
            possible_keys = [
                metric_key,
                metric_key.rstrip("s"),  # offices_sold -> office_sold
                metric_key + "s",  # office_sold -> office_solds
                metric_key.replace("s_sold", "_sold"),  # offices_sold -> office_sold
                metric_key.replace("_sold", "s_sold"),  # office_sold -> offices_sold
                metric_key.replace("flat", "flat_sold").replace("_sold_sold", "_sold"),
                metric_key.replace("office", "office_sold").replace("_sold_sold", "_sold"),
                metric_key.replace("shop", "shop_sold").replace("_sold_sold", "_sold"),
            ]
            
            # Add variations without underscores
            base_metric = metric_key.replace("_", "")
            for key in list(COLUMN_MAP.keys()):
                if key.replace("_", "").replace("s", "") == base_metric.replace("s", ""):
                    possible_keys.append(key)
            
            # Try to find a match
            for variant in possible_keys:
                if variant in COLUMN_MAP:
                    print(f"🔄 NORMALIZED METRIC: {metric_key} -> {variant}")
                    metric_key = variant
                    break
            
            # If still not found after all attempts
            if metric_key not in COLUMN_MAP:
                print(f"⚠️ Metric '{metric_key}' not in COLUMN_MAP. Attempting detection from query...")
                detected = detect_metric_from_query(query, COLUMN_MAP.keys())
                if detected:
                    print(f"✅ Detected metric from query: {detected}")
                    metric_key = detected
                else:
                    print(f"❌ Could not normalize or detect metric. Available: {list(COLUMN_MAP.keys())}")

    # Rest of your code continues...
    query_lower = query.lower()
    is_general_analysis = False
    
    pattern1 = re.match(r'^(analysis|overview|summary|report)\s+(of|for)\s+[\w\s]+', query_lower)
    pattern2 = re.match(r'^(analyze|analyse|summarize|summarise)\s+[\w\s]+', query_lower)
    
    general_keywords = ["analysis", "analyze", "analyse", "overview", "summary", "summarize", "summarise", "report"]
    specific_keywords = ["price", "rate", "cost", "sales", "revenue", "turnover", "flat", "flats", "apartment", "residential", "shop", "shops", "retail", "store", "commercial", "office", "offices", "demand", "unit", "units"]
    
    has_general_keyword = any(kw in query_lower for kw in general_keywords)
    has_specific_keyword = any(kw in query_lower for kw in specific_keywords)
    
    if pattern1 or pattern2 or (has_general_keyword and not has_specific_keyword):
        is_general_analysis = True
        metric_key = "total_sales" if "total_sales" in COLUMN_MAP else list(COLUMN_MAP.keys())[0]
        print(f"✅ GENERAL ANALYSIS MODE")
    else:
        print(f"🎯 SPECIFIC METRIC MODE")

    if not is_general_analysis and (not metric_key or metric_key not in COLUMN_MAP):
        detected_metric = detect_metric_from_query(query, COLUMN_MAP.keys())
        if detected_metric:
            metric_key = detected_metric
        else:
            return Response({"error": "Could not detect metric", "available_metrics": list(COLUMN_MAP.keys())}, status=400)

    excel_column = COLUMN_MAP[metric_key]

    required_cols = [location_col, year_col, excel_column]
    for col in required_cols:
        if col not in df.columns:
            return Response({"error": f"Missing column: {col}"}, status=500)

    df[location_col] = df[location_col].astype(str).str.strip().str.lower()
    available_locations = df[location_col].unique().tolist()
    
    areas_normalized = []
    matched_areas = {}
    
    for area in areas:
        if not area:
            continue
        area_clean = str(area).strip().lower()
        if area_clean in available_locations:
            areas_normalized.append(area_clean)
            matched_areas[area] = area_clean
        else:
            match = fuzzy_match_location(area_clean, available_locations, threshold=75)
            if match:
                areas_normalized.append(match)
                matched_areas[area] = match
    
    if not areas_normalized:
        return Response({"error": "No matching areas found"}, status=404)

    filtered = df[df[location_col].isin(areas_normalized)]
    if filtered.empty:
        return Response({"error": "No data found"}, status=404)

    n_years = extract_years_from_text(time_range)
    if n_years:
        max_year = int(filtered["year"].max())
        min_year = max_year - (n_years - 1)
        filtered = filtered[(filtered["year"] >= min_year) & (filtered["year"] <= max_year)]

    filtered[excel_column] = pd.to_numeric(filtered[excel_column], errors="coerce")
    filtered = filtered.dropna(subset=[excel_column])

    labels = sorted(filtered["year"].unique().astype(int).tolist())
    series = []
    chart_metrics = []
    
    if is_general_analysis:
        priority_metrics = ["flat_sold", "office_sold", "shop_sold", "demand"]  # Changed to singular
        available_metrics = [(k, v) for k, v in COLUMN_MAP.items() if k in priority_metrics and v in df.columns]
        # ... rest of the code
        if not available_metrics:
            available_metrics = list(COLUMN_MAP.items())[:3]
        else:
            available_metrics = available_metrics[:3]
        
        for m_key, m_col in available_metrics:
            try:
                filtered[m_col] = pd.to_numeric(filtered[m_col], errors='coerce')
                for area in areas_normalized:
                    area_df = filtered[filtered[location_col] == area]
                    values = []
                    for y in labels:
                        year_data = area_df[area_df[year_col] == y][m_col]
                        val = year_data.mean()
                        values.append(float(val) if pd.notnull(val) else 0)
                    series.append({"name": f"{area.title()} - {m_key.replace('_', ' ').title()}", "data": values})
                    chart_metrics.append(m_key)
            except Exception as e:
                print(f"Error adding {m_key}: {e}")
                continue
    else:
        for area in areas_normalized:
            area_df = filtered[filtered[location_col] == area]
            values = []
            for y in labels:
                val = area_df[area_df[year_col] == y][excel_column].mean()
                values.append(float(val) if pd.notnull(val) else 0)
            series.append({"name": area.title(), "data": values})
        chart_metrics = [metric_key]

    original_names = list(matched_areas.keys())

    if is_general_analysis:
        insights = []
        for m_key, m_col in COLUMN_MAP.items():
            if m_col in df.columns:
                try:
                    metric_data = filtered[m_col].dropna()
                    if not metric_data.empty:
                        avg = round(pd.to_numeric(metric_data, errors='coerce').mean(), 2)
                        if pd.notnull(avg):
                            insights.append(f"• {m_key.replace('_', ' ').title()}: {avg:,.0f}")
                except:
                    continue
        summary = f"📊 Complete Analysis for {', '.join(original_names)}:\n" + "\n".join(insights[:5]) + "\n\n💡 Chart shows unit-based metrics. See table for all values."

    elif "compare" in query_lower or "vs" in query_lower or len(areas_normalized) > 1:
        comparison_results = {}
        for area in original_names:
            area_df = filtered[filtered[location_col] == matched_areas[area]]
            if area_df.empty:
                comparison_results[area] = 0
                continue
            area_df = area_df.sort_values(year_col)
            start_val = area_df.iloc[0][excel_column]
            end_val = area_df.iloc[-1][excel_column]
            if pd.notnull(start_val) and start_val != 0:
                comparison_results[area] = round(((end_val - start_val) / start_val) * 100, 2)
            else:
                comparison_results[area] = 0
        best_area = max(comparison_results, key=comparison_results.get)
        summary = f"📊 {metric_key.replace('_', ' ').title()} Comparison:\n{', '.join([f'{k}: {v:+.1f}%' for k, v in comparison_results.items()])}.\n🏆 {best_area} leads."

    elif any(word in query_lower for word in ["growth", "increase", "rise", "trend"]):
        sorted_df = filtered.sort_values(year_col)
        if len(sorted_df) < 2:
            summary = "Insufficient data."
        else:
            start_val = sorted_df.iloc[0][excel_column]
            end_val = sorted_df.iloc[-1][excel_column]
            start_year = int(sorted_df.iloc[0][year_col])
            end_year = int(sorted_df.iloc[-1][year_col])
            if pd.notnull(start_val) and start_val != 0:
                growth_pct = round(((end_val - start_val) / start_val) * 100, 2)
                summary = f"📈 {metric_key.replace('_', ' ').title()} {'increased' if growth_pct > 0 else 'decreased'} by {abs(growth_pct)}% ({start_year}-{end_year})."
            else:
                summary = "Unable to calculate growth."
    else:
        avg_value = round(filtered[excel_column].mean(), 2)
        max_value = filtered[excel_column].max()
        min_value = filtered[excel_column].min()
        max_idx = filtered[excel_column].idxmax()
        min_idx = filtered[excel_column].idxmin()
        max_year = int(filtered.loc[max_idx, year_col])
        min_year = int(filtered.loc[min_idx, year_col])
        metric_name = metric_key.replace('_', ' ').title()
        summary = f"📊 {metric_name} Analysis for {', '.join(original_names)}:\n• Average: {avg_value:,.0f}\n• Peak: {round(max_value, 2):,.0f} ({max_year})\n• Lowest: {round(min_value, 2):,.0f} ({min_year})"

    if is_general_analysis:
        table_cols = [location_col, year_col]
        rename_map = {location_col: "location"}
        for m_key, m_col in list(COLUMN_MAP.items())[:4]:
            if m_col in df.columns:
                table_cols.append(m_col)
                rename_map[m_col] = m_key
        table = filtered[table_cols].copy()
        table = table.rename(columns=rename_map)
    else:
        table = filtered[[location_col, excel_column]].copy()
        table = table.rename(columns={location_col: "location", excel_column: metric_key})
    
    table_dict = table.to_dict(orient="records")
    for row in table_dict:
        for key, value in row.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                row[key] = None

    return Response({
        "intent": intent,
        "detected_metric": metric_key,
        "is_general_analysis": is_general_analysis,
        "chart_title": "Market Overview" if is_general_analysis else f"{metric_key.replace('_', ' ').title()} Over Time",
        "chart_type": "multi_metric" if is_general_analysis else "single_metric",
        "chart_metrics": chart_metrics,
        "summary": summary,
        "matched_locations": matched_areas,
        "chart": {"labels": labels, "series": series},
        "table": table_dict,
        "file_used": uploaded_file.name if uploaded_file else "data.xlsx"
    })


@api_view(['POST'])
def download_pdf(request):
    """
    Generate PDF report from analysis data
    """
    try:
        data = request.data
        
        # Extract data from request
        summary = data.get('summary', 'No summary available')
        chart_data = data.get('chart', {})
        table_data = data.get('table', [])
        chart_title = data.get('chart_title', 'Analysis Report')
        detected_metric = data.get('detected_metric', '')
        matched_locations = data.get('matched_locations', {})
        is_general = data.get('is_general_analysis', False)
        
        # Debug logging
        print(f"📊 PDF Request Data:")
        print(f"  - Chart Title: {chart_title}")
        print(f"  - Detected Metric: {detected_metric}")
        print(f"  - Matched Locations: {matched_locations}")
        print(f"  - Is General: {is_general}")
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=30
        )
        
        # Container for PDF elements
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Custom heading style
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        # Body text style
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#2c3e50')
        )
        
        # Small text style
        small_style = ParagraphStyle(
            'Small',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#7f8c8d')
        )
        
        # ===== HEADER SECTION =====
        title = Paragraph(f"<b>{chart_title}</b>", title_style)
        elements.append(title)
        elements.append(Spacer(1, 10))
        
        # Extract location names from matched_locations or table_data
        location_names = []
        if matched_locations and isinstance(matched_locations, dict):
            location_names = list(matched_locations.keys())
        elif table_data and len(table_data) > 0:
            # Extract unique locations from table
            unique_locs = set()
            for row in table_data:
                if 'location' in row:
                    unique_locs.add(row['location'].title())
            location_names = list(unique_locs)
        
        locations_str = ', '.join(location_names) if location_names else 'N/A'
        
        # Metadata box
        meta_html = f"""
        <para alignment="left">
        <b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
        <b>Locations:</b> {locations_str}<br/>
        """
        
        if not is_general and detected_metric:
            meta_html += f"<b>Metric:</b> {detected_metric.replace('_', ' ').title()}<br/>"
        elif is_general:
            meta_html += f"<b>Analysis Type:</b> General Market Overview<br/>"
        
        meta_html += "</para>"
        
        elements.append(Paragraph(meta_html, body_style))
        elements.append(Spacer(1, 20))
        
        # Horizontal line
        elements.append(Spacer(1, 5))
        
        # ===== SUMMARY SECTION =====
        elements.append(Paragraph("<b>Executive Summary</b>", heading_style))
        
        # Process summary text - remove emojis for PDF
        summary_lines = summary.split('\n')
        for line in summary_lines:
            if line.strip():
                # Remove common emojis
                clean_line = line.strip()
                clean_line = clean_line.replace('📊', '').replace('💡', '')
                clean_line = clean_line.replace('📈', '').replace('🏆', '')
                clean_line = clean_line.strip()
                
                if clean_line:
                    elements.append(Paragraph(clean_line, body_style))
                    elements.append(Spacer(1, 5))
        
        elements.append(Spacer(1, 20))
        
        # ===== CHART SECTION =====
        if chart_data and chart_data.get('labels') and chart_data.get('series'):
            elements.append(Paragraph("<b>Visual Analysis</b>", heading_style))
            elements.append(Spacer(1, 10))
            
            try:
                # Create line chart
                drawing = Drawing(480, 280)  # Increased height for legend
                lc = HorizontalLineChart()
                lc.x = 40
                lc.y = 60  # More space at bottom for legend
                lc.height = 180
                lc.width = 400
                
                labels = chart_data['labels']
                series = chart_data['series']
                
                # Set data
                lc.data = [s['data'] for s in series]
                lc.categoryAxis.categoryNames = [str(l) for l in labels]
                
                # Style category axis
                lc.categoryAxis.labels.angle = 0
                lc.categoryAxis.labels.fontSize = 9
                lc.categoryAxis.labels.dy = -5
                
                # Style value axis
                lc.valueAxis.valueMin = 0
                lc.valueAxis.labels.fontSize = 9
                lc.valueAxis.labelTextFormat = '%d'
                
                # Define colors
                chart_colors = [
                    colors.HexColor('#3498db'),  # Blue
                    colors.HexColor('#e74c3c'),  # Red
                    colors.HexColor('#2ecc71'),  # Green
                    colors.HexColor('#f39c12'),  # Orange
                    colors.HexColor('#9b59b6'),  # Purple
                    colors.HexColor('#1abc9c'),  # Turquoise
                ]
                
                # Style lines
                for i in range(len(series)):
                    lc.lines[i].strokeColor = chart_colors[i % len(chart_colors)]
                    lc.lines[i].strokeWidth = 2
                    lc.lines[i].symbol = None  # No symbols
                
                drawing.add(lc)
                elements.append(drawing)
                
                # Add legend with better formatting
                elements.append(Spacer(1, 10))
                legend_items = []
                for i, s in enumerate(series):
                    color = chart_colors[i % len(chart_colors)].hexval()
                    legend_items.append(
                        f"<font color='{color}'>■</font> {s['name']}"
                    )
                
                # Split legend into multiple lines if too many items
                if len(legend_items) > 3:
                    # Show 3 items per line
                    for i in range(0, len(legend_items), 3):
                        legend_line = " | ".join(legend_items[i:i+3])
                        elements.append(Paragraph(legend_line, small_style))
                else:
                    legend_text = " | ".join(legend_items)
                    elements.append(Paragraph(legend_text, small_style))
                
            except Exception as e:
                print(f"❌ Chart generation error: {str(e)}")
                import traceback
                traceback.print_exc()
                elements.append(Paragraph(f"Chart generation failed: {str(e)}", small_style))
            
            elements.append(Spacer(1, 25))
        
        # ===== DATA TABLE SECTION =====
        if table_data and len(table_data) > 0:
            elements.append(Paragraph("<b>Detailed Data</b>", heading_style))
            elements.append(Spacer(1, 10))
            
            try:
                # Prepare table headers
                headers = list(table_data[0].keys())
                table_content = [[h.replace('_', ' ').title() for h in headers]]
                
                # Add data rows
                for row in table_data:
                    table_row = []
                    for key in headers:
                        value = row.get(key)
                        if value is None:
                            table_row.append('-')
                        elif key == 'year':
                            # Format year without commas
                            table_row.append(str(int(value)))
                        elif isinstance(value, float):
                            # Format large numbers with commas
                            table_row.append(f"{value:,.2f}")
                        elif isinstance(value, int):
                            table_row.append(f"{value:,}")
                        else:
                            table_row.append(str(value).title())
                    table_content.append(table_row)
                
                # Create table with dynamic column widths
                col_widths = [480 / len(headers)] * len(headers)
                t = Table(table_content, colWidths=col_widths, repeatRows=1)
                
                # Style table
                table_style = TableStyle([
                    # Header row
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('TOPPADDING', (0, 0), (-1, 0), 10),
                    
                    # Data rows
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 1), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ])
                
                t.setStyle(table_style)
                elements.append(t)
                
            except Exception as e:
                print(f"❌ Table generation error: {str(e)}")
                import traceback
                traceback.print_exc()
                elements.append(Paragraph(f"Table generation failed: {str(e)}", small_style))
        
        # ===== FOOTER =====
        elements.append(Spacer(1, 30))
        footer_text = "This report was automatically generated by the Real Estate Analysis System."
        elements.append(Paragraph(footer_text, small_style))
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF from buffer
        pdf = buffer.getvalue()
        buffer.close()
        
        # Create HTTP response
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        response['Access-Control-Expose-Headers'] = 'Content-Disposition'
        
        return response
        
    except Exception as e:
        print(f"❌ PDF Generation Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
