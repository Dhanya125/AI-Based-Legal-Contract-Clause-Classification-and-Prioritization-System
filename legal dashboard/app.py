# ================================
# LEGAL CLAUSE ANALYZER - FULLY WORKING
# ================================

from flask import Flask, render_template, request, send_file, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pdfplumber
import io
import re
from datetime import datetime
import os

app = Flask(__name__)

# CONFIG
MODEL_PATH = "model/best_model.pt"
CONFIDENCE_THRESHOLD = 50
OTHER_LABEL = "Other"

# LABELS
labels = [
    "Termination For Convenience", "IP Ownership Assignment", "Non Compete",
    "Exclusivity", "License Grant", "Cap On Liability", "Audit Rights",
    "Revenue/Profit Sharing", "Warranty Duration", "Anti Assignment",
    "Confidentiality", "Indemnification", "Non Transferable License",
    "Liquidated Damages", "Post Termination Services", "Uncapped Liability",
    "Minimum Commitment", "Most Favored Nation", "Non Disparagement",
    "No Solicit Of Employees", "No Solicit Of Customers", "Competitive Restriction",
    "Insurance", "Affiliate License", "Covenant Not To Sue", "Volume Restriction",
    "Irrevocable License", "Termination Certificate", "Delivery Obligation", "Governing Law"
]

# PRIORITY
HIGH_PRIORITY = [
    "Termination For Convenience", "IP Ownership Assignment", "Non Compete",
    "Exclusivity", "Cap On Liability", "Confidentiality", "Indemnification",
    "Uncapped Liability", "Competitive Restriction", "Covenant Not To Sue"
]
MEDIUM_PRIORITY = [
    "License Grant", "Audit Rights", "Revenue/Profit Sharing", "Warranty Duration",
    "Anti Assignment", "Non Transferable License", "Liquidated Damages",
    "Post Termination Services", "Minimum Commitment", "Most Favored Nation",
    "Non Disparagement", "No Solicit Of Employees", "No Solicit Of Customers",
    "Insurance", "Affiliate License", "Volume Restriction", "Irrevocable License",
    "Termination Certificate", "Delivery Obligation"
]
LOW_PRIORITY = ["Governing Law"]

# LOAD MODEL
try:
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased", num_labels=len(labels)
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("✓ Model loaded successfully")
    else:
        print(f"⚠ Model file not found at {MODEL_PATH}, using random weights for demo")
        model.eval()
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# ================================
# RULE BOOSTER - SIMPLE WORD MATCHING
# ================================

def rule_boost(text):
    text_lower = text.lower()
    text_upper = text.upper()
    
    # SECTION HEADERS (most reliable)
    header_rules = [
        ("EXCLUSIVITY", "Exclusivity", 95),
        ("TERMINATION AND SURVIVAL", "Termination For Convenience", 95),
        ("LIMITATION OF LIABILITY", "Cap On Liability", 95),
        ("GOVERNING LAW", "Governing Law", 95),
        ("INDEMNIFICATION", "Indemnification", 95),
        ("CONFIDENTIALITY", "Confidentiality", 95),
    ]
    
    for header, label, conf in header_rules:
        if header in text_upper and len(text) < 200:
            return label, conf
    
    # Keyword rules
    keyword_rules = {
        "Anti Assignment": ["may not assign", "shall not assign", "cannot assign", "without consent", "no assignment"],
        "Non Compete": ["will not compete", "shall not compete", "non-compete", "similar platform"],
        "Audit Rights": ["examine", "audit", "books and records", "right to examine"],
        "Cap On Liability": ["limitation of liability", "no liability", "indirect damages", "consequential damages"],
        "Termination For Convenience": ["may terminate", "either party may terminate", "ceases to function"],
        "Indemnification": ["indemnify", "hold harmless", "defend"],
        "Confidentiality": ["confidential", "trade secret", "non-disclosure"],
        "License Grant": ["hereby grants", "license to use", "non-exclusive license"],
        "Liquidated Damages": ["interest on overdue", "late payment", "liquidated damages"],
        "Exclusivity": ["exclusive", "exclusivity", "sole provider"],
        "Governing Law": ["governed by", "laws of", "governing law"],
        "Insurance": ["insurance", "coverage", "insured"],
        "Delivery Obligation": ["deliver", "provide", "furnish"],
    }
    
    for label, keywords in keyword_rules.items():
        for keyword in keywords:
            if keyword in text_lower:
                confidence = 85 + (len(keyword.split()) * 5)
                return label, min(95, confidence)
    
    return None, 0

# ================================
# CLAUSE SPLITTING
# ================================

def split_into_clauses(text):
    # Split by numbered sections
    clauses = re.split(r'\n(?=\d+\.\s+[A-Z])', text)
    if len(clauses) > 1:
        return [c.strip() for c in clauses if len(c.strip()) > 50]
    
    # Split by double newlines
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return [p for p in paragraphs if len(p) > 50]

# ================================
# CLASSIFIER
# ================================

def classify_clause(text):
    # Rule booster first
    rule_label, rule_conf = rule_boost(text)
    
    if rule_label:
        print(f"🎯 RULE: {rule_label}")
        priority = "High" if rule_label in HIGH_PRIORITY else "Medium" if rule_label in MEDIUM_PRIORITY else "Low"
        return rule_label, priority
    
    # ML model
    if model is None:
        return OTHER_LABEL, "Low"
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        label = labels[pred.item()]
        
        # Override obvious errors
        text_lower = text.lower()
        
        if label == "Non Compete" and "assign" in text_lower and "compete" not in text_lower:
            label = "Anti Assignment"
            print(f"🔄 OVERRIDE: → Anti Assignment")
        
        if label == "License Grant" and "liability" in text_lower and "license" not in text_lower:
            label = "Cap On Liability"
            print(f"🔄 OVERRIDE: → Cap On Liability")
        
        if label == "Exclusivity" and any(w in text_lower for w in ["examine", "audit", "books"]):
            label = "Audit Rights"
            print(f"🔄 OVERRIDE: → Audit Rights")
        
        if label == "Volume Restriction" and any(w in text_lower for w in ["terminate", "breach"]):
            label = "Termination For Convenience"
            print(f"🔄 OVERRIDE: → Termination")
        
        if label == "Confidentiality" and "limitation of liability" in text_lower:
            label = "Cap On Liability"
            print(f"🔄 OVERRIDE: → Cap On Liability")
        
        priority = "High" if label in HIGH_PRIORITY else "Medium" if label in MEDIUM_PRIORITY else "Low"
        print(f"🤖 MODEL: {label}")
        
        return label, priority
        
    except Exception as e:
        print(f"Error: {e}")
        return OTHER_LABEL, "Low"

# ================================
# PDF EXTRACTION
# ================================

def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return text

# ================================
# FLASK ROUTES
# ================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400
    
    try:
        text = extract_pdf(file)
        clauses = split_into_clauses(text)
        print(f"\n📄 Found {len(clauses)} clauses\n")
        
        results = []
        for i, clause in enumerate(clauses):
            print(f"\n--- Clause {i+1} ---")
            label, priority = classify_clause(clause)
            results.append({
                "id": i + 1,
                "clause_full": clause,
                "clause_preview": clause[:300] + "..." if len(clause) > 300 else clause,
                "label": label,
                "priority": priority
            })
        
        # Calculate statistics
        high_count = sum(1 for r in results if r['priority'] == 'High')
        medium_count = sum(1 for r in results if r['priority'] == 'Medium')
        low_count = sum(1 for r in results if r['priority'] == 'Low')
        
        return jsonify({
            'success': True,
            'results': results,
            'stats': {
                'total': len(results),
                'high': high_count,
                'medium': medium_count,
                'low': low_count
            }
        })
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/export_pdf')
def export_pdf():
    from weasyprint import HTML
    import json
    results = json.loads(request.args.get('results', '[]'))
    
    html = f"""
    <html>
    <head><title>Legal Analysis Report</title>
    <style>
        body {{ font-family: Arial; margin: 40px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
    </style>
    </head>
    <body>
        <h1>Legal Clause Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Clauses: {len(results)}</p>
        <table>
            <tr><th>#</th><th>Clause</th><th>Label</th><th>Priority</th></tr>
            {''.join(f'<tr><td>{r["id"]}</td><td>{r["clause_preview"]}</td><td>{r["label"]}</td><td>{r["priority"]}</td></tr>' for r in results)}
        </table>
    </body>
    </html>
    """
    pdf = io.BytesIO()
    HTML(string=html).write_pdf(pdf)
    pdf.seek(0)
    return send_file(pdf, download_name="report.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)