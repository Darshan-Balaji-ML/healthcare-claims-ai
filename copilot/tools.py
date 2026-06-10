import joblib
import pandas as pd
import numpy as np

# Load the trained denial predictor from Phase 1
DENIAL_MODEL = joblib.load('models/denial_predictor_lr.pkl')

# ICD chapter to CPT category compatibility map
ICD_CPT_COMPATIBILITY = {
    'E': ['99000-99499', '80000-89999'],   # Endocrine -> E&M, lab
    'I': ['93000-93799', '99000-99499'],   # Circulatory -> Cardiology, E&M
    'J': ['94000-94799', '99000-99499'],   # Respiratory -> Pulmonary, E&M
    'M': ['27000-29999', '99000-99499'],   # Musculoskeletal -> Ortho, E&M
    'F': ['90000-90899', '99000-99499'],   # Mental health -> Psych, E&M
    'Z': ['99000-99499', '90000-90899'],   # Preventive -> E&M, vaccines
}

def validate_icd_cpt_pair(icd_code: str, cpt_code: str) -> dict:
    """
    Checks if an ICD-10 and CPT code pairing is clinically consistent.
    Returns compatibility status and explanation.
    """
    icd_chapter = icd_code[0].upper()
    try:
        cpt_num = int(cpt_code)
    except ValueError:
        return {'compatible': False, 'reason': f'Invalid CPT code format: {cpt_code}'}

    valid_ranges = ICD_CPT_COMPATIBILITY.get(icd_chapter, [])
    for r in valid_ranges:
        low, high = map(int, r.split('-'))
        if low <= cpt_num <= high:
            return {
                'compatible': True,
                'icd_code': icd_code,
                'cpt_code': cpt_code,
                'reason': f'ICD chapter {icd_chapter} is clinically consistent with CPT range {r}',
            }
    return {
        'compatible': False,
        'icd_code': icd_code,
        'cpt_code': cpt_code,
        'reason': f'ICD chapter {icd_chapter} has no known clinical association with CPT {cpt_code}. This may trigger a medical necessity denial.',
    }


def run_denial_predictor(billed_amount: float, days_to_submission: int,
                         icd_chapter: str, cpt_category: str,
                         provider_specialty: str) -> dict:
    """
    Runs the trained claim denial prediction model.
    Returns denial probability and top risk factors.
    """
    # Build a row with all expected columns set to 0
    columns = [
        'billed_to_median_ratio', 'high_cost_outlier', 'duplicate_flag', 'provider_denial_rate',
        'provider_specialty_Cardiology', 'provider_specialty_Dermatology',
        'provider_specialty_Emergency Medicine', 'provider_specialty_Family Practice',
        'provider_specialty_Gastroenterology', 'provider_specialty_Internal Medicine',
        'provider_specialty_Neurology', 'provider_specialty_Oncology',
        'provider_specialty_Ophthalmology', 'provider_specialty_Orthopedic Surgery',
        'provider_specialty_Pain Management', 'provider_specialty_Physical Therapy',
        'provider_specialty_Psychiatry', 'provider_specialty_Radiology',
        'provider_specialty_Urology',
        'patient_gender_F', 'patient_gender_M',
        'filing_delay_bucket_0-7', 'filing_delay_bucket_8-30',
        'filing_delay_bucket_31-60', 'filing_delay_bucket_61+',
        'patient_age_group_<30', 'patient_age_group_30-50',
        'patient_age_group_50-65', 'patient_age_group_65-80', 'patient_age_group_80+',
        'icd_chapter_Chapter II - Neoplasms', 'icd_chapter_Chapter IV - Endocrine/Metabolic',
        'icd_chapter_Chapter IX - Circulatory', 'icd_chapter_Chapter V - Mental Health',
        'icd_chapter_Chapter VI - Nervous System', 'icd_chapter_Chapter X - Respiratory',
        'icd_chapter_Chapter XI - Digestive', 'icd_chapter_Chapter XII - Skin',
        'icd_chapter_Chapter XIII - Musculoskeletal', 'icd_chapter_Chapter XIV - Genitourinary',
        'icd_chapter_Chapter XIX - Injury', 'icd_chapter_Chapter XXI - Preventive',
    ]
    row = {col: 0 for col in columns}

    # Fill in what we know from the agent input
    median_billed = 150.0  # reasonable default median
    row['billed_to_median_ratio'] = billed_amount / median_billed
    row['high_cost_outlier'] = 1 if billed_amount > 500 else 0

    # Filing delay bucket
    if days_to_submission <= 7:
        row['filing_delay_bucket_0-7'] = 1
    elif days_to_submission <= 30:
        row['filing_delay_bucket_8-30'] = 1
    elif days_to_submission <= 60:
        row['filing_delay_bucket_31-60'] = 1
    else:
        row['filing_delay_bucket_61+'] = 1

    # Provider specialty
    specialty_col = f'provider_specialty_{provider_specialty}'
    if specialty_col in row:
        row[specialty_col] = 1

    # ICD chapter mapping
    icd_chapter_map = {
        'E': 'icd_chapter_Chapter IV - Endocrine/Metabolic',
        'I': 'icd_chapter_Chapter IX - Circulatory',
        'J': 'icd_chapter_Chapter X - Respiratory',
        'M': 'icd_chapter_Chapter XIII - Musculoskeletal',
        'F': 'icd_chapter_Chapter V - Mental Health',
        'Z': 'icd_chapter_Chapter XXI - Preventive',
    }
    icd_col = icd_chapter_map.get(icd_chapter.upper())
    if icd_col:
        row[icd_col] = 1

    # Default age group and gender
    row['patient_age_group_50-65'] = 1
    row['patient_gender_M'] = 1

    try:
        input_data = pd.DataFrame([row])
        prob = DENIAL_MODEL.predict_proba(input_data)[0][1]
        risk_level = 'HIGH' if prob > 0.6 else 'MEDIUM' if prob > 0.3 else 'LOW'
        return {
            'denial_probability': round(float(prob), 3),
            'risk_level': risk_level,
            'main_risk_factors': [
                'Late filing' if days_to_submission > 60 else None,
                'High billed amount' if billed_amount > 500 else None,
            ],
        }
    except Exception as e:
        return {'error': str(e), 'note': 'Model input format mismatch - check training schema'}


def lookup_icd_code(description: str) -> dict:
    """
    Looks up ICD-10 codes matching a plain-English diagnosis description.
    Returns the most likely ICD chapter and example codes.
    """
    description_lower = description.lower()
    matches = []
    keyword_map = {
        'diabetes':     ('E', 'Chapter IV: Endocrine',          ['E11.9', 'E11.65', 'E11.40']),
        'heart':        ('I', 'Chapter IX: Circulatory',        ['I21.0', 'I50.9', 'I10']),
        'chest pain':   ('I', 'Chapter IX: Circulatory',        ['I20.9', 'R07.9']),
        'hypertension': ('I', 'Chapter IX: Circulatory',        ['I10', 'I11.9']),
        'pneumonia':    ('J', 'Chapter X: Respiratory',         ['J18.9', 'J18.1']),
        'asthma':       ('J', 'Chapter X: Respiratory',         ['J45.20', 'J45.40']),
        'depression':   ('F', 'Chapter V: Mental health',       ['F32.1', 'F33.0']),
        'anxiety':      ('F', 'Chapter V: Mental health',       ['F41.1', 'F41.9']),
        'knee':         ('M', 'Chapter XIII: Musculoskeletal',  ['M17.11', 'M25.361']),
        'back pain':    ('M', 'Chapter XIII: Musculoskeletal',  ['M54.5', 'M54.4']),
        'vaccination':  ('Z', 'Preventive care',                ['Z23', 'Z00.00']),
        'checkup':      ('Z', 'Preventive care',                ['Z00.00', 'Z00.01']),
    }
    for keyword, (chapter, chapter_name, codes) in keyword_map.items():
        if keyword in description_lower:
            matches.append({
                'keyword_matched': keyword,
                'icd_chapter': chapter,
                'chapter_name': chapter_name,
                'example_codes': codes,
            })
    if not matches:
        return {'found': False, 'note': 'No matching ICD chapter found. Please consult the full ICD-10 index.'}
    return {'found': True, 'matches': matches}

# Tool schemas — Claude reads 'description' fields to decide when to call each tool
TOOL_SCHEMAS = [
    {
        'name': 'validate_icd_cpt_pair',
        'description': 'Checks if an ICD-10 diagnosis code and a CPT procedure code are clinically compatible. Use this whenever the user provides both an ICD code and a CPT code and asks about medical necessity or potential denial risk.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'icd_code': {'type': 'string', 'description': 'ICD-10 diagnosis code, e.g. E11.9'},
                'cpt_code': {'type': 'string', 'description': 'CPT procedure code, e.g. 99213'},
            },
            'required': ['icd_code', 'cpt_code'],
        },
    },
    {
        'name': 'run_denial_predictor',
        'description': 'Runs the trained machine learning model to predict the probability that a claim will be denied. Use when the user provides specific claim details and wants a denial risk assessment.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'billed_amount': {'type': 'number', 'description': 'Amount billed in dollars'},
                'days_to_submission': {'type': 'integer', 'description': 'Days between service date and claim submission'},
                'icd_chapter': {'type': 'string', 'description': 'First letter of the ICD-10 code, e.g. E for E11.9'},
                'cpt_category': {'type': 'string', 'description': 'CPT code range category, e.g. E&M'},
                'provider_specialty': {'type': 'string', 'description': 'Provider specialty, e.g. Internal Medicine'},
            },
            'required': ['billed_amount', 'days_to_submission', 'icd_chapter', 'cpt_category', 'provider_specialty'],
        },
    },
    {
        'name': 'lookup_icd_code',
        'description': 'Looks up ICD-10 codes matching a plain-English description of a diagnosis or condition. Use when the user describes a medical condition and wants to know the relevant ICD codes.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'description': {'type': 'string', 'description': 'Plain-English description of the condition, e.g. type 2 diabetes with complications'},
            },
            'required': ['description'],
        },
    },
]

# Tool dispatcher — maps tool name to function
TOOL_FUNCTIONS = {
    'validate_icd_cpt_pair': validate_icd_cpt_pair,
    'run_denial_predictor': run_denial_predictor,
    'lookup_icd_code': lookup_icd_code,
}