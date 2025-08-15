from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import os
import re
import psycopg2
from psycopg2.extras import RealDictCursor
import spacy
from spacy.matcher import PhraseMatcher
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app)

# Database connection parameters
DB_PARAMS = {
    'dbname':   os.getenv('PG_DB', 'gisdb'),
    'user':     os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD', 'YourNewP@ssw0rd'),
    'host':     os.getenv('PG_HOST', 'localhost'),
    'port':     os.getenv('PG_PORT', '5432'),
}

# Utility: get database connection
def get_connection():
    return psycopg2.connect(**DB_PARAMS)

# Initialize spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# -- 1. Load boundary names dynamically --
with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM boundaries;")
        boundary_names = [row[0] for row in cur.fetchall()]

# Lowercase list for matching
boundary_names_lower = [name.lower() for name in boundary_names]

boundary_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
boundary_patterns = [nlp.make_doc(name) for name in boundary_names]
boundary_matcher.add("BOUNDARY", boundary_patterns)

# -- 2. Dataset synonyms mapping --
DATASET_SYNONYMS = {
    'coastal flood depth high': 'frm_fh_coastal_depth_h',
    'coastal flood depth medium': 'frm_fh_coastal_depth_m',
    'coastal flood depth low': 'frm_fh_coastal_depth_l',
    'coastal flood extent high': 'frm_fh_coastal_extent_h',
    'coastal flood extent medium': 'frm_fh_coastal_extent_m',
    'coastal flood extent low': 'frm_fh_coastal_extent_l',
    'river flood depth high': 'frm_fh_river_depth_h',
    'river flood depth medium': 'frm_fh_river_depth_m',
    'river flood depth low': 'frm_fh_river_depth_l',
    'river flood extent high': 'frm_fh_river_extent_h',
    'river flood extent medium': 'frm_fh_river_extent_m',
    'river flood extent low': 'frm_fh_river_extent_l',
    'surface water extent high': 'frm_fh_surface_water_extent_h',
    'surface water extent medium': 'frm_fh_surface_water_extent_m',
    'surface water extent low': 'frm_fh_surface_water_extent_l',
    'roads': 'gis_osm_roads_free_1',
    'railways': 'gis_osm_railways_free_1',
    'waterways': 'gis_osm_waterways_free_1',
    'national nature reserves': 'nnr_scotland',
    'ramsar sites': 'ramsar_scotland',
    'special areas of conservation': 'sac_scotland',
    'special protection areas': 'spa_scotland',
    'sites of special scientific interest': 'sssi_scotland',
    'corine land cover': 'u2018_clc2018_v2020_20u1'
}

dataset_keys = list(DATASET_SYNONYMS.keys())
dataset_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
dataset_patterns = [nlp.make_doc(name) for name in dataset_keys]
dataset_matcher.add("DATASET", dataset_patterns)

# -- 3. Regex for numeric+unit parsing --
unit_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(km|kilometers|m|meters|mi|miles)", re.IGNORECASE)
unit_to_meters = {'km':1000, 'kilometers':1000, 'm':1, 'meters':1, 'mi':1609.34, 'miles':1609.34}

def parse_units(text: str) -> float:
    match = unit_pattern.search(text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    return value * unit_to_meters.get(unit, 1)

# -- Fuzzy best match helper --
def best_match(candidates, text, threshold=0.5):
    """Return the candidate with highest similarity to text if above threshold."""
    best = None
    best_ratio = threshold
    for c in candidates:
        ratio = SequenceMatcher(None, text, c).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = c
    return best

# -- 4. Process NL query with fuzzy fallback --
def process_query(text: str) -> dict:
    text_lower = text.lower()
    doc = nlp(text)
    # Exact phrase matches
    b_matches = boundary_matcher(doc)
    d_matches = dataset_matcher(doc)
    boundaries = {doc[start:end].text for _, start, end in b_matches}
    datasets = {doc[start:end].text for _, start, end in d_matches}
    # Fuzzy boundary if needed
    if not boundaries:
        fm = best_match(boundary_names_lower, text_lower, threshold=0.6)
        if fm:
            idx = boundary_names_lower.index(fm)
            boundaries.add(boundary_names[idx])
    # Fuzzy dataset if needed
    if not datasets:
        fm = best_match(dataset_keys, text_lower, threshold=0.6)
        if fm:
            datasets.add(fm)
    distance = parse_units(text)
    irrelevant = not (boundaries and datasets)
    return {
        'boundaries': list(boundaries),
        'datasets': list(datasets),
        'distance_m': distance,
        'irrelevant': irrelevant
    }

# -- 5. Build SQL, execute, return WGS84 GeoJSON with metadata --
def build_and_run(text: str) -> dict:
    pr = process_query(text)
    if pr['irrelevant']:
        return {'error': 'Sorry, this prototype cannot handle that request.'}

    boundary = pr['boundaries'][0].lower()
    dataset_key = pr['datasets'][0]
    table = DATASET_SYNONYMS[dataset_key]
    dist = pr['distance_m']

    sql = f"""
    SELECT t.*, ST_AsGeoJSON(ST_Transform(t.geom,4326))::json AS geometry
    FROM {table} t
    JOIN boundaries b ON t.bndry_id = b.bndry_id
    WHERE LOWER(b.name) = %s
    """
    params = [boundary]
    if dist is not None:
        sql += " AND ST_DWithin(t.geom, b.geom, %s)"
        params.append(dist)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    features = []
    for row in rows:
        props = dict(row)
        geometry = props.pop('geometry', None)
        props.pop('geom', None)
        props.pop('bndry_id', None)
        features.append({'type': 'Feature', 'geometry': geometry, 'properties': props})

    return {
        'type': 'FeatureCollection',
        'metadata': {'source': dataset_key},
        'features': features
    }

# -- Vector layer endpoint --
VECTOR_LAYERS = ['boundaries'] + list(DATASET_SYNONYMS.values())

@app.route('/<layer>/<int:bndry_id>', methods=['GET'])
def get_vector_layer(layer, bndry_id):
    if layer not in VECTOR_LAYERS:
        abort(404, description="Layer not found")
    print(f"📥 Received GET for layer='{layer}', bndry_id={bndry_id}")
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = f"SELECT *, ST_AsGeoJSON(ST_Transform(geom,4326))::json AS geometry FROM {layer} WHERE bndry_id=%s"
            cur.execute(sql, (bndry_id,))
            rows = cur.fetchall()
    print(f"📤 Responding with {len(rows)} features (200 OK)")
    features = []
    for row in rows:
        props = dict(row)
        geometry = props.pop('geometry', None)
        props.pop('geom', None)
        props.pop('bndry_id', None)
        features.append({'type': 'Feature', 'geometry': geometry, 'properties': props})

    return jsonify({
        'type': 'FeatureCollection',
        'metadata': {'source': layer},
        'features': features
    })

# -- Natural language query endpoint with logging --
@app.route('/query', methods=['POST'])
def nl_query():
    data = request.get_json() or {}
    text = data.get('nl', '')
    print(f"📥 Received NL query: '{text}'")
    result = build_and_run(text)
    if 'error' in result:
        print(f"📤 Error: {result['error']}")
        print("📤 Responding 400 Bad Request")
        return jsonify(result), 400
    print("📤 Responding 200 OK")
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
