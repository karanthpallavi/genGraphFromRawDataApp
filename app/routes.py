from flask import Blueprint, render_template, request, Response, current_app,jsonify, send_file
import os
import time
from werkzeug.utils import secure_filename
from .utils import generate_output_filename, readIndivSheetsTransformToCSV, preprocessSSBRRequestsFile, \
    readOntologyCSVAndBuildDataTriples, process_directory, upload_to_graphdb, run_sparql_query
from os import listdir
from os.path import isfile, join
import csv
import io
#from bs4 import BeautifulSoup
import pandas as pd

bp = Blueprint('main', __name__, template_folder='templates')

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

# Streaming generator for progress
def generate_progress_messages():
    steps = [
        "Starting file upload...",
        "Validating file...",
        "Saving file...",
        "Transforming raw data file to specific template csv files...",
        "Generating graph files in ttl format from template csv files...",
        "Graph files generated, Please check the output folder for template csv files generated and output/output_valid_graphs folder for graph files!...",
        "Uploading graph files to GraphDB Repository",
        "Use endpoint - http://127.0.0.1:5000/query to query the data in GraphDB"
    ]
    for step in steps:
        yield f"data:{step}\n\n"
        time.sleep(1)  # Simulate processing time

@bp.route("/generate_csv", methods=["POST"])
def generate_csv():
    try:
        # Get table data from request
        data = request.json.get("data", [])
        if not data:
            return jsonify({"error": "No data received"}), 400

        # Convert to DataFrame and save as CSV in memory
        df = pd.DataFrame(data[1:], columns=data[0])  # First row is headers
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Send the CSV file as a response
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="query_results.csv"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/generate_excel", methods=["POST"])
def generate_excel():
    try:
        data = request.json.get("data", [])
        if not data or len(data) < 2:  # Ensure there are headers and at least one row
            return jsonify({"error": "No valid data received"}), 400

        df = pd.DataFrame(data[1:], columns=data[0])  # First row is headers

        # Create an in-memory BytesIO buffer for the Excel file
        output = io.BytesIO()

        # Use `ExcelWriter` to properly write the DataFrame into an Excel file
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="QueryResults")
            writer.book.close()  # Ensure all writes are properly closed

        # Move back to the beginning of the buffer so Flask can read it
        output.seek(0)

        return send_file(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="query_results.xlsx"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Home route
@bp.route('/')
def home():
    return render_template('upload.html')

# Upload route
@bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        current_app.logger.warning("No file part in the request.")  # Log at WARNING level
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        current_app.logger.warning("No file selected for uploading.")  # Log at WARNING level
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "Invalid file type. Please upload an excel file", 400

    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        current_app.logger.info(f"File uploaded: {file.filename}")  # Log at INFO level
        templateFilePath = os.path.join(current_app.config['UPLOAD_FOLDER'],'templateFile.csv')

        # Read the Excel file into a Pandas DataFrame
        #df = pd.read_excel(filepath)

        # Call generate schema functions here
        preprocessSSBRRequestsFile(file_path,templateFilePath)

        owlFilePath = os.path.join(current_app.config['UPLOAD_FOLDER'],'digitrubber-full.ttl')

        #Call to generate ttl files for generated/transformed csv files
        onlyfiles = [f for f in listdir(current_app.config['OUTPUT_FOLDER']) if isfile(join(current_app.config['OUTPUT_FOLDER'], f))]
        current_app.logger.info(f"File to generate Graph from:",onlyfiles)

        for eachFile in onlyfiles:
            eachFileFullPath = (current_app.config['OUTPUT_FOLDER'])+"//"+eachFile
            fileNameWithoutExt = eachFile[:-4]
            #print("File name without ext ",fileNameWithoutExt)
            readOntologyCSVAndBuildDataTriples(owlFilePath,eachFileFullPath,fileNameWithoutExt)
        input_directory = current_app.config['OUTPUT_GRAPH_FOLDER']
        output_directory = "output_valid_graphs/"
        process_directory(input_directory, output_directory)

        # Upload TTL files to GraphDB with secure credentials
        username = current_app.config.get('GRAPHDB_USERNAME')
        password = current_app.config.get('GRAPHDB_PASSWORD')
        for ttl_file in os.listdir(output_directory):
            ttl_path = os.path.join(output_directory, ttl_file)
            graphdb_repo_url = f"http://localhost:7200/repositories/{current_app.config['GRAPHDB_REPO']}"
            upload_to_graphdb(graphdb_repo_url, ttl_path, username, password)
            #upload_to_graphdb(current_app.config['GRAPHDB_REPO'], ttl_path, username, password)

        return Response(generate_progress_messages(), content_type='text/event-stream')

    except Exception as e:
        return f"An error occurred: {e}"

@bp.route('/query', methods=['GET', 'POST'])
def query_graphdb_old():
    if request.method == 'GET':
        example_query = """
SELECT ?s ?p ?o
WHERE {
  ?s ?p ?o
}
LIMIT 10
"""
        return render_template('query_form.html', example_query=example_query)

    elif request.method == 'POST':
        query = request.form.get('query', '')
        if not query:
            return "SPARQL query is required.", 400

        try:
            repo_url = f"http://localhost:7200/repositories/{current_app.config['GRAPHDB_REPO']}"
            username = current_app.config.get('GRAPHDB_USERNAME')
            password = current_app.config.get('GRAPHDB_PASSWORD')

            results = run_sparql_query(repo_url, query, username, password)

            print("SPARQL Results:", results)  # Debug: Print results in the terminal

            return render_template('query_results.html', results=results)
        except Exception as e:
            print("Error:", e)  # Debug: Print error in the terminal
            return f"An error occurred: {e}", 500

@bp.route('/query', methods=['POST'])
def query_graphdb():
    schema = request.form.get("schema", "")
    measurement_type = request.form.get("measurement_type", "")
    selected_slots = request.form.getlist("measurement_slots[]")  # ✅ Read multiple selected slots

    if schema != "measurement" or not selected_slots:
        return "Measurement schema and at least one slot are required.", 400

    # 🔥 Build SPARQL Query Dynamically
    slot_filters = " ".join([f"?slot = \"{slot}\"^^xsd:string ||" for slot in selected_slots])
    slot_filters = slot_filters.rstrip(" ||")  # Remove last OR condition
    print(slot_filters)

    # 🔥 Generate a Natural Language Query (NLQ)
    slot_list = ", ".join(selected_slots)
    nl_query = f"Show measurements for {slot_list}."

    try:


        # ✅ Pass the NLQ back to the form
        return render_template("query_form.html", nl_query=nl_query)
    except Exception as e:
        return f"An error occurred: {e}", 500

@bp.route('/get_slots', methods=['GET'])
def get_slots():
    slots = []
    csv_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'filtered_slots.csv')

    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)  # Read CSV with headers
            for row in reader:
                if "Slot Name" in row and "Range Name" in row:  # Ensure expected columns exist
                    slots.append({"slot_name": row["Slot Name"], "range_name": row["Range Name"]})

        return jsonify(slots)
    except Exception as e:
        return jsonify({"error": str(e)}), 500