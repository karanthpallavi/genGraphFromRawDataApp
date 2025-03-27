Overview
The web application is designed to streamline the transformation of raw Arlanxeo data files (.xlsx format) into a structured and queryable graph database. 
It automates data cleaning, transformation, and graph construction while providing an intuitive user interface for data upload and querying.

Key Features
1.	Raw Data Ingestion:
o	Accepts raw Arlanxeo data files in .xlsx format.
o	Parses and processes the input file for further transformation.
2.	Data Cleaning & Transformation:
o	Transforms duplicate quality names into standardized names.
o	Replaces characters to ensure compliance with IRI standards.
3.	Template Generation & Data Population:
o	Generates template CSV files from Measurement Graph Pattern.
o	Populates the template CSV files using the transformed data from the raw file.
4.	Graph Construction:
o	Builds an OWL graph from the filled template CSV files.
o	Imports the constructed graph into the Graph DB repository named Arlanxeo_Polymers.
5.	User Interface & Functionality:
o	Data Upload: End users can upload raw data files through a web-based UI (available at 127.0.0.1:5000/).
o	Graph Querying: Provides an endpoint (127.0.0.1:5000/query) for querying the graph database.
o	Predefined SPARQL Query: Default SPARQL query allows domain experts to retrieve all measurements related to raw polymer properties.
o	Data Export: Enables downloading query results in CSV or Excel format.

Technology Stack
•	Backend: Python (Flask)
•	Database: GraphDB (Semantic Repository)
•	Data Processing: Pandas, RDFlib
•	Frontend: HTML, JavaScript, Bootstrap

Usage Workflow
1.	User uploads a raw .xlsx file through the web UI.
2.	The system processes the file: 
o	Cleans and transforms data.
o	Generates template CSVs and fills them with transformed data.
o	Constructs an OWL graph and imports it into the GraphDB repository.
3.	User queries the data using the provided SPARQL endpoint – http:// 127.0.0.1:5000/query.
4.	Query results can be downloaded in CSV or Excel format.

Conclusion 
This web application offers a comprehensive solution for processing, transforming, and querying Arlanxeo raw data files efficiently. 
By automating data cleaning and graph construction, it enhances usability for domain experts, enabling seamless data analysis and retrieval from a flexible graph database GraphDB.
