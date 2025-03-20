import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key')
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    OUTPUT_FOLDER = os.path.join(os.getcwd(), 'output')
    OUTPUT_GRAPH_FOLDER = os.path.join(os.getcwd(),'output/output_graphs')
    TEMPLATE_FOLDER = os.path.join(os.getcwd(),'templates')
    GRAPHDB_REPO = 'ArlanxeoPolymers'
    GRAPHDB_USERNAME = 'test'
    GRAPHDB_PASSWORD = '1234'
    #OUTPUT_VALID_GRAPH_FOLDER = os.path.join(os.getcwd(),'output_graphs/output_valid_graphs')
    ALLOWED_EXTENSIONS = {'xls', 'xlsx'}
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB limit
