from flask import Flask
import os
import logging
from flask import Flask


def create_app():
    app = Flask(__name__,static_folder='static',template_folder='app/templates')
    app.config.from_object('config.Config')

    # Ensure folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_GRAPH_FOLDER'], exist_ok=True)
    #os.makedirs(app.config['OUTPUT_VALID_GRAPH_FOLDER'], exist_ok=True)

    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB limit

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the log level to DEBUG or higher
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to the console
            logging.FileHandler('app.log', mode='a'),  # Log to a file
        ]
    )

    app.logger.info("Starting the Flask application...")

    # Register blueprints
    from .routes import bp
    app.register_blueprint(bp)

    print("Templates folder:", os.path.abspath('app/templates'))
    print("Upload.html exists:", os.path.isfile('app/templates/upload.html'))

    return app
