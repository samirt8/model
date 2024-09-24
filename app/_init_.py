from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Configuration de l'application (si nécessaire)
    app.config.from_object('config.Config')
    
    # Importation des routes
    from . import routes
    app.register_blueprint(routes.bp)

    return app

