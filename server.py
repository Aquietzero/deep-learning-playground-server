from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from controllers.gridworld import GridWorldController

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

socketio = SocketIO(
    app,
    cors_allowed_origins='*',
    cors_credentials=False)

GridWorldController(app, socketio)

if __name__ == '__main__':
    socketio.run(app, debug=True)
