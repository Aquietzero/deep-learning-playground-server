from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from controllers.gridworld import GridWorldController

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

@app.route('/')
def hello_world():
    res = {
        'msg': 'ok',
        'data': [1, 2, 3, 4, 5]
    }
    return res

GridWorldController(app)

socketio = SocketIO(
    app,
    cors_allowed_origins='*',
    cors_credentials=False)

if __name__ == '__main__':
    socketio.run(app)
