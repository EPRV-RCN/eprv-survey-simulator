from flask import Flask

app = Flask(__name__)

from . import calc_shell
app.config.from_mapping( SECRET_KEY=b'_\x91\x06U\x93\x06V9Ej\xe4\xb9\xbd\x13]\x03',)
app.register_blueprint(calc_shell.bp)
