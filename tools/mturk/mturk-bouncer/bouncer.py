import sqlite3
from flask import g
from flask import request
from flask import jsonify
import os

from flask import Flask

app = Flask(__name__)

DATABASE = "./data/database.db"


def get_db():
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


@app.route("/requestaccess", methods=["GET"])
def request_access():
    """Checks whether the worker has already participated in the experiment"""

    worker_id = request.args.get("wid")
    task_namespace = request.args.get("tns")
    experiment_id = request.args.get("eid")

    if len(worker_id) > 0:
        db = get_db()
        cursor = db.cursor()
        access = check_access(cursor, worker_id, task_namespace, experiment_id)
    else:
        access = False
    result = jsonify(access=access)

    return result


def check_access(cursor, worker_id, task_namespace, experiment_id):
    """Check if there already exists a row for this worker and experiment combination"""
    cursor.execute(
        "SELECT count(*) FROM workers WHERE worker_id = ? AND task_namespace = ? AND experiment_id = ?",
        (worker_id, task_namespace, experiment_id),
    )
    return cursor.fetchone()[0] == 0


@app.route("/ban", methods=["GET"])
def ban():
    """Bans a worker form participating again in the experiment.
    If the worker has already been banned, do not do anything but return a 403 error.
    """

    worker_id = request.args.get("wid")
    task_namespace = request.args.get("tns")
    experiment_id = request.args.get("eid")

    db = get_db()
    cursor = db.cursor()
    if not check_access(cursor, worker_id, task_namespace, experiment_id):
        resp = jsonify(success=False)
        resp.status_code = 403
        return resp

    cursor.execute(
        "insert into workers (worker_id, task_namespace, experiment_id) values (?,?,?)",
        (worker_id, task_namespace, experiment_id),
    )
    db.commit()

    return jsonify(success=True)


def init_db():
    """Initialize the SQLite3 database"""

    if os.path.exists(DATABASE):
        return

    with app.app_context():
        db = get_db()
        with app.open_resource("schema.sql", mode="r") as f:
            db.cursor().executescript(f.read())
        db.commit()


if __name__ == "__main__":
    init_db()
