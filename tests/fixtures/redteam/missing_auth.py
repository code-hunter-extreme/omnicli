from flask import Flask, request

app = Flask(__name__)


@app.post("/admin/delete")
def delete_user():
    payload = request.get_json()
    delete_user_by_id(payload["user_id"])
    return {"deleted": True}
