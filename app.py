"""
app.py — AI Job Hunting Copilot Databricks App.

Routes:
  GET  /                 chat interface (home)
  POST /api/chat          send a message to the agent, get a reply + actions taken
  GET  /pipeline          view saved jobs by stage, add interview notes
  POST /pipeline/stage     update a job's pipeline stage
  POST /pipeline/note      add an interview note
  GET  /profile           view/edit profile + skills
  POST /profile            save profile + skills
"""

import os
import traceback

from flask import Flask, jsonify, render_template, request, session

import agent
import agent_tools as T

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")


def _session_history() -> list[dict]:
    return session.get("chat_history", [])


def _save_history(history: list[dict]) -> None:
    # Keep only the last few turns so the request payload stays small.
    session["chat_history"] = history[-12:]


@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message can't be empty."}), 400

    try:
        user = T.get_or_create_demo_user()
        history = _session_history()
        result = agent.run_agent(user["user_id"], message, history)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": result["reply"]})
        _save_history(history)

        return jsonify({"reply": result["reply"], "actions": result["actions"]})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": f"Something went wrong: {e}"}), 500


@app.route("/api/chat/reset", methods=["POST"])
def api_chat_reset():
    session["chat_history"] = []
    return jsonify({"ok": True})


@app.route("/pipeline")
def pipeline():
    user = T.get_or_create_demo_user()
    stage_filter = request.args.get("stage") or None
    jobs = T.list_pipeline(user["user_id"], stage=stage_filter)
    stale = T.stale_applications(user["user_id"], days=7)
    return render_template("pipeline.html", jobs=jobs, stale=stale, stage_filter=stage_filter)


@app.route("/pipeline/stage", methods=["POST"])
def pipeline_stage():
    user = T.get_or_create_demo_user()
    job_id = request.form.get("job_id")
    new_stage = request.form.get("new_stage")
    try:
        T.update_stage(user["user_id"], job_id, new_stage)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True})


@app.route("/pipeline/note", methods=["POST"])
def pipeline_note():
    user = T.get_or_create_demo_user()
    job_id = request.form.get("job_id")
    note_text = (request.form.get("note_text") or "").strip()
    follow_up_at = request.form.get("follow_up_at") or None
    if not note_text:
        return jsonify({"error": "Note text can't be empty."}), 400
    try:
        T.add_interview_note(user["user_id"], job_id, note_text, follow_up_at)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True})


@app.route("/profile", methods=["GET", "POST"])
def profile():
    user = T.get_or_create_demo_user()

    if request.method == "POST":
        target_roles = [r.strip() for r in (request.form.get("target_roles") or "").split(",") if r.strip()]
        target_locations = [l.strip() for l in (request.form.get("target_locations") or "").split(",") if l.strip()]
        min_salary = request.form.get("min_salary") or None
        remote_only = request.form.get("remote_only") == "on"

        T.update_profile(
            user["user_id"],
            headline=request.form.get("headline") or None,
            summary=request.form.get("summary") or None,
            target_roles=target_roles or None,
            target_locations=target_locations or None,
            min_salary=float(min_salary) if min_salary else None,
            remote_only=remote_only,
        )

        skills = []
        names = request.form.getlist("skill_name")
        years = request.form.getlist("skill_years")
        for name, yrs in zip(names, years):
            if name.strip():
                skills.append({
                    "skill_name": name.strip(),
                    "years_experience": float(yrs) if yrs else None,
                })
        T.set_skills(user["user_id"], skills)

    profile_data = T.get_profile(user["user_id"])
    skills_data = T.get_skills(user["user_id"])
    return render_template("profile.html", profile=profile_data, skills=skills_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
