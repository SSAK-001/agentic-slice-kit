from __future__ import annotations
import json

import hashlib
import hmac
import html
import os
import secrets
import time
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from demo.impactloop.flow import build_flow
from slice import callback, runner
from slice.config import settings
from slice.store import Store

DB = os.environ.get("SLICE_DB", "impactloop.db")
app = FastAPI(title="ImpactLoop")
SESSIONS: dict[str, dict[str, Any]] = {}


def db() -> Store:
    store = Store(DB)
    store.db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'student',
        created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS problems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        created_by INTEGER NOT NULL,
        run_id TEXT,
        created_at REAL NOT NULL
    );
    """)
    mentor = store.db.execute(
        "SELECT id FROM users WHERE email=?",
        ("mentor@impactloop.local",),
    ).fetchone()
    if mentor is None:
        store.db.execute(
            "INSERT INTO users(name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
            (
                "ImpactLoop Mentor",
                "mentor@impactloop.local",
                hash_password("mentor123"),
                "mentor",
                time.time(),
            ),
        )
    return store


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return salt.hex() + ":" + digest.hex()


def check_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split(":", 1)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 120_000)
        return hmac.compare_digest(candidate.hex(), digest_hex)
    except ValueError:
        return False


def esc(value: Any) -> str:
    return html.escape(str(value))


def current_user(request: Request) -> dict[str, Any] | None:
    token = request.cookies.get("impactloop_session")
    return SESSIONS.get(token) if token else None


def layout(body: str, user: dict[str, Any] | None = None) -> HTMLResponse:
    nav = f'<span class="user">{esc(user["name"])} · {esc(user["role"])}</span> <a href="/logout">Log out</a>' if user else '<a href="/login">Log in</a>'
    return HTMLResponse(f'''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>ImpactLoop</title>
<style>
:root{{--bg:#090b12;--panel:#121827;--line:#29344c;--text:#f6f8fc;--muted:#94a0b8;--cyan:#00e5ff;--violet:#9146ff;--green:#35d07f;--amber:#ffc857;}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 80% 0%,#29114d 0,transparent 35rem),var(--bg);color:var(--text);font:15px system-ui,sans-serif}} nav{{border-bottom:1px solid var(--line);padding:16px 5%;display:flex;justify-content:space-between;align-items:center;background:#090b12e8;position:sticky;top:0}} nav a{{color:var(--cyan);text-decoration:none;margin-left:18px}} .brand{{font-weight:900;font-size:21px}} main{{max-width:1120px;margin:auto;padding:38px 22px 80px}} h1{{font-size:clamp(32px,6vw,64px);letter-spacing:-.06em;margin:0 0 12px}} h2{{letter-spacing:-.03em}} .muted,.user{{color:var(--muted)}} .grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}} .card{{grid-column:span 12;background:linear-gradient(145deg,#171f31,#0f1420);border:1px solid var(--line);border-radius:18px;padding:22px}} .half{{grid-column:span 6}} .third{{grid-column:span 4}} label{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em;margin:14px 0 6px;font-weight:800}} input,textarea,select{{width:100%;padding:13px;border:1px solid var(--line);border-radius:10px;background:#0b101b;color:var(--text);font:inherit}} textarea{{min-height:130px}} button,.button{{display:inline-block;margin-top:16px;padding:12px 16px;border:0;border-radius:10px;background:linear-gradient(135deg,var(--cyan),#83f5ff);color:#05070c;font-weight:900;text-decoration:none;cursor:pointer}} .violet{{background:linear-gradient(135deg,var(--violet),#c19aff);color:white}} .danger{{color:#ff8192}} .ok{{color:var(--green)}} .warning{{color:var(--amber)}} .badge{{display:inline-block;border-radius:999px;padding:6px 10px;background:#24304a;color:var(--cyan);font-size:12px;font-weight:800}} .problem{{padding:16px 0;border-bottom:1px solid var(--line)}} .problem:last-child{{border:0}} .problem h3{{margin:0 0 6px}} .steps{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}} .step{{padding:9px 12px;border-radius:999px;background:#202a40;color:#cbd7ea;font-size:12px}} pre{{white-space:pre-wrap;color:#d5deed;background:#0a0f18;padding:14px;border-radius:10px;overflow:auto}} @media(max-width:760px){{.half,.third{{grid-column:span 12}}}}
</style></head><body><nav><div class="brand">↗ ImpactLoop</div><div>{nav}</div></nav><main>{body}</main></body></html>''')


def require_user(request: Request) -> dict[str, Any] | RedirectResponse:
    user = current_user(request)
    return user or RedirectResponse("/login", status_code=303)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = current_user(request)
    if not user:
        return layout('<div class="card"><p class="badge">AGENTIC CAMPUS PLATFORM</p><h1>Turn campus problems into proven ability.</h1><p class="muted">Students create real work, mentors guide the hard decisions, and agents verify the evidence.</p><a class="button" href="/signup">Create account</a><a class="button violet" href="/login">Sign in</a></div>', None)
    s = db()
    problems = s.db.execute("SELECT * FROM problems ORDER BY created_at DESC").fetchall()
    rows = "".join(f'<div class="problem"><h3>{esc(p["title"])}</h3><p class="muted">{esc(p["description"])}</p><span class="badge">{esc(p["run_id"] or "Not started")}</span> <a class="button" href="/run/{esc(p["id"])}">Open</a></div>' for p in problems)
    mentor_link = '<a class="button violet" href="/mentor">Mentor queue</a>' if user["role"] == "mentor" else ""
    return layout(f'<div class="grid"><section class="card"><p class="badge">WORKSPACE</p><h1>Hello, {esc(user["name"])}.</h1><p class="muted">Create a challenge and let ImpactLoop coordinate the next steps.</p><a class="button" href="/problems/new">Create a problem</a>{mentor_link}</section><section class="card"><h2>Campus problems</h2>{rows or "<p class=muted>No problems yet. Create the first one.</p>"}</section></div>', user)


@app.get("/signup", response_class=HTMLResponse)
def signup_page():
    return layout('<div class="card half"><h2>Create student account</h2><form method="post"><label>Name</label><input name="name" required><label>Email</label><input type="email" name="email" required><label>Password</label><input type="password" name="password" minlength="6" required><button>Create account</button></form><p class="muted">Mentors use the separate mentor login.</p></div>')


@app.post("/signup")
def signup(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    s = db()
    try:
        s.db.execute("INSERT INTO users(name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)", (name.strip(), email.lower().strip(), hash_password(password), "student", time.time()))
    except Exception:
        return RedirectResponse("/signup?error=exists", status_code=303)
    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return layout("""<div class="grid"><section class="card half"><h2>Sign in</h2><form method="post" action="/login"><label>Email</label><input type="email" name="email" required><label>Password</label><input type="password" name="password" required><button>Sign in</button></form></section><section class="card half"><h2>Demo mentor access</h2><p class="muted">For the hackathon demo, create this account once:</p><pre>mentor@impactloop.local\nmentor123</pre><p class="warning">Change this before public deployment.</p></section></div>""")


@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    s = db()
    row = s.db.execute("SELECT * FROM users WHERE email=?", (email.lower().strip(),)).fetchone()
    if not row or not check_password(password, row["password_hash"]):
        return RedirectResponse("/login?error=invalid", status_code=303)
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = dict(row)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("impactloop_session", token, httponly=True, samesite="lax")
    return response


@app.get("/mentor")
def mentor(request: Request):
    user = require_user(request)
    if isinstance(user, RedirectResponse) or user["role"] != "mentor":
        return RedirectResponse("/login", status_code=303)
    s = db()
    qs = callback.pending(s)
    cards = "".join(f'<div class="problem"><h3>{esc(q.question)}</h3><p class="muted">Run: {esc(q.run_id)}</p><form method="post" action="/mentor/{esc(q.id)}"><textarea name="answer" required placeholder="Your decision..."></textarea><button>Submit decision</button></form></div>' for q in qs)
    return layout(f'<section class="card"><p class="badge">MENTOR WORKSPACE</p><h1>Decisions that keep students moving.</h1><p class="muted">These are questions agents cannot safely decide alone.</p>{cards or "<p class=ok>No decisions waiting.</p>"}</section>', user)


@app.post("/mentor/{qid}")
def mentor_answer(request: Request, qid: str, answer: str = Form(...)):
    user = require_user(request)
    if isinstance(user, RedirectResponse) or user["role"] != "mentor":
        return RedirectResponse("/login", status_code=303)
    callback.answer(db(), qid, answer.strip(), who=user["email"])
    return RedirectResponse("/mentor", status_code=303)


@app.get("/problems/new", response_class=HTMLResponse)
def new_problem_page(request: Request):
    user = require_user(request)
    if isinstance(user, RedirectResponse): return user
    return layout('<section class="card"><p class="badge">NEW CHALLENGE</p><h1>What should students solve?</h1><form method="post"><label>Challenge title</label><input name="title" placeholder="Students miss useful campus events" required><label>Problem description</label><textarea name="description" placeholder="Describe the real campus problem and why it matters." required></textarea><button>Launch agent workflow</button></form></section>', user)


@app.post("/problems/new")
def new_problem(request: Request, title: str = Form(...), description: str = Form(...)):
    user = require_user(request)
    if isinstance(user, RedirectResponse): return user
    s = db()
    cur = s.db.execute("INSERT INTO problems(title,description,created_by,created_at) VALUES(?,?,?,?)", (title.strip(), description.strip(), user["id"], time.time()))
    problem_id = cur.lastrowid
    run_id = s.create_run("impactloop", {"problem_id": problem_id, "title": title.strip(), "created_by": user["email"]})
    s.db.execute("UPDATE problems SET run_id=? WHERE id=?", (run_id, problem_id))
    s.append(run_id, "input", {"title": title.strip(), "text": description.strip()}, produced_by=f"user:{user['email']}")
    # Live models are used here. The existing Budget enforces SLICE_MAX_TOKENS_PER_RUN.
    runner.advance(s, run_id, build_flow(), settings())
    return RedirectResponse(f"/run/{problem_id}", status_code=303)


@app.get("/run/{problem_id}", response_class=HTMLResponse)
def run_page(request: Request, problem_id: int):
    user = require_user(request)
    if isinstance(user, RedirectResponse): return user
    s = db()
    problem = s.db.execute("SELECT * FROM problems WHERE id=?", (problem_id,)).fetchone()
    if not problem: return RedirectResponse("/", status_code=303)
    records = s.replay(problem["run_id"])
    labels = " → ".join(r.kind.replace("_", " ").title() for r in records if r.kind not in {"input"})
    final = records[-1].payload if records else {}
    return layout(f'<section class="card"><p class="badge">LIVE PROJECT</p><h1>{esc(problem["title"])}</h1><p class="muted">{esc(problem["description"])}</p><div class="steps">{esc(labels)}</div><p><span class="badge">{esc(s.get_state(problem["run_id"]).value)}</span></p><h2>Latest agent output</h2><pre>{esc(final)}</pre><a class="button" href="/">Back to workspace</a></section>', user)


@app.get("/logout")
def logout(request: Request):
    token = request.cookies.get("impactloop_session")
    if token: SESSIONS.pop(token, None)
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie("impactloop_session")
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



@app.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request):
    user = require_user(request)
    if isinstance(user, RedirectResponse):
        return user

    store = db()
    row = store.db.execute(
        "SELECT * FROM student_profiles WHERE user_id=?",
        (user["id"],),
    ).fetchone()

    profile = dict(row) if row else {
        "skills": "[]",
        "interests": "[]",
        "availability": "",
        "preferred_role": "",
        "bio": "",
        "evidence_links": "[]",
    }

    skills = ", ".join(json.loads(profile["skills"]))
    interests = ", ".join(json.loads(profile["interests"]))
    links = "\n".join(json.loads(profile["evidence_links"]))

    return layout(f"""
    <section class="card">
      <p class="badge">STUDENT PROFILE</p>
      <h1>Make your abilities discoverable.</h1>
      <p class="muted">
        ImpactLoop uses this information to match you to meaningful campus work.
      </p>

      <form method="post" action="/profile">
        <label>Skills</label>
        <input name="skills" value="{esc(skills)}"
               placeholder="Python, Figma, research, communication" required>

        <label>Interests</label>
        <input name="interests" value="{esc(interests)}"
               placeholder="student experience, sustainability, design">

        <label>Weekly availability</label>
        <input name="availability" value="{esc(profile["availability"])}"
               placeholder="6 hours per week" required>

        <label>Preferred role</label>
        <input name="preferred_role" value="{esc(profile["preferred_role"])}"
               placeholder="UX designer, researcher, data organiser">

        <label>Short bio</label>
        <textarea name="bio" required>{esc(profile["bio"])}</textarea>

        <label>Evidence or project links</label>
        <textarea name="evidence_links"
                  placeholder="One link or project name per line">{esc(links)}</textarea>

        <button>Save profile</button>
      </form>
    </section>
    """, user)


@app.post("/profile")
def save_profile(
    request: Request,
    skills: str = Form(...),
    interests: str = Form(""),
    availability: str = Form(...),
    preferred_role: str = Form(""),
    bio: str = Form(""),
    evidence_links: str = Form(""),
):
    user = require_user(request)
    if isinstance(user, RedirectResponse):
        return user

    store = db()

    def split_lines(value: str) -> list[str]:
        return [
            item.strip()
            for item in value.replace(",", "\n").splitlines()
            if item.strip()
        ]

    values = (
        user["id"],
        json.dumps(split_lines(skills)),
        json.dumps(split_lines(interests)),
        availability.strip(),
        preferred_role.strip(),
        bio.strip(),
        json.dumps(split_lines(evidence_links)),
        time.time(),
    )

    store.db.execute(
        """
        INSERT INTO student_profiles
        (user_id, skills, interests, availability, preferred_role, bio,
         evidence_links, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          skills=excluded.skills,
          interests=excluded.interests,
          availability=excluded.availability,
          preferred_role=excluded.preferred_role,
          bio=excluded.bio,
          evidence_links=excluded.evidence_links
        """,
        values,
    )

    return RedirectResponse("/", status_code=303)
