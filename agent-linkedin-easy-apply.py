"""
LinkedIn Easy Apply — fully tool-calling agent (langchain create_agent).

A single LLM drives the whole flow via tools: open the search, list jobs, read each
job, and complete the Easy Apply modal. Goal: submit APPLY_TARGET new, confirmed
applications (already-applied / failed don't count).

Each confirmed application is saved as one markdown file:
    data/<keyword>-<location>/<job_id>-<company>.md

Config lives in `instructions.md` (search filter + answering policy). Async Playwright +
async tools + agent.ainvoke() (the agent graph runs on an asyncio event loop).
"""

import asyncio
import datetime
import os
import re
from urllib.parse import urlencode

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from playwright.async_api import async_playwright

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

INSTRUCTION_FILE = os.path.join(BASE_DIR, "instructions.md")  # search filter + answering policy
RESUME_PDF = os.path.join(BASE_DIR, "resume.pdf")             # your resume (not tracked)
USER_DATA_DIR = os.path.join(BASE_DIR, ".linkedin-profile")   # persistent LinkedIn login
DATA_DIR = os.path.join(BASE_DIR, "data")                     # output
TOKEN_LOG = os.path.join(BASE_DIR, "tokens.txt")

APPLY_TARGET = 10
DEEPSEEK_MODEL = "deepseek-v4-flash"
RECURSION_LIMIT = 500

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}


# --- Token logging -----------------------------------------------------------

class TokenLogger(BaseCallbackHandler):
    """Prepend 'in N out N' to TOKEN_LOG after every LLM call (newest on top)."""

    def __init__(self, path):
        self.path = path
        self.total_in = 0
        self.total_out = 0

    def on_llm_end(self, response, **kwargs):
        tin = tout = 0
        try:
            usage = getattr(response.generations[0][0].message, "usage_metadata", None) or {}
            tin = usage.get("input_tokens", 0)
            tout = usage.get("output_tokens", 0)
        except Exception:
            pass
        self.total_in += tin
        self.total_out += tout
        old = ""
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                old = f.read()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"in {tin} out {tout}\n" + old)


# --- Plain helpers (no page / no AI) -----------------------------------------

def slugify(name: str) -> str:
    name = name.strip().lower().replace(" ", "-")
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    name = re.sub(r"-{2,}", "-", name)
    return name.strip("-. ") or "unknown"


def location_slug(location: str) -> str:
    """'San Francisco, CA' -> 'san-francisco-california' (expand the state abbrev)."""
    parts = [p.strip() for p in location.split(",")]
    if len(parts) >= 2:
        city, st = parts[0], parts[1]
        return slugify(f"{city} {US_STATES.get(st.lower(), st)}")
    return slugify(location)


def parse_duration_seconds(text: str) -> int:
    m = re.search(r"(\d+)\s*([a-z]*)", text.lower())
    if not m:
        return 0
    n, unit = int(m.group(1)), (m.group(2) or "h")
    if unit.startswith("mo"):
        return n * 2592000  # month(s)
    if unit.startswith("w"):
        return n * 604800   # week(s)
    if unit.startswith("d"):
        return n * 86400    # day(s)
    return n * 3600         # hour(s), default


def extract_section(text, header):
    out, cap = [], False
    for line in text.splitlines():
        if line.strip().startswith("## "):
            if cap:
                break
            cap = line.strip() == header
            continue
        if cap:
            out.append(line)
    return "\n".join(out).strip()


def read_instruction(path: str) -> dict:
    """Parse the '## Search filter' section into LinkedIn URL params."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing instructions file: {path} (copy instructions.example.md)")
    text = open(path, encoding="utf-8").read()
    raw = {}
    for line in extract_section(text, "## Search filter").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        raw[k.strip().lower()] = v.strip()
    params = {}
    if raw.get("keywords"):
        params["keywords"] = raw["keywords"]
    if raw.get("location"):
        params["location"] = raw["location"]
    sec = parse_duration_seconds(raw.get("date_posted", ""))
    if sec:
        params["f_TPR"] = f"r{sec}"
    if raw.get("easy_apply", "").lower() in ("true", "yes", "1", "on"):
        params["f_AL"] = "true"
    if raw.get("remote", "").lower() in ("true", "yes", "1", "on"):
        params["f_WT"] = "2"  # LinkedIn work-type: 2 = Remote (1 on-site, 3 hybrid)
    return params


def distill_instructions() -> str:
    """The answering policy + screening answers the agent uses to fill forms."""
    if not os.path.isfile(INSTRUCTION_FILE):
        return ""
    text = open(INSTRUCTION_FILE, encoding="utf-8").read()
    return (
        f"Answering policy:\n{extract_section(text, '## Answering policy')}\n\n"
        f"Screening answers:\n{extract_section(text, '## Screening question answers')}"
    ).strip()


# --- Async page helpers ------------------------------------------------------

async def first_text(page, selectors):
    for sel in selectors:
        el = await page.query_selector(sel)
        if el:
            t = (await el.inner_text()).strip()
            if t:
                return t
    return ""


async def applied_confirmed(page):
    try:
        return await page.evaluate(
            """() => {
                const d = document.querySelector('[role=dialog]');
                const t = ((d ? d.innerText : document.body.innerText) || '').toLowerCase();
                return t.includes('was sent') || t.includes('application sent');
            }"""
        )
    except Exception:
        return False


async def close_any_modal(page):
    for sel in ("button[aria-label*='Dismiss' i]", "button[aria-label*='Discard' i]"):
        b = await page.query_selector(sel)
        if b:
            try:
                await b.click()
                await page.wait_for_timeout(600)
            except Exception:
                pass


# Tags visible modal fields with data-fill-idx. Radios/checkboxes are kept even when the
# native input is hidden, and are labeled with their question (fieldset legend).
_FIELDS_JS = """(maxN) => {
    const modal = document.querySelector('.jobs-easy-apply-modal')
        || document.querySelector('[role=dialog]') || document;
    const els = Array.from(modal.querySelectorAll('input, select, textarea'));
    const out = []; let idx = 0;
    for (const e of els) {
        const tag = e.tagName.toLowerCase();
        const type = (e.getAttribute('type')||'').toLowerCase();
        if (tag==='input' && type==='hidden') continue;
        const isChoice = (type==='radio' || type==='checkbox');
        const st = getComputedStyle(e); const r = e.getBoundingClientRect();
        const visible = (r.width>0 && r.height>0 && st.visibility!=='hidden' && st.display!=='none');
        if (!visible && !isChoice) continue;
        let kind = tag==='select'?'select':(tag==='textarea'?'textarea':(type||'text'));
        let own='';
        if (e.id){const l=document.querySelector('label[for="'+e.id+'"]'); if(l)own=l.innerText;}
        if (!own) own = e.getAttribute('aria-label')||e.getAttribute('placeholder')||e.value||'';
        own = own.replace(/\\s+/g,' ').trim();
        let label = own;
        if (isChoice) {
            let q = '';
            const fs = e.closest('fieldset');
            if (fs && fs.querySelector('legend')) q = fs.querySelector('legend').innerText;
            if (!q) { const g = e.closest('[role=group],[role=radiogroup]'); if (g) q = g.getAttribute('aria-label')||''; }
            q = (q||'').replace(/\\s+/g,' ').trim();
            label = (q ? q + ' = ' : '') + (own || 'option');
        }
        label = label.slice(0,140);
        let options=[];
        if (tag==='select') options=Array.from(e.options).map(o=>o.text.trim()).filter(Boolean).slice(0,20);
        e.setAttribute('data-fill-idx', idx);
        out.push({i:idx,kind,label,value:(e.value||'').slice(0,40),options});
        idx++; if(idx>=maxN)break;
    }
    return out;
}"""

# --- Run-time config + shared state ------------------------------------------

SEARCH_PARAMS = read_instruction(INSTRUCTION_FILE)
JOBS_URL = "https://www.linkedin.com/jobs/search/?" + urlencode(SEARCH_PARAMS)
KEYWORD = SEARCH_PARAMS.get("keywords", "jobs")
LOCATION = SEARCH_PARAMS.get("location", "")
OUT_DIR = os.path.join(DATA_DIR, f"{slugify(KEYWORD)}-{location_slug(LOCATION)}")
DISTILLED = distill_instructions()

S = {"page": None, "context": None, "applied": 0, "target": APPLY_TARGET, "cur": {}}


def _page():
    return S["page"]


def write_applied_md(cur) -> str:
    """One markdown confirmation per successful application under data/<keyword>-<location>/."""
    os.makedirs(OUT_DIR, exist_ok=True)
    name = f"{cur.get('job_id', 'unknown')}-{slugify(cur.get('company', 'unknown'))}.md"
    path = os.path.join(OUT_DIR, name)
    when = datetime.date.today().isoformat()
    parts = [f"Applied: {when}", "", f"# {(cur.get('title') or '').strip()}",
             f"Company: {(cur.get('company') or '').strip()}", ""]
    cdesc = (cur.get("company_description") or "").strip()
    if cdesc:
        parts += [cdesc, ""]
    parts += ["---", "", (cur.get("description") or "").strip(), ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


# --- Async tools -------------------------------------------------------------

@tool
async def go_to_search() -> str:
    """Open the LinkedIn jobs search (filtered per instructions.md). Call once at the start."""
    page = _page()
    await page.goto(JOBS_URL, wait_until="domcontentloaded", timeout=60000)
    try:
        await page.wait_for_selector("li[data-occludable-job-id]", timeout=20000)
    except Exception:
        return "search opened but no jobs visible"
    return f"search opened: {page.url}"


@tool
async def list_jobs() -> str:
    """List visible jobs as 'job_id | APPLIED|new | title'. ONLY open the 'new' ones —
    'APPLIED' jobs are already done and must be skipped."""
    page = _page()
    for _ in range(5):
        try:
            await page.evaluate("() => { const i=document.querySelectorAll('li[data-occludable-job-id]'); if(i.length) i[i.length-1].scrollIntoView(); }")
        except Exception:
            pass
        await page.wait_for_timeout(800)
    items = await page.evaluate(
        """() => {
            const out = [];
            document.querySelectorAll('li[data-occludable-job-id]').forEach(li => {
                const id = li.getAttribute('data-occludable-job-id');
                if (!id) return;
                const applied = /\\bApplied\\b/.test(li.innerText || '');
                const a = li.querySelector('a.job-card-container__link, a[class*="job-card-list__title"], a');
                const title = (a ? a.innerText : (li.innerText || '')).replace(/\\s+/g, ' ').trim().slice(0, 60);
                out.push({ id, applied, title });
            });
            return out;
        }"""
    )
    lines = [f"{it['id']} | {'APPLIED' if it['applied'] else 'new'} | {it['title']}" for it in items[:25]]
    return "JOBS (open ONLY the 'new' ones):\n" + "\n".join(lines) if lines else "no jobs visible"


@tool
async def next_page(start: int) -> str:
    """Load the next page of results. start = 25, 50, 75, ... 'no more results' when exhausted."""
    page = _page()
    await page.goto(JOBS_URL + f"&start={start}", wait_until="domcontentloaded", timeout=60000)
    try:
        await page.wait_for_selector("li[data-occludable-job-id]", timeout=15000)
    except Exception:
        return "no more results"
    return f"page start={start} loaded"


@tool
async def open_job(job_id: str) -> str:
    """Open a job's detail pane by its id (resets the current-job context)."""
    page = _page()
    await page.goto(JOBS_URL + f"&currentJobId={job_id}", wait_until="domcontentloaded", timeout=60000)
    try:
        await page.wait_for_selector(".jobs-description__content, #job-details", timeout=20000)
    except Exception:
        return "opened but no description pane"
    await page.wait_for_timeout(1200)
    for sel in (".jobs-description__footer-button", "button.show-more-less-html__button"):
        b = await page.query_selector(sel)
        if b:
            try:
                await b.click()
                await page.wait_for_timeout(400)
            except Exception:
                pass
            break
    S["cur"] = {"job_id": str(job_id)}
    return f"opened job {job_id}"


@tool
async def read_job() -> str:
    """Read the open job (title, company, company blurb, description). Call before applying."""
    page = _page()
    title = await first_text(page, [".job-details-jobs-unified-top-card__job-title", ".jobs-unified-top-card__job-title", "h1"])
    company = await first_text(page, [".job-details-jobs-unified-top-card__company-name", ".jobs-unified-top-card__company-name"])
    company_desc = await first_text(page, [".jobs-company__company-description", ".jobs-company__box", ".jobs-company"])
    desc = await first_text(page, ["#job-details", ".jobs-description__content", ".jobs-box__html-content"])
    S["cur"].update({"title": title, "company": company, "company_description": company_desc, "description": desc})
    return f"TITLE: {title}\nCOMPANY: {company}\nDESC: {desc[:1000]}"


@tool
async def open_easy_apply() -> str:
    """Open the Easy Apply modal for the current job. Reports if it's external (skip) or no button."""
    page = _page()
    btn = await page.query_selector("button.jobs-apply-button")
    if not btn:
        return "no Easy Apply button (maybe already applied or external) — skip this job"
    aria = ((await btn.get_attribute("aria-label")) or "").lower()
    if "company website" in aria:
        return "EXTERNAL application (company website) — skip this job"
    try:
        await btn.click()
        await page.wait_for_timeout(2000)
    except Exception as e:
        return f"could not open Easy Apply: {e}"
    return "Easy Apply modal opened — call read_form next"


@tool
async def read_form() -> str:
    """List the open Easy Apply modal's fields, step buttons, and all visible buttons."""
    page = _page()
    fields = await page.evaluate(_FIELDS_JS, 40)
    lines = [
        f"{f['i']} | {f['kind']} | {f['label']} | value='{f['value']}'"
        + (f" | options={f['options']}" if f["options"] else "")
        for f in fields
    ]
    btns = []
    for name, sel in [
        ("submit", "button[aria-label*='Submit application' i]"),
        ("review", "button[aria-label*='Review' i]"),
        ("next", "button[aria-label*='Continue to next step' i]"),
    ]:
        if await page.query_selector(sel):
            btns.append(name)
    all_btns = await page.evaluate(
        """() => {
            const modal = document.querySelector('.jobs-easy-apply-modal')
                || document.querySelector('[role=dialog]') || document.body;
            const out = [];
            modal.querySelectorAll('button, [role=button]').forEach(b => {
                const st = getComputedStyle(b); const r = b.getBoundingClientRect();
                if (!(r.width>0&&r.height>0)||st.visibility==='hidden'||st.display==='none') return;
                const t = (b.innerText || b.getAttribute('aria-label') || '').replace(/\\s+/g,' ').trim();
                if (t && t.length <= 40) out.push(t);
            });
            return [...new Set(out)].slice(0, 15);
        }"""
    )
    return (
        "FIELDS:\n" + ("\n".join(lines) if lines else "(none)")
        + "\nSTEP_BUTTONS: " + (", ".join(btns) or "none")
        + "\nBUTTONS: " + (", ".join(all_btns) or "none")
    )


@tool
async def fill_field(index: int, value: str) -> str:
    """Type text into the text/number/textarea field with this index."""
    try:
        await _page().fill(f'[data-fill-idx="{index}"]', str(value)[:200], timeout=6000)
        return f"filled {index}"
    except Exception as e:
        return f"fill failed: {e}"


@tool
async def choose_option(index: int) -> str:
    """Select the radio/checkbox with this index (clicks its label — LinkedIn hides the input)."""
    page = _page()
    sel = f'[data-fill-idx="{index}"]'
    try:
        clicked = await page.evaluate(
            """(sel) => {
                const el = document.querySelector(sel);
                if (!el) return false;
                let lab = el.id ? document.querySelector('label[for="'+el.id+'"]') : null;
                if (!lab) lab = el.closest('label');
                if (lab) { lab.click(); return true; }
                el.click(); return true;
            }""",
            sel,
        )
        await page.wait_for_timeout(300)
        if not clicked:
            await page.check(sel, timeout=4000, force=True)
        return f"checked {index}"
    except Exception:
        try:
            await page.check(sel, timeout=4000, force=True)
            return f"checked {index} (force)"
        except Exception as e2:
            return f"check failed: {e2}"


@tool
async def select_dropdown(index: int, value: str) -> str:
    """Pick the option (by exact visible text) in the dropdown <select> with this index."""
    try:
        await _page().select_option(f'[data-fill-idx="{index}"]', label=str(value), timeout=6000)
        return f"selected '{value}' in {index}"
    except Exception as e:
        return f"select failed: {e}"


@tool
async def upload_resume() -> str:
    """Attach the resume PDF to a file input in the modal, if present."""
    page = _page()
    n = 0
    for h in await page.query_selector_all(".jobs-easy-apply-modal input[type=file], [role=dialog] input[type=file]"):
        try:
            await h.set_input_files(RESUME_PDF)
            n += 1
        except Exception:
            pass
    await page.wait_for_timeout(1500)
    return f"uploaded to {n} input(s)"


@tool
async def click_next() -> str:
    """Advance the Easy Apply form: click 'Continue to next step' or 'Review your application'."""
    page = _page()
    for sel in ("button[aria-label*='Review' i]",
                "button[aria-label*='Continue to next step' i]",
                "button[aria-label*='next' i]"):
        b = await page.query_selector(sel)
        if b:
            try:
                await b.click()
                await page.wait_for_timeout(1500)
                return "advanced to next step"
            except Exception as e:
                return f"advance failed: {e}"
    return "no next/review button — maybe the Submit button is available now"


@tool
async def submit_application() -> str:
    """Click 'Submit application'. Only a CONFIRMED submission counts and saves the .md file."""
    page = _page()
    b = await page.query_selector("button[aria-label*='Submit application' i]")
    if not b:
        return "no Submit button yet — fill required fields and click_next until it appears"
    try:
        await b.click()
        await page.wait_for_timeout(2500)
    except Exception as e:
        return f"submit click failed: {e}"
    confirmed = await applied_confirmed(page)
    await close_any_modal(page)
    if not confirmed:
        return "submitted but NOT confirmed — a required field may be unanswered. Call read_form and fix it."
    if not S["cur"].get("description"):
        return "CONFIRMED but read_job was not called first, so nothing was saved. Call read_job before applying next time."
    S["applied"] += 1
    base = os.path.basename(write_applied_md(S["cur"]))
    if S["applied"] >= S["target"]:
        return f"APPLIED + CONFIRMED ({S['applied']}/{S['target']}), saved {base}. TARGET REACHED — reply DONE and stop."
    return f"APPLIED + CONFIRMED ({S['applied']}/{S['target']}), saved {base}. Move on to the next job."


@tool
async def click_button(text: str) -> str:
    """Click a visible modal/dialog button whose text contains `text` (e.g. 'Continue
    applying', 'Got it'). Use to get PAST reminders/interstitials — never to back out."""
    page = _page()
    hit = await page.evaluate(
        """(text) => {
            const modal = document.querySelector('.jobs-easy-apply-modal')
                || document.querySelector('[role=dialog]') || document.body;
            const t = text.toLowerCase();
            for (const b of modal.querySelectorAll('button, [role=button]')) {
                const st = getComputedStyle(b); const r = b.getBoundingClientRect();
                if (!(r.width>0&&r.height>0)||st.visibility==='hidden'||st.display==='none') continue;
                const bt = (b.innerText || b.getAttribute('aria-label') || '').replace(/\\s+/g,' ').trim();
                if (bt.toLowerCase().includes(t)) { b.click(); return bt; }
            }
            return '';
        }""",
        text,
    )
    await page.wait_for_timeout(1500)
    return f"clicked '{hit}'" if hit else f"no button matching '{text}'"


@tool
async def skip_job(reason: str) -> str:
    """Skip the current job (external or can't complete) and move on. Nothing is saved."""
    await close_any_modal(_page())
    return f"skipped: {reason}"


TOOLS = [
    go_to_search, list_jobs, next_page, open_job, read_job,
    open_easy_apply, read_form, fill_field, choose_option, select_dropdown,
    upload_resume, click_next, click_button, submit_application, skip_job,
]

SYSTEM = (
    "You are an autonomous LinkedIn Easy Apply agent driving a real browser via tools. "
    f"GOAL: submit {APPLY_TARGET} NEW, CONFIRMED job applications, then stop.\n\n"
    "list_jobs marks each job 'APPLIED' or 'new'. ONLY work on 'new' jobs — NEVER open a "
    "job marked 'APPLIED'. Use next_page / list_jobs to get more 'new' jobs.\n\n"
    "For each 'new' job: open_job -> read_job (ALWAYS, before applying) -> open_easy_apply "
    "(if external, skip_job) -> loop read_form -> fill_field / choose_option / "
    "select_dropdown / upload_resume as needed -> click_next, until a Submit button exists, "
    "then submit_application. Use choose_option for radio/checkbox, select_dropdown for "
    "real dropdowns.\n\n"
    "YOU MUST ACTUALLY APPLY to every 'new' job — carry it through submit_application until "
    "CONFIRMED. Do NOT skip_job unless open_easy_apply reported EXTERNAL. If read_form shows "
    "a reminder/interstitial (e.g. 'Continue applying'), use click_button to proceed. After "
    "a CONFIRMED submission, move on. Only a CONFIRMED submission counts. When you reach the "
    "target, reply DONE and stop.\n\n"
    "Answer screening questions using the answering policy and screening answers below — "
    "never leave a required field blank.\n\n"
    f"CANDIDATE DATA AND POLICY:\n{DISTILLED}"
)

GOAL = (
    f"Apply to {APPLY_TARGET} '{KEYWORD}' Easy Apply jobs"
    + (f" in {LOCATION}" if LOCATION else "")
    + ". Start by calling go_to_search, then list_jobs."
)

token_logger = TokenLogger(TOKEN_LOG)
llm = ChatDeepSeek(model=DEEPSEEK_MODEL, callbacks=[token_logger])
agent = create_agent(llm, TOOLS, system_prompt=SYSTEM)


async def main():
    print(f"Search params: {SEARCH_PARAMS}", flush=True)
    print(f"Output dir: {OUT_DIR}\nTarget: {APPLY_TARGET} applications\n", flush=True)
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(USER_DATA_DIR, headless=False)
        S["context"] = ctx
        S["page"] = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await S["page"].goto("https://www.linkedin.com", wait_until="domcontentloaded", timeout=60000)
        print("Opened:", await S["page"].title(), flush=True)
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": GOAL}]},
                config={"recursion_limit": RECURSION_LIMIT},
            )
            msgs = result.get("messages", []) if isinstance(result, dict) else []
            if msgs:
                print("\nAGENT FINAL:", str(getattr(msgs[-1], "content", ""))[:600], flush=True)
        except Exception as e:
            print("\nagent error:", repr(e), flush=True)
        print(
            f"\nApplied {S['applied']}/{S['target']}. "
            f"Tokens in {token_logger.total_in} out {token_logger.total_out}",
            flush=True,
        )
        await ctx.close()


if __name__ == "__main__":
    asyncio.run(main())
