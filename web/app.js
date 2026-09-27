// QA Agent chat: a chat-style front end for the FastAPI backend (same origin).
// A message with no run open starts a QA run; inside a run, messages are questions
// answered from the run's data (POST /runs/{id}/chat).
(function () {
  "use strict";

  const POLL_MS = 2000;
  const HISTORY_POLL_MS = 8000;
  const NODE_LABELS = {
    requirement_analyzer: "Analyse requirement",
    test_case_generator: "Generate test cases",
    test_planner: "Plan test execution",
    test_executor: "Execute tests",
    result_analyzer: "Analyse results",
    failure_analyzer: "Analyse failures",
    root_cause_analyzer: "Find root cause",
    bug_report_generator: "Write bug reports",
    confidence_checker: "Check confidence",
    human_review: "Human review",
    final_report_generator: "Final QA report",
  };
  const STATUS_LABELS = {
    running: "Running",
    awaiting_execution: "Waiting to execute",
    awaiting_review: "Needs your review",
    completed: "Completed",
    error: "Error",
  };
  const EXAMPLES = [
    ["Login", "A registered user can log in with email and password on the login page and through POST /api/v2/login. Valid credentials (registered.user@example.com / N3w-Passw0rd!) open the account dashboard; invalid credentials return 401 and show \"Invalid email or password.\""],
    ["Password reset", "A user who forgot their password can request a reset link on the Forgot password page. The page shows the same message whether or not the email is registered, and the reset link can be used only once."],
    ["Sign up", "A visitor can create an account on the sign-up page with an email and a password of at least 12 characters including a letter, a digit and a symbol. Registering an email that already exists shows an error."],
  ];

  const $ = (id) => document.getElementById(id);
  const thread = $("thread");
  const input = $("input");

  const state = {
    runs: [],
    activeId: null,
    data: null, // artifacts of the active run
    pollTimer: null,
    lastKey: null,
    sending: false,
    // After execute/review the backend answers 202 before the status flips to "running";
    // keep polling until it leaves the status the action was taken in.
    awaitChange: null, // { from, until }
    mode: "requirement", // welcome screen: "requirement" | "ticket"
  };

  const SAMPLE_TICKET = {
    ticket_id: "SD-1042",
    title: "Account is not locked after repeated failed logins",
    description: "A customer reports that they can keep guessing passwords on the login page without ever being locked out.",
    steps_to_reproduce: [
      "Open http://127.0.0.1:8765/login.html",
      "Enter registered.user@example.com with the wrong password Wrong-Passw0rd!1 and click Log in",
      "Repeat the failed login until it has failed 3 times",
      "Click Log in a 4th time with the same wrong password",
    ].join("\n"),
    expected_result: "After the 4th attempt the page says the account is locked",
    actual_result: "The page still says \"Invalid email or password.\"",
    environment: "http://127.0.0.1:8765/login.html",
  };
  const VERDICT = {
    confirmed: ["🐞", "verdict-confirmed"],
    not_reproduced: ["✅", "verdict-ok"],
    inconclusive: ["⚠️", "verdict-warn"],
    pending: ["⏳", "verdict-pending"],
  };
  const isTicketRun = (run) => Boolean(run && run.metadata && run.metadata.mode === "bug_verification");
  const ticketOf = (run) => (run && run.metadata && run.metadata.bug_ticket) || null;
  const runTitle = (run) => {
    const t = ticketOf(run);
    return t ? `🐞 ${t.ticket_id ? `${t.ticket_id} · ` : ""}${t.title}` : run.requirement;
  };

  // --- helpers -------------------------------------------------------------------

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  // Tiny, safe Markdown: escape first, then **bold**, `code`, "- " lists, paragraphs.
  function md(text) {
    const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
    const out = [];
    let list = null;
    for (const raw of String(text || "").split("\n")) {
      const line = raw.trimEnd();
      const item = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
      if (item) {
        list = list || [];
        list.push(`<li>${inline(item[1])}</li>`);
        continue;
      }
      if (list) { out.push(`<ul>${list.join("")}</ul>`); list = null; }
      if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    if (list) out.push(`<ul>${list.join("")}</ul>`);
    return out.join("");
  }

  async function api(method, path, body) {
    const res = await fetch(path, {
      method,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = text; }
    if (!res.ok) {
      const detail = data && data.detail;
      const message = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : `HTTP ${res.status}`;
      throw new Error(message);
    }
    return data;
  }

  function timeAgo(iso) {
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return new Date(iso).toLocaleDateString();
  }

  const chatKey = (id) => `qa-chat-${id}`;
  function loadChat(id) {
    try { return JSON.parse(localStorage.getItem(chatKey(id)) || "[]"); } catch { return []; }
  }
  function saveChat(id, messages) {
    try { localStorage.setItem(chatKey(id), JSON.stringify(messages.slice(-60))); } catch { /* storage unavailable */ }
  }

  const nearBottom = () => thread.scrollHeight - thread.scrollTop - thread.clientHeight < 120;
  const scrollDown = () => { thread.scrollTop = thread.scrollHeight; };

  // --- system status & history -------------------------------------------------------

  async function loadSystemStatus() {
    const box = $("system-status");
    try {
      const s = await api("GET", "/system/status");
      const warnings = (s.warnings || []).map((w) => `<span class="warn">⚠ ${esc(w)}</span>`).join("");
      box.innerHTML = `Model: <b>${esc(s.llm_model)}</b>${s.llm_configured ? "" : " (no API key)"}${warnings}`;
    } catch {
      box.innerHTML = '<span class="warn">⚠ Backend not reachable</span>';
    }
  }

  async function loadHistory() {
    try {
      state.runs = (await api("GET", "/runs")).sort((a, b) => b.created_at.localeCompare(a.created_at));
    } catch { return; }
    renderHistory();
  }

  function renderHistory() {
    const nav = $("history");
    if (!state.runs.length) {
      nav.innerHTML = '<div class="history-empty">No runs yet. Describe a requirement to start one.</div>';
      return;
    }
    nav.innerHTML = state.runs.map((r) => `
      <button type="button" class="history-item${r.run_id === state.activeId ? " active" : ""}" data-run="${esc(r.run_id)}">
        <span class="h-text">${esc(runTitle(r))}</span>
        <span class="h-meta"><span class="dot ${esc(r.status)}"></span>${esc(STATUS_LABELS[r.status] || r.status)} · ${esc(timeAgo(r.created_at))}</span>
      </button>`).join("");
  }

  // --- navigation ------------------------------------------------------------------

  function openNew() {
    stopPolling();
    state.activeId = null;
    state.data = null;
    state.lastKey = null;
    if (location.hash) history.replaceState(null, "", location.pathname);
    $("chat-title").textContent = "New QA run";
    $("chat-subtitle").textContent = "Describe a requirement and the agent will test it";
    $("run-badge").hidden = true;
    input.placeholder = "Describe a requirement to test…";
    renderWelcome();
    renderHistory();
    closeSidebar();
    input.focus();
  }

  function openRun(runId) {
    stopPolling();
    state.awaitChange = null;
    state.activeId = runId;
    state.data = null;
    state.lastKey = null;
    if (location.hash !== `#run=${runId}`) history.replaceState(null, "", `#run=${runId}`);
    $("pause-option").hidden = true;
    $("composer").hidden = false;
    input.placeholder = "Ask about this run, e.g. \"Why did TC-005 fail?\"";
    thread.innerHTML = '<div class="msg agent"><div class="avatar">QA</div><div class="bubble"><div class="agent-text typing"><span></span><span></span><span></span></div></div></div>';
    renderHistory();
    closeSidebar();
    poll();
  }

  function stopPolling() {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  // --- polling the active run ------------------------------------------------------

  async function fetchArtifacts(runId) {
    const get = (p) => api("GET", `/runs/${runId}${p}`).catch(() => null);
    const [run, analysis, cases, results, failures, rca, bugs, review, report] = await Promise.all([
      get(""), get("/requirement-analysis"), get("/test-cases"), get("/results"), get("/failures"),
      get("/root-cause"), get("/bugs"), get("/review"), get("/report"),
    ]);
    return { run, analysis, cases: cases || [], results: results || [], failures: failures || [], rca, bugs: bugs || [], review, report };
  }

  async function poll() {
    const runId = state.activeId;
    if (!runId) return;
    let data;
    try {
      data = await fetchArtifacts(runId);
    } catch {
      data = null;
    }
    if (runId !== state.activeId) return; // user switched runs meanwhile
    if (!data || !data.run) {
      thread.innerHTML = `<div class="card errors">This run could not be loaded. The backend keeps runs in memory, so runs from before a backend restart are gone.</div>`;
      return;
    }
    state.data = data;
    const key = `${data.run.status}|${data.run.updated_at}|${(data.run.current_nodes || []).join()}`;
    if (key !== state.lastKey) {
      state.lastKey = key;
      renderRun();
      const summary = state.runs.find((r) => r.run_id === runId);
      if (!summary || summary.status !== data.run.status) loadHistory();
    } else {
      updateRunningTimers();
    }
    const wait = state.awaitChange;
    if (wait && (data.run.status !== wait.from || Date.now() > wait.until)) state.awaitChange = null;
    if (data.run.status === "running" || state.awaitChange) state.pollTimer = setTimeout(poll, POLL_MS);
  }

  // --- rendering -------------------------------------------------------------------

  function renderWelcome() {
    const ticket = state.mode === "ticket";
    $("composer").hidden = ticket;
    $("pause-option").hidden = ticket;
    const tabs = `<div class="mode-tabs" role="tablist">
        <button type="button" role="tab" class="mode-tab${ticket ? "" : " active"}" data-mode="requirement">📄 Test a requirement</button>
        <button type="button" role="tab" class="mode-tab${ticket ? " active" : ""}" data-mode="ticket">🐞 Verify a bug ticket</button>
      </div>`;
    if (!ticket) {
      thread.innerHTML = `
        <div class="welcome">
          <div class="welcome-mark">QA</div>
          <h1>What should I test?</h1>
          ${tabs}
          <p>Describe a feature or requirement. I will analyse it, write test cases, run API and UI tests against the configured app, find root causes and report bugs. Then ask me anything about the results.</p>
          <div class="examples">
            ${EXAMPLES.map(([title, text], i) => `<button type="button" class="example" data-example="${i}"><b>${esc(title)}</b>${esc(text.slice(0, 110))}…</button>`).join("")}
          </div>
        </div>`;
      return;
    }
    thread.innerHTML = `
      <div class="welcome">
        <div class="welcome-mark">QA</div>
        <h1>Is this bug real?</h1>
        ${tabs}
        <p>Paste a ticket from the service desk. I will follow its steps to reproduce against the test environment and tell you whether the bug is <b>confirmed</b>, <b>not reproducible</b> or <b>inconclusive</b>, with the evidence.</p>
      </div>
      <form id="ticket-form" class="card ticket-form" novalidate>
        <div class="grid-2">
          <label>Ticket number<input name="ticket_id" placeholder="SD-1042"></label>
          <label>Title *<input name="title" placeholder="Account is not locked after failed logins"></label>
        </div>
        <label>Description<textarea name="description" rows="2" placeholder="What the customer reported, in their words"></textarea></label>
        <label>Steps to reproduce (one per line)<textarea name="steps_to_reproduce" rows="4" placeholder="Open the login page&#10;Enter a wrong password 3 times&#10;Try to log in again"></textarea></label>
        <div class="grid-2">
          <label>Expected result *<textarea name="expected_result" rows="2" placeholder="The account is locked"></textarea></label>
          <label>Actual result (reported) *<textarea name="actual_result" rows="2" placeholder="The login still says invalid password"></textarea></label>
        </div>
        <label>Environment / URL<input name="environment" placeholder="http://127.0.0.1:8765/login.html"></label>
        <p id="ticket-error" class="form-error" hidden></p>
        <div class="actions">
          <button type="submit" class="btn">🔎 Verify bug</button>
          <button type="button" class="btn ghost" data-action="sample-ticket">Fill a sample ticket</button>
          <label class="inline-check"><input type="checkbox" name="pause"> Review tests before they run</label>
        </div>
      </form>`;
  }

  const agentMsg = (html) => `<div class="msg agent"><div class="avatar">QA</div><div class="bubble">${html}</div></div>`;
  const userMsg = (text) => `<div class="msg user"><div class="bubble">${esc(text)}</div></div>`;

  function renderRun() {
    const d = state.data;
    const run = d.run;
    const stick = nearBottom() || !thread.querySelector(".card");
    $("chat-title").textContent = runTitle(run);
    $("chat-subtitle").textContent = `Run ${run.run_id.slice(0, 8)} · ${run.project} · started ${timeAgo(run.created_at)}`;
    const badge = $("run-badge");
    badge.hidden = false;
    badge.className = `badge ${run.status}`;
    badge.textContent = STATUS_LABELS[run.status] || run.status;

    const parts = [isTicketRun(run) ? ticketMsg(ticketOf(run)) : userMsg(run.requirement)];
    const cards = [];
    if (run.verdict) cards.push(verdictCard(run.verdict, d.cases));
    cards.push(progressCard(run));
    if (d.analysis) cards.push(analysisCard(d.analysis));
    if (d.cases.length) cards.push(casesCard(d.cases));
    if (run.status === "awaiting_execution") cards.push(executeCard(d.cases.length));
    if (d.results.length) cards.push(resultsCard(d.results, d.cases));
    if (d.rca && d.rca.analysis && d.rca.analysis.findings && d.rca.analysis.findings.length) cards.push(rootCauseCard(d.rca.analysis));
    if (d.bugs.length) cards.push(bugsCard(d.bugs));
    if (d.review && d.review.pending && run.status === "awaiting_review") cards.push(reviewCard(d.review.pending));
    else if (d.review && d.review.latest && d.review.latest.human_decision) cards.push(decisionNote(d.review.latest));
    if (d.report) cards.push(reportCard(d.report, run.run_id));
    if (run.errors && run.errors.length) cards.push(errorsCard(run.errors));
    if (run.error) cards.push(`<div class="card errors"><h3>⚠ The run stopped</h3>${esc(run.error)}</div>`);
    parts.push(agentMsg(cards.join("")));

    const hint = nextStepHint(run);
    if (hint) parts.push(agentMsg(`<div class="agent-text">${md(hint)}</div>`));
    for (const m of loadChat(run.run_id)) parts.push(m.role === "user" ? userMsg(m.content) : agentMsg(`<div class="agent-text">${md(m.content)}</div>`));
    if (state.sending) parts.push(agentMsg('<div class="agent-text typing"><span></span><span></span><span></span></div>'));

    thread.innerHTML = parts.join("");
    if (stick) scrollDown();
  }

  function nextStepHint(run) {
    switch (run.status) {
      case "running": return "Working on it… you can already ask questions about what is done so far.";
      case "awaiting_execution": return "The test cases are ready. Review them, then press **Run tests**.";
      case "awaiting_review": return "The run needs your decision before the final report. Check the review card above.";
      case "completed": return isTicketRun(run)
        ? "Verification complete. Ask me anything, for example \"Which step shows the bug?\" or \"Write a reply for the service desk ticket\"."
        : "The run is complete. Ask me anything about it, for example \"Why did a test fail?\" or \"Summarise the bugs\".";
      default: return "";
    }
  }

  function ticketMsg(t) {
    const steps = (t.steps_to_reproduce || []).map((s) => `<li>${esc(s)}</li>`).join("");
    return `<div class="msg user"><div class="bubble ticket-bubble">
        <div class="ticket-head">🐞 Bug ticket${t.ticket_id ? ` <b>${esc(t.ticket_id)}</b>` : ""}</div>
        <div class="ticket-title">${esc(t.title)}</div>
        ${t.description ? `<div class="ticket-desc">${esc(t.description)}</div>` : ""}
        ${steps ? `<div class="ticket-label">Steps to reproduce</div><ol>${steps}</ol>` : ""}
        <div class="ticket-label">Expected</div><div>${esc(t.expected_result)}</div>
        <div class="ticket-label">Actual (reported)</div><div>${esc(t.actual_result)}</div>
        ${t.environment ? `<div class="ticket-label">Environment</div><div>${esc(t.environment)}</div>` : ""}
      </div></div>`;
  }

  function verdictCard(v, cases) {
    const [icon, cls] = VERDICT[v.status] || ["ℹ️", "verdict-pending"];
    const titles = Object.fromEntries((cases || []).map((c) => [c.test_case_id, c.title]));
    const rows = [
      ["Failed: bug reproduced", v.failed_tests, "failed"],
      ["Passed: behaved as expected", v.passed_tests, "passed"],
      ["Could not run", v.inconclusive_tests, "skipped"],
    ].filter(([, ids]) => ids && ids.length)
      .map(([label, ids, chip]) => `<div class="verdict-row"><span class="chip ${chip}">${label}</span> ${ids.map((id) => `<b>${esc(id)}</b>${titles[id] ? ` ${esc(titles[id])}` : ""}`).join(" · ")}</div>`)
      .join("");
    const mark = v.status === "pending" ? '<span class="spinner"></span>' : icon;
    const repro = v.reproduction_test
      ? `<div class="verdict-row muted">Reproduction test (follows the ticket's steps): <b>${esc(v.reproduction_test)}</b> ${esc(titles[v.reproduction_test] || "")}</div>`
      : "";
    return `<div class="card verdict ${cls}"><div class="verdict-icon">${mark}</div>
      <div><div class="verdict-title">${esc(v.headline)}</div><div class="verdict-text">${esc(v.explanation)}</div>${repro}${rows}</div></div>`;
  }

  function progressCard(run) {
    const icon = { completed: "✅", error: "❌", skipped: "⏭", waiting: "⏸", pending: "○" };
    const rows = (run.nodes || []).map((n) => {
      const dur = n.status === "running"
        ? `<span class="dur" data-started="${esc(n.started_at || "")}"></span>`
        : n.duration_ms != null ? `<span class="dur">${(n.duration_ms / 1000).toFixed(1)} s</span>` : "";
      const mark = n.status === "running" ? '<span class="spinner"></span>' : icon[n.status] || "○";
      return `<li class="${esc(n.status)}"><span class="step-icon">${mark}</span>${esc(NODE_LABELS[n.name] || n.name)}${n.visits > 1 ? ` <span class="muted">×${n.visits}</span>` : ""}${dur}</li>`;
    }).join("");
    const done = run.status !== "running";
    return `<div class="card"><details${done ? "" : " open"}><summary style="color:inherit;font-weight:600">Agent workflow · ${esc(STATUS_LABELS[run.status] || run.status)}</summary><ul class="steps" style="margin-top:.6rem">${rows}</ul></details></div>`;
  }

  function updateRunningTimers() {
    for (const el of thread.querySelectorAll(".dur[data-started]")) {
      const started = el.dataset.started;
      if (started) el.textContent = `${((Date.now() - new Date(started).getTime()) / 1000).toFixed(0)} s`;
    }
  }

  const list = (items) => (items && items.length ? `<ul>${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>` : "");

  function analysisCard(a) {
    const extra = [
      ["Business rules", a.business_rules], ["Edge cases", a.edge_cases],
      ["Risk areas", a.risk_areas], ["Ambiguities", a.ambiguities],
    ].filter(([, v]) => v && v.length).map(([t, v]) => `<h4>${t}</h4>${list(v)}`).join("");
    return `<div class="card"><h3>🔍 Requirement analysis</h3>
      <p style="margin:0 0 .4rem">${esc(a.summary)}</p>
      ${a.feature ? `<span class="chip">${esc(a.feature)}</span>` : ""}
      <h4>Acceptance criteria</h4>${list(a.acceptance_criteria)}
      ${extra ? `<details><summary>More details</summary>${extra}</details>` : ""}</div>`;
  }

  function casesCard(cases) {
    const rows = cases.map((c) => `<tr>
        <td><b>${esc(c.test_case_id)}</b></td>
        <td>${esc(c.title)}<details><summary>Steps</summary><ol>${(c.steps || []).map((s) => `<li>${esc(s)}</li>`).join("")}</ol><div class="muted"><b>Expected:</b> ${esc(c.expected_result)}</div></details></td>
        <td><span class="chip">${esc(c.automation_type)}</span></td>
        <td><span class="chip ${esc(c.priority)}">${esc(c.priority)}</span></td></tr>`).join("");
    return `<div class="card"><h3>🧪 Test cases (${cases.length})</h3><div class="table-wrap"><table>
      <thead><tr><th>ID</th><th>Title</th><th>Runs as</th><th>Priority</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }

  function executeCard(count) {
    return `<div class="card action"><h3>▶ Ready to execute</h3>${count} test case(s) are planned. Run them against the configured test environment?
      <div class="actions"><button type="button" class="btn" data-action="execute">▶ Run tests</button></div></div>`;
  }

  function resultsCard(results, cases) {
    const titles = Object.fromEntries(cases.map((c) => [c.test_case_id, c.title]));
    const count = (s) => results.filter((r) => r.status === s).length;
    const n = results.length;
    const [p, f, e, s] = ["passed", "failed", "error", "skipped"].map(count);
    const pct = (x) => `${((x / n) * 100).toFixed(1)}%`;
    const rows = results.map((r) => `<tr>
        <td><b>${esc(r.test_case_id)}</b></td><td>${esc(titles[r.test_case_id] || "")}</td>
        <td>${esc((r.evidence && r.evidence.automation_type) || "")}</td>
        <td><span class="chip ${esc(r.status)}">${esc(r.status)}</span></td>
        <td>${r.duration_ms != null ? `${r.duration_ms} ms` : ""}</td>
        <td class="msg-cell">${esc(r.message || "")}</td></tr>`).join("");
    return `<div class="card"><h3>📊 Test results</h3>
      <div class="stats">
        <div class="stat"><b>${n}</b><span>Total</span></div>
        <div class="stat"><b style="color:var(--pass)">${p}</b><span>Passed</span></div>
        <div class="stat"><b style="color:var(--fail)">${f}</b><span>Failed</span></div>
        <div class="stat"><b style="color:var(--warn)">${e}</b><span>Errors</span></div>
        <div class="stat"><b>${s}</b><span>Skipped</span></div>
        <div class="stat"><b>${Math.round((p / n) * 100)}%</b><span>Pass rate</span></div>
      </div>
      <div class="bar"><i style="width:${pct(p)};background:#34d399"></i><i style="width:${pct(f)};background:#f87171"></i><i style="width:${pct(e)};background:#fbbf24"></i><i style="width:${pct(s)};background:#d1d5db"></i></div>
      <div class="table-wrap"><table><thead><tr><th>ID</th><th>Title</th><th>Runs as</th><th>Status</th><th>Time</th><th>Message</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }

  function rootCauseCard(a) {
    const items = a.findings.map((f) => {
      const cause = f.probable_root_cause || {};
      return `<div style="margin-top:.6rem">
        <b>${esc((f.test_case_ids || []).join(", "))}</b> · <span class="chip">${esc(f.suspected_origin || "")}</span>
        <span class="chip ${esc(f.severity_suggestion || "")}">${esc(f.severity_suggestion || "")}</span>
        <span class="muted"> confidence ${Math.round((f.confidence || 0) * 100)}%</span>
        <p style="margin:.35rem 0">${esc(cause.description || f.summary || "")}</p>
        ${cause.reasoning ? `<details><summary>Reasoning</summary><p class="muted">${esc(cause.reasoning)}</p></details>` : ""}</div>`;
    }).join("");
    return `<div class="card"><h3>🧠 Root cause analysis</h3><div class="muted">${esc(a.summary || "")}</div>${items}</div>`;
  }

  function bugsCard(bugs) {
    const items = bugs.map((b) => `<div style="padding:.7rem 0;border-top:1px solid var(--border)">
        <b>${esc(b.id)} · ${esc(b.title)}</b>
        <div style="margin:.3rem 0"><span class="chip ${esc(b.severity)}">severity: ${esc(b.severity)}</span> <span class="chip">priority: ${esc(b.priority)}</span> <span class="chip">${esc(String(b.report_type || "").replace(/_/g, " "))}</span></div>
        <div>${esc(b.summary)}</div>
        <details><summary>Reproduction & evidence</summary>
          <h4>Steps</h4><ol>${(b.reproduction_steps || []).map((s) => `<li>${esc(s)}</li>`).join("")}</ol>
          <h4>Expected</h4>${esc(b.expected_result)}<h4>Actual</h4>${esc(b.actual_result)}
        </details></div>`).join("");
    return `<div class="card"><h3>🐞 Bug reports (${bugs.length})</h3>${items}</div>`;
  }

  function reviewCard(r) {
    return `<div class="card review"><h3>🙋 Your review is needed</h3>
      <div><b>Why:</b> ${esc(r.reason)}</div>
      <div style="margin-top:.4rem"><b>AI recommendation:</b> ${esc(r.ai_recommendation)}</div>
      <div style="margin-top:.4rem"><b>Proposed action:</b> ${esc(r.proposed_action)}</div>
      <div style="margin-top:.4rem" class="muted">Confidence ${Math.round((r.confidence || 0) * 100)}%</div>
      <textarea id="review-comment" class="comment" placeholder="Comment (required for Re-analyse: tell the agent what to reconsider)"></textarea>
      <div class="actions">
        <button type="button" class="btn" data-action="review" data-decision="APPROVE">✔ Approve</button>
        <button type="button" class="btn danger" data-action="review" data-decision="REJECT">✖ Reject</button>
        <button type="button" class="btn ghost" data-action="review" data-decision="REQUEST_REANALYSIS">↻ Re-analyse</button>
      </div></div>`;
  }

  function decisionNote(r) {
    return `<div class="card"><h3>🙋 Human review</h3>Decision: <span class="chip">${esc(r.human_decision)}</span>${r.reviewer_comment ? ` — ${esc(r.reviewer_comment)}` : ""}</div>`;
  }

  function reportCard(rep, runId) {
    return `<div class="card"><h3>📄 Final QA report <span class="badge ${rep.status === "passed" ? "completed" : "error"}">${esc(rep.status)}</span></h3>
      <div>${md(rep.summary)}</div>
      <div class="stats" style="margin-top:.6rem">
        <div class="stat"><b>${Math.round((rep.pass_rate || 0) * 100)}%</b><span>Pass rate</span></div>
        <div class="stat"><b>${rep.total_tests}</b><span>Tests</span></div>
        <div class="stat"><b>${(rep.bug_reports || []).length}</b><span>Bugs</span></div>
        ${rep.confidence_score ? `<div class="stat"><b>${Math.round(rep.confidence_score.score * 100)}%</b><span>Confidence</span></div>` : ""}
      </div>
      <a class="btn ghost" style="text-decoration:none;display:inline-block" href="/runs/${esc(runId)}/report.md" download>⬇ Download report (.md)</a></div>`;
  }

  function errorsCard(errors) {
    return `<div class="card errors"><h3>⚠ Problems during the run</h3><ul>${errors.map((e) => `<li><b>${esc(NODE_LABELS[e.node] || e.node)}:</b> ${esc(String(e.message).slice(0, 300))}</li>`).join("")}</ul></div>`;
  }

  // --- actions ---------------------------------------------------------------------

  async function startRun(requirement) {
    state.sending = true;
    setComposerBusy(true);
    thread.innerHTML = userMsg(requirement) + agentMsg('<div class="agent-text typing"><span></span><span></span><span></span></div>');
    try {
      const run = await api("POST", "/runs", { requirement, project: "default", pause_before_execution: $("pause").checked });
      state.sending = false;
      await loadHistory();
      openRun(run.run_id);
    } catch (err) {
      state.sending = false;
      thread.innerHTML = userMsg(requirement) + agentMsg(`<div class="card errors">Could not start the run: ${esc(err.message)}</div>`);
    } finally {
      setComposerBusy(false);
    }
  }

  async function startTicketRun(ticket, pause) {
    state.sending = true;
    const steps = String(ticket.steps_to_reproduce || "").split("\n").filter((line) => line.trim());
    thread.innerHTML = ticketMsg({ ...ticket, steps_to_reproduce: steps })
      + agentMsg('<div class="agent-text typing"><span></span><span></span><span></span></div>');
    try {
      const run = await api("POST", "/runs", { bug_ticket: ticket, project: "default", pause_before_execution: pause });
      state.sending = false;
      await loadHistory();
      openRun(run.run_id);
    } catch (err) {
      state.sending = false;
      thread.innerHTML = agentMsg(`<div class="card errors">Could not start the verification: ${esc(err.message)}</div>`);
    }
  }

  async function ask(question) {
    const runId = state.activeId;
    const messages = loadChat(runId);
    const history = messages.slice(-16);
    messages.push({ role: "user", content: question });
    saveChat(runId, messages);
    state.sending = true;
    setComposerBusy(true);
    renderRun();
    scrollDown();
    let answer;
    try {
      answer = (await api("POST", `/runs/${runId}/chat`, { message: question, history })).answer;
    } catch (err) {
      answer = `⚠ Sorry, I could not answer: ${err.message}`;
    }
    state.sending = false;
    setComposerBusy(false);
    if (state.activeId !== runId) return;
    const updated = loadChat(runId);
    updated.push({ role: "assistant", content: answer });
    saveChat(runId, updated);
    renderRun();
    scrollDown();
  }

  async function runAction(button) {
    const runId = state.activeId;
    const action = button.dataset.action;
    const buttons = thread.querySelectorAll("[data-action]");
    buttons.forEach((b) => { b.disabled = true; });
    try {
      if (action === "execute") {
        await api("POST", `/runs/${runId}/execute`);
      } else if (action === "review") {
        const decision = button.dataset.decision;
        const comment = ($("review-comment") || {}).value || "";
        if (decision === "REQUEST_REANALYSIS" && !comment.trim()) {
          alert("Please write in the comment what the agent should reconsider.");
          buttons.forEach((b) => { b.disabled = false; });
          return;
        }
        await api("POST", `/runs/${runId}/review`, { decision, comment: comment.trim() || null, reviewer: "chat user" });
      }
    } catch (err) {
      alert(`Action failed: ${err.message}`);
      buttons.forEach((b) => { b.disabled = false; });
      return;
    }
    state.lastKey = null;
    state.awaitChange = { from: state.data.run.status, until: Date.now() + 30000 };
    stopPolling();
    poll();
    loadHistory();
  }

  function setComposerBusy(busy) {
    $("send").disabled = busy;
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
  }

  function closeSidebar() { $("sidebar").classList.remove("open"); }

  // --- events ----------------------------------------------------------------------

  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || state.sending) return;
    input.value = "";
    autosize();
    if (state.activeId) ask(text);
    else startRun(text);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  input.addEventListener("input", autosize);

  $("new-run").addEventListener("click", openNew);
  $("toggle-sidebar").addEventListener("click", () => $("sidebar").classList.toggle("open"));
  $("history").addEventListener("click", (event) => {
    const item = event.target.closest("[data-run]");
    if (item) openRun(item.dataset.run);
  });
  thread.addEventListener("submit", (event) => {
    if (event.target.id !== "ticket-form") return;
    event.preventDefault();
    const form = event.target;
    const values = Object.fromEntries(new FormData(form).entries());
    const missing = [["title", "Title"], ["expected_result", "Expected result"], ["actual_result", "Actual result"]]
      .filter(([k]) => !String(values[k] || "").trim())
      .map(([, label]) => label);
    if (missing.length) {
      const error = $("ticket-error");
      error.textContent = `Please fill in: ${missing.join(", ")}.`;
      error.hidden = false;
      return;
    }
    const pause = form.elements.pause.checked;
    delete values.pause;
    startTicketRun(values, pause);
  });

  thread.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-mode]");
    if (tab) {
      state.mode = tab.dataset.mode;
      renderWelcome();
      return;
    }
    if (event.target.closest("[data-action=sample-ticket]")) {
      const form = $("ticket-form");
      for (const [k, v] of Object.entries(SAMPLE_TICKET)) if (form.elements[k]) form.elements[k].value = v;
      return;
    }
    const example = event.target.closest("[data-example]");
    if (example) {
      input.value = EXAMPLES[Number(example.dataset.example)][1];
      autosize();
      input.focus();
      return;
    }
    const action = event.target.closest("[data-action]");
    if (action) runAction(action);
  });

  // --- start -----------------------------------------------------------------------

  loadSystemStatus();
  loadHistory().then(() => {
    const match = location.hash.match(/^#run=([\w-]+)$/);
    if (match) openRun(match[1]);
    else openNew();
  });
  setInterval(loadHistory, HISTORY_POLL_MS);
  setInterval(updateRunningTimers, 1000);
})();
