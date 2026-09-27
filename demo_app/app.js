// Client-side behaviour for the demo app. Every action goes through the demo auth API
// (/api/v2/*, served by qa_agent.tools.demo_server), which implements the knowledge
// base's login and password reset API documents, so UI and API tests describe the
// same product. Client-side checks only give faster feedback; the API is the authority.
(function () {
  const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const POLICY = "Password must be at least 12 characters and include a letter, a digit and a symbol.";
  const TOKEN_KEY = "demoAccessToken";
  const TEXT = {
    en: {
      title: "Forgot your password?",
      instructions: "Enter your registered email address and we will send you a reset link.",
      submit: "Send reset link",
    },
    es: {
      title: "¿Olvidaste tu contraseña?",
      instructions: "Introduce tu correo registrado y te enviaremos un enlace para restablecerla.",
      submit: "Enviar enlace",
    },
  };
  const ERRORS = {
    invalid_credentials: "Invalid email or password.",
    network: "Cannot reach the server. Is the demo server running?",
    account_locked: "Your account is locked after too many failed attempts. Try again in 15 minutes.",
    invalid_email: "Please enter a valid email address.",
    email_taken: "An account with this email already exists.",
    weak_password: POLICY,
    too_many_requests: "Too many reset requests. Please try again later.",
    token_invalid: "This reset link is invalid.",
    token_used: "This reset link has already been used.",
    token_expired: "This reset link has expired. Please request a new one.",
  };

  const byId = (id) => document.getElementById(id);
  const show = (el, text) => { if (text !== undefined) el.textContent = text; el.hidden = false; };
  const hide = (el) => { el.hidden = true; };
  const policyOk = (p) => p.length >= 12 && /[A-Za-z]/.test(p) && /\d/.test(p) && /[^A-Za-z0-9]/.test(p);
  const errorText = (data) => ERRORS[data && data.error] || "Something went wrong. Please try again.";

  async function api(method, path, body, token) {
    const headers = {};
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (token) headers.Authorization = `Bearer ${token}`;
    try {
      const res = await fetch(path, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      const data = res.status === 204 ? null : await res.json().catch(() => null);
      return { status: res.status, data };
    } catch (err) {
      return { status: 0, data: { error: "network" } };
    }
  }

  // Disable the submit button while a request is in flight.
  async function busy(form, work) {
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    try { await work(); } finally { button.disabled = false; }
  }

  // --- Forgot password -----------------------------------------------------------
  const forgot = byId("forgot-form");
  if (forgot) {
    byId("language").addEventListener("change", (event) => {
      const t = TEXT[event.target.value];
      document.querySelector("[data-testid=page-title]").textContent = t.title;
      document.querySelector("[data-testid=instructions]").textContent = t.instructions;
      document.querySelector("[data-testid=send-reset-link]").textContent = t.submit;
    });
    forgot.addEventListener("submit", (event) => {
      event.preventDefault();
      const email = byId("email").value.trim();
      hide(byId("email-error"));
      hide(byId("confirmation"));
      if (!EMAIL.test(email)) {
        show(byId("email-error"), ERRORS.invalid_email);
        return;
      }
      busy(forgot, async () => {
        const { status, data } = await api("POST", "/api/v2/password-reset", { email });
        // Same message whether or not the email is registered (no enumeration).
        if (status === 202) show(byId("confirmation"), data.message);
        else show(byId("email-error"), errorText(data));
      });
    });
  }

  // --- Reset password (link from the reset email) --------------------------------
  const reset = byId("reset-form");
  if (reset) {
    const token = new URLSearchParams(window.location.search).get("token");
    reset.addEventListener("submit", (event) => {
      event.preventDefault();
      const error = byId("reset-error");
      hide(error);
      const password = byId("new-password").value;
      const confirm = byId("confirm-password").value;
      if (!token) return show(error, ERRORS.token_invalid);
      if (!policyOk(password)) return show(error, POLICY);
      if (password !== confirm) return show(error, "Passwords do not match.");
      busy(reset, async () => {
        const { status, data } = await api("POST", "/api/v2/password-reset/confirm", {
          token,
          new_password: password,
        });
        if (status === 200) {
          sessionStorage.removeItem(TOKEN_KEY); // all sessions were revoked
          hide(reset);
          show(byId("reset-success"));
        } else {
          show(error, errorText(data));
        }
      });
    });
  }

  // --- Log in --------------------------------------------------------------------
  const login = byId("login-form");
  if (login) {
    const notice = new URLSearchParams(window.location.search).get("notice");
    if (notice === "registered") show(byId("login-notice"), "Account created. Please log in.");
    if (notice === "logged-out") show(byId("login-notice"), "You have been logged out.");

    login.addEventListener("submit", (event) => {
      event.preventDefault();
      const message = byId("login-message");
      const email = byId("login-email").value.trim();
      const password = byId("login-password").value;
      hide(message);
      busy(login, async () => {
        const { status, data } = await api("POST", "/api/v2/login", { email, password });
        const ok = status === 200;
        message.className = ok ? "success" : "error";
        show(message, ok ? "Welcome back!" : errorText(data));
        if (ok) {
          sessionStorage.setItem(TOKEN_KEY, data.access_token);
          setTimeout(() => location.assign("dashboard.html"), 800);
        }
      });
    });
  }

  // --- Sign up -------------------------------------------------------------------
  const signup = byId("signup-form");
  if (signup) {
    signup.addEventListener("submit", (event) => {
      event.preventDefault();
      const error = byId("signup-error");
      hide(error);
      const email = byId("signup-email").value.trim();
      const password = byId("signup-password").value;
      if (!EMAIL.test(email)) return show(error, ERRORS.invalid_email);
      if (!policyOk(password)) return show(error, POLICY);
      if (password !== byId("signup-confirm").value) return show(error, "Passwords do not match.");
      busy(signup, async () => {
        const { status, data } = await api("POST", "/api/v2/register", { email, password });
        if (status === 201) location.assign("login.html?notice=registered");
        else show(error, errorText(data));
      });
    });
  }

  // --- Dashboard (signed-in area) -----------------------------------------------
  const logout = byId("logout");
  if (logout) {
    const token = sessionStorage.getItem(TOKEN_KEY);
    api("GET", "/api/v2/me", undefined, token).then(({ status, data }) => {
      if (status !== 200) {
        sessionStorage.removeItem(TOKEN_KEY);
        location.replace("login.html");
        return;
      }
      byId("user-email").textContent = data.email;
      byId("account-initial").textContent = data.email.charAt(0).toUpperCase();
      show(byId("dashboard-content"));
    });
    logout.addEventListener("click", async () => {
      await api("POST", "/api/v2/logout", undefined, token);
      sessionStorage.removeItem(TOKEN_KEY);
      location.assign("login.html?notice=logged-out");
    });
  }

  // --- Demo inbox (stands in for real email) ------------------------------------
  const outbox = byId("outbox-form");
  if (outbox) {
    outbox.addEventListener("submit", (event) => {
      event.preventDefault();
      const email = byId("outbox-email").value.trim();
      const list = byId("outbox-list");
      list.replaceChildren();
      busy(outbox, async () => {
        const { data } = await api("GET", `/api/demo/outbox?email=${encodeURIComponent(email)}`);
        const emails = (data && data.emails) || [];
        byId("outbox-empty").hidden = emails.length > 0;
        for (const mail of emails.slice().reverse()) {
          const item = document.createElement("li");
          const link = document.createElement("a");
          link.href = mail.link;
          link.textContent = "Open reset link";
          item.append(`${mail.subject} → `, link);
          list.append(item);
        }
      });
    });
  }
})();
