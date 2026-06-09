const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let toastTimer;
function toast(msg, kind = "") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show " + kind;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast " + kind), 3200);
}
async function call(cmd, args) {
  try {
    return await invoke(cmd, args);
  } catch (e) {
    toast(String(e), "err");
    throw e;
  }
}

/* ── tabs ── */
$$(".nav-item").forEach((b) =>
  b.addEventListener("click", () => {
    $$(".nav-item").forEach((x) => x.classList.remove("active"));
    $$(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("#" + b.dataset.tab).classList.add("active");
    refreshTab(b.dataset.tab);
  })
);
function refreshTab(t) {
  if (t === "dashboard") loadDashboard();
  if (t === "providers") loadProviders();
  if (t === "serving") loadEngines();
  if (t === "models") loadModels();
}

/* ── dashboard ── */
async function loadDashboard() {
  const st = await call("gateway_status");
  paintGateway(st);
  if (!$("#gw-host").value) $("#gw-host").value = st.host;
  if (!$("#gw-port").value) $("#gw-port").value = st.port;

  const info = await call("app_info");
  $("#info-body").innerHTML = [
    ["data dir", info.data_dir],
    ["providers", info.providers_path],
    ["logs", info.logs_dir],
    ["models", info.models_dir],
  ].map(([k, v]) => `<div class="k">${esc(k)}</div><div>${esc(v)}</div>`).join("");

  loadSetup();
}
function paintGateway(st) {
  const pill = $("#gw-pill");
  pill.textContent = st.running ? `gateway on · :${st.port}` : "gateway off";
  pill.className = "pill " + (st.running ? "on" : "off");
  $("#gw-baseurl").textContent = `http://${st.host}:${st.port}/v1`;
  $("#gw-start").disabled = st.running;
  $("#gw-stop").disabled = !st.running;
}
async function loadSetup() {
  $("#setup-body").textContent = "Detecting…";
  const r = await call("setup_report");
  const gpu = r.gpus.length ? r.gpus.map(esc).join("<br>") : '<span class="muted">none (CPU only)</span>';
  const ready = r.runtime_ready
    ? '<span class="tag up">ready</span>'
    : '<span class="tag wait">install Docker for image serving</span>';
  $("#setup-body").innerHTML =
    `<div class="k">GPU</div><div>${gpu}</div>` +
    `<div class="k">Docker</div><div>${esc(r.docker)} &nbsp; ${ready}</div>`;
}
$("#gw-start").addEventListener("click", async () => {
  const host = $("#gw-host").value.trim() || null;
  const port = parseInt($("#gw-port").value) || null;
  const st = await call("gateway_start", { host, port });
  paintGateway(st);
  toast("gateway started", "ok");
});
$("#gw-stop").addEventListener("click", async () => {
  await call("gateway_stop");
  paintGateway(await call("gateway_status"));
  toast("gateway stopped");
});
$("#setup-refresh").addEventListener("click", loadSetup);

/* ── providers ── */
async function loadProviders() {
  const list = await call("provider_list");
  const tb = $("#p-table tbody");
  if (!list.length) {
    tb.innerHTML = `<tr><td colspan="7" class="muted">No providers yet.</td></tr>`;
    return;
  }
  tb.innerHTML = list.map((p) => {
    const models = p.models.length ? p.models.join(", ") : "*";
    return `<tr>
      <td><input type="checkbox" class="switch" data-name="${esc(p.name)}" ${p.enabled ? "checked" : ""}></td>
      <td>${esc(p.name)}</td>
      <td>${esc(p.kind)}</td>
      <td class="mono">${esc(p.base_url)}</td>
      <td>${esc(models)}</td>
      <td>${p.needs_key ? "🔑" : "—"}</td>
      <td><button class="btn small danger" data-rm="${esc(p.name)}">Remove</button></td>
    </tr>`;
  }).join("");
  tb.querySelectorAll(".switch").forEach((c) =>
    c.addEventListener("change", () =>
      call("provider_set_enabled", { name: c.dataset.name, enabled: c.checked }).then(() => refreshGwPill())
    )
  );
  tb.querySelectorAll("[data-rm]").forEach((b) =>
    b.addEventListener("click", async () => {
      await call("provider_remove", { name: b.dataset.rm });
      toast("provider removed");
      loadProviders();
    })
  );
}
$("#p-add").addEventListener("click", async () => {
  const name = $("#p-name").value.trim();
  const base_url = $("#p-baseurl").value.trim();
  if (!name || !base_url) return toast("name and base URL are required", "err");
  await call("provider_add", {
    input: {
      name,
      base_url,
      kind: $("#p-kind").value,
      api_key: $("#p-key").value || null,
      models: $("#p-models").value.split(",").map((s) => s.trim()).filter(Boolean),
    },
  });
  ["#p-name", "#p-baseurl", "#p-key", "#p-models"].forEach((s) => ($(s).value = ""));
  toast("provider saved", "ok");
  loadProviders();
});
async function refreshGwPill() {
  paintGateway(await call("gateway_status"));
}

/* ── serving / engines ── */
async function loadEngines() {
  const list = await call("engine_list");
  const tb = $("#e-table tbody");
  if (!list.length) {
    tb.innerHTML = `<tr><td colspan="5" class="muted">No engines running.</td></tr>`;
    return;
  }
  tb.innerHTML = list.map((e) => {
    let tag = '<span class="tag down">stopped</span>';
    if (e.running) tag = e.healthy ? '<span class="tag up">healthy</span>' : '<span class="tag wait">starting…</span>';
    const err = e.last_error ? `<div class="muted">${esc(e.last_error)}</div>` : "";
    return `<tr>
      <td>${esc(e.name)}${err}</td>
      <td>${tag}</td>
      <td>${e.pid ?? "—"}</td>
      <td>${esc(e.last_started_at ?? "—")}</td>
      <td class="row gap">
        <button class="btn small" data-log="${esc(e.name)}">Logs</button>
        <button class="btn small danger" data-stop="${esc(e.name)}">Stop</button>
      </td>
    </tr>`;
  }).join("");
  tb.querySelectorAll("[data-stop]").forEach((b) =>
    b.addEventListener("click", async () => {
      await call("engine_stop", { name: b.dataset.stop });
      toast("engine stopped");
      loadEngines();
    })
  );
  tb.querySelectorAll("[data-log]").forEach((b) =>
    b.addEventListener("click", () => showLog(b.dataset.log))
  );
}
$("#s-go").addEventListener("click", async () => {
  const name = $("#s-name").value.trim();
  const model = $("#s-model").value.trim();
  const port = parseInt($("#s-port").value);
  if (!name || !model || !port) return toast("name, model and port are required", "err");
  const extra = $("#s-extra").value.trim();
  $("#s-status").textContent = "starting…";
  await call("serve_model", {
    input: {
      name,
      model,
      port,
      engine: $("#s-engine").value,
      runtime: $("#s-runtime").value,
      gpus: $("#s-gpus").value.trim() === "" ? null : $("#s-gpus").value.trim(),
      image: $("#s-image").value.trim() || null,
      wsl_distro: $("#s-wsl").value.trim() || null,
      served_id: null,
      hf_token: $("#s-hf").value || null,
      container_port: null,
      health_timeout: null,
      extra_args: extra ? extra.split(/\s+/) : [],
    },
  });
  toast("launching '" + name + "' — pulling image / loading weights may take a while", "ok");
  loadEngines();
});
$("#e-refresh").addEventListener("click", loadEngines);

async function showLog(name) {
  $("#log-title").textContent = "Log — " + name;
  $("#log-body").textContent = "Loading…";
  $("#log-modal").classList.remove("hidden");
  try {
    $("#log-body").textContent = (await call("engine_log", { name, lines: 300 })) || "(empty)";
  } catch {
    $("#log-body").textContent = "(no log)";
  }
}
$("#log-close").addEventListener("click", () => $("#log-modal").classList.add("hidden"));

/* ── models ── */
async function loadModels() {
  const list = await call("model_list");
  const tb = $("#m-table tbody");
  if (!list.length) {
    tb.innerHTML = `<tr><td colspan="3" class="muted">No local models.</td></tr>`;
    return;
  }
  tb.innerHTML = list.map((m) =>
    `<tr><td>${esc(m.repo)}</td><td>${(m.size_bytes / 1e9).toFixed(1)} GB</td><td>${m.file_count}</td></tr>`
  ).join("");
}
$("#m-refresh").addEventListener("click", loadModels);

/* ── events ── */
listen("serve-result", (ev) => {
  const p = ev.payload || {};
  $("#s-status").textContent = "";
  if (p.ok) toast("'" + p.name + "' is serving", "ok");
  else toast("'" + p.name + "' failed: " + (p.error || "unknown"), "err");
  loadEngines();
});

/* poll engines while the Serving tab is open */
setInterval(() => {
  if ($("#serving").classList.contains("active")) loadEngines();
}, 5000);

loadDashboard();
