const select = (selector) => document.querySelector(selector);
const countFormat = new Intl.NumberFormat("en-US");
const moneyFormat = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const smallPercent = new Intl.NumberFormat("en-US", { style: "percent", maximumSignificantDigits: 2 });
const percent = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 2 });
const metric = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });
const mixColors = ["#4355ce", "#15a18a", "#a28aeb", "#ed8c55", "#8fa634"];
const fieldLabels = {
  txn_id: "Transaction ID", step: "Time step", type: "Type", amount: "Amount",
  nameOrig: "Origin", oldbalanceOrg: "Origin balance before", newbalanceOrig: "Origin balance after",
  nameDest: "Destination", oldbalanceDest: "Destination balance before", newbalanceDest: "Destination balance after",
};

function scorePercent(value) {
  if (value > 0 && value < 0.00000001) return "<0.000001%";
  if (value > 0 && value < 0.0001) return smallPercent.format(value);
  if (value < 1 && percent.format(value) === "100%") return ">99.99%";
  return percent.format(value);
}

function element(tag, text) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  return node;
}

function message(selector, text, isError = false) {
  const node = select(selector);
  node.textContent = text;
  node.dataset.error = String(isError);
}

async function api(path) {
  const key = sessionStorage.getItem("finguard_api_key");
  const response = await fetch(path, { headers: key ? { "X-API-Key": key } : {} });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

async function loadHealth() {
  const node = select("#health-label");
  try {
    const health = await api("/health");
    const ready = health.status === "ready";
    node.dataset.ready = String(ready);
    node.textContent = ready ? "Service online" : "Setup required";
  } catch {
    node.dataset.ready = "false";
    node.textContent = "Service offline";
  }
}

function renderMix(byType, total) {
  const stops = [];
  const rows = [];
  let position = 0;
  for (const [name, count] of Object.entries(byType)) {
    const color = mixColors[rows.length % mixColors.length];
    const share = total ? count / total : 0;
    stops.push(`${color} ${position}% ${position + share * 100}%`);
    position += share * 100;
    const row = element("li");
    row.style.setProperty("--swatch", color);
    row.append(element("i"), element("span", name), element("b", countFormat.format(count)), element("small", metric.format(share)));
    rows.push(row);
  }
  select("#mix-chart").style.background = stops.length ? `conic-gradient(${stops.join(", ")})` : "";
  select("#type-count").textContent = rows.length;
  select("#type-list").replaceChildren(...rows);
}

async function loadSummary() {
  try {
    const { model, models, default_model_id, transactions, by_type } = await api("/dashboard/summary");
    select("#stat-total").textContent = countFormat.format(transactions);
    for (const [id, value] of [["prauc", model.pr_auc], ["precision", model.precision], ["recall", model.recall]]) {
      select(`#stat-${id}`).textContent = value == null ? "—" : metric.format(value);
    }
    select("#model-name").textContent = model.name;
    select("#model-features").textContent = model.feature_set === "pre_txn" ? "Pre-transaction" : model.feature_set;
    select("#model-threshold").textContent = scorePercent(model.threshold);
    renderModelMetrics(models, default_model_id);
    renderMix(by_type, transactions);
  } catch (error) {
    select("#model-name").textContent = "Unavailable";
    message("#summary-message", error.message, true);
  }
}

function comparisonRow(name, isDefault, values) {
  const row = element("tr");
  const title = element("th", name);
  title.scope = "row";
  if (isDefault) title.append(element("small", "Default"));
  row.append(title, ...values.map(value => element("td", value)));
  return row;
}

function renderModelMetrics(models, defaultId) {
  const rows = models.map(model => {
    const values = [model.test_pr_auc, model.test_precision, model.test_recall]
      .map(value => value == null ? "—" : metric.format(value));
    return comparisonRow(model.name, model.id === defaultId, [...values, scorePercent(model.threshold)]);
  });
  select("#model-metrics").replaceChildren(...rows);
  if (models[0]?.n_test) {
    select("#test-period").textContent = `Evaluated on ${countFormat.format(models[0].n_test)} held-out transactions.`;
  }
  const fallbacks = models.filter(model => model.threshold_method === "fallback_0.5");
  const note = select("#threshold-note");
  note.hidden = !fallbacks.length;
  note.textContent = `50% fallback threshold applies when validation precision falls short of the 80% target. Models: ${fallbacks.map(model => model.name).join(", ")}.`;
}

function renderScores(report) {
  const score = report.scores.find(model => model.model_id === report.default_model_id);
  select("#score-heading").textContent = `${score.name} score`;
  const rows = report.scores.map(model => comparisonRow(model.name, model.model_id === report.default_model_id,
    [scorePercent(model.model_score), scorePercent(model.flag_threshold), model.model_flags_fraud ? "Flagged" : "Below threshold"]));
  select("#transaction-scores").replaceChildren(...rows);
  const probability = score.model_score;
  const flagged = score.model_flags_fraud;
  select("#score-percent").textContent = scorePercent(probability);
  select("#score-value").textContent = probability.toString();
  select("#score-marker").style.left = `${probability * 100}%`;
  select("#threshold-marker").style.left = `${score.flag_threshold * 100}%`;
  select(".score-card").dataset.flagged = String(flagged);
  select("#risk-tag").textContent = flagged ? "Flagged by model" : "Below threshold";
  select("#risk-tag").dataset.level = flagged ? "drift" : "stable";
  select("#score-explanation").textContent =
    `Flag threshold ${scorePercent(score.flag_threshold)} (${score.flag_threshold})`;
  if (Array.isArray(score.features_used)) {
    select("#score-explanation").append(` · ${score.features_used.length} model inputs`);
  }
}

function renderFields(txn) {
  const rows = Object.entries(txn).map(([name, value]) => {
    const row = element("div");
    const isMoney = name === "amount" || name.toLowerCase().includes("balance");
    row.append(element("dt", fieldLabels[name] ?? name), element("dd", isMoney ? moneyFormat.format(value) : value));
    return row;
  });
  select("#transaction-fields").replaceChildren(...rows);
}

function renderPolicies(policies) {
  const rows = policies.map((policy, index) => {
    const row = element("details");
    row.className = "policy";
    row.open = index === 0;
    const summary = element("summary", policy.policy_id);
    summary.append(element("small", `Match ${Number(policy.score).toFixed(3)}`));
    row.append(summary, element("p", policy.text.replace(/^# .*\n/, "")));
    return row;
  });
  select("#policy-results").replaceChildren(...(rows.length ? rows : ["No matching policies."]));
}

let lookupVersion = 0;
async function investigate(event) {
  event.preventDefault();
  const id = new FormData(event.currentTarget).get("txn");
  const version = ++lookupVersion;
  select("#review-id").value = id;
  location.hash = "investigate";
  select("#review-title").textContent = `Transaction ${id}`;
  select("#review-content").hidden = true;
  message("#lookup-message", "Loading transaction…");
  try {
    const path = `/transactions/${encodeURIComponent(id)}`;
    const [txn, report] = await Promise.all([api(path), api(path + "/scores")]);
    if (version !== lookupVersion) return;
    renderScores(report);
    renderFields(txn);
    select("#policy-results").textContent = "Loading policies…";
    select("#review-content").hidden = false;
    message("#lookup-message", "");
    const ratio = txn.oldbalanceOrg > 0 ? txn.amount / txn.oldbalanceOrg : 0;
    const query = `${txn.type} transaction amount ${txn.amount}; origin balance ${txn.oldbalanceOrg}; amount balance ratio ${ratio.toFixed(2)}; destination balance ${txn.oldbalanceDest}`;
    try {
      const policies = await api("/policies/search?" + new URLSearchParams({ q: query, k: "3" }));
      if (version === lookupVersion) renderPolicies(policies);
    } catch (error) {
      if (version === lookupVersion) select("#policy-results").textContent = error.message;
    }
  } catch (error) {
    if (version === lookupVersion) message("#lookup-message", error.message, true);
  }
}

async function loadDrift() {
  const refresh = select("#refresh-drift");
  if (refresh.disabled) return;
  refresh.disabled = true;
  select("#drift-status").textContent = "Calculating…";
  select("#drift-grid").replaceChildren();
  select("#drift-note").textContent = "";
  try {
    const report = await api("/monitoring/drift?sample_size=10000");
    const headings = { stable: "Features stable", warning: "Drift warning", drift: "Drift detected" };
    select("#drift-status").textContent = headings[report.status] ?? report.status;
    select("#drift-note").textContent = `Latest ${countFormat.format(report.sample_size)} transactions compared with training data.`;
    const rows = Object.entries(report.features).map(([name, feature]) => {
      const row = element("div");
      row.className = "drift-row";
      const status = element("span", feature.status);
      status.className = "status";
      status.dataset.level = feature.status;
      row.append(element("span", fieldLabels[name] ?? name), element("strong", feature.psi.toFixed(4)), status);
      return row;
    });
    select("#drift-grid").replaceChildren(...rows);
  } catch (error) {
    select("#drift-status").textContent = "Report unavailable";
    select("#drift-note").textContent = error.message;
  } finally {
    refresh.disabled = false;
  }
}

function showView() {
  const target = ["investigate", "monitoring"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "overview";
  for (const section of document.querySelectorAll(".view")) section.hidden = section.id !== target;
  for (const link of document.querySelectorAll("nav a")) {
    if (link.hash === "#" + target) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  window.scrollTo(0, 0);
  if (target === "monitoring") loadDrift();
}

// Radial linework echoes the engraving on a banknote.
for (let index = 0; index < 40; index++) {
  const ellipse = element("i");
  ellipse.style.setProperty("--angle", `${index * 4.5}deg`);
  select(".rosette").append(ellipse);
}
for (const form of document.querySelectorAll("[data-lookup]")) form.addEventListener("submit", investigate);
select("#refresh-drift").addEventListener("click", loadDrift);
window.addEventListener("hashchange", showView);
showView();
loadHealth();
loadSummary();
