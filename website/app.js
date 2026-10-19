const SITE_DATA_URL = "./assets/data/site-data.json?v=20261019-02";
const EPISODES_URL = "./assets/data/episodes.json?v=20261019-02";

const STAT_ICONS = {
  test: `
    <svg viewBox="0 0 24 24" class="stat-icon-svg" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M6.25 4.75H17.75V19.25H6.25V4.75Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" />
      <path d="M9 9H15M9 12H15M9 15H13" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
      <path d="M14.75 4.75V7.25H9.25V4.75" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" />
    </svg>
  `,
  train: `
    <svg viewBox="0 0 24 24" class="stat-icon-svg" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 3.75L19.25 7.5L12 11.25L4.75 7.5L12 3.75Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" />
      <path d="M4.75 12L12 15.75L19.25 12M4.75 16.25L12 20L19.25 16.25" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
  `,
  domains: `
    <svg viewBox="0 0 24 24" class="stat-icon-svg" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M5.25 5.25H11V11H5.25V5.25ZM13 5.25H18.75V11H13V5.25ZM5.25 13H11V18.75H5.25V13ZM13 13H18.75V18.75H13V13Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" />
    </svg>
  `,
  families: `
    <svg viewBox="0 0 24 24" class="stat-icon-svg" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="5.25" r="2.25" stroke="currentColor" stroke-width="1.8" />
      <circle cx="6" cy="18.5" r="2.25" stroke="currentColor" stroke-width="1.8" />
      <circle cx="18" cy="18.5" r="2.25" stroke="currentColor" stroke-width="1.8" />
      <path d="M12 7.5V11.75M6 16.25V13.75C6 12.65 6.9 11.75 8 11.75H16C17.1 11.75 18 12.65 18 13.75V16.25" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
    </svg>
  `,
  models: `
    <svg viewBox="0 0 24 24" class="stat-icon-svg" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M7.25 7.25H16.75V16.75H7.25V7.25Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" />
      <path d="M10 10H14V14H10V10Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" />
      <path d="M9.5 4.25V7.25M14.5 4.25V7.25M9.5 16.75V19.75M14.5 16.75V19.75M4.25 9.5H7.25M4.25 14.5H7.25M16.75 9.5H19.75M16.75 14.5H19.75" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
    </svg>
  `,
};

function byId(id) {
  return document.getElementById(id);
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function formatMetric(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "—";
}

function formatRangeValue(value) {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 2,
  }).format(value);
}

function formatRange(minimum, maximum) {
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) return "—";
  return `${formatRangeValue(minimum)} – ${formatRangeValue(maximum)}`;
}

async function loadJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  return response.json();
}

/* -------------------------------------------------------------------------
   Stats
   ------------------------------------------------------------------------- */

function renderStats(stats) {
  const root = byId("stats-grid");
  if (!root) return;

  if (!Array.isArray(stats) || !stats.length) {
    root.innerHTML = `<div class="empty-state" style="grid-column: 1 / -1;">Benchmark statistics are unavailable.</div>`;
    return;
  }

  const iconFor = (label, index) => {
    const lower = String(label || "").toLowerCase();
    if (lower.includes("test")) return STAT_ICONS.test;
    if (lower.includes("train")) return STAT_ICONS.train;
    if (lower.includes("domain")) return STAT_ICONS.domains;
    if (lower.includes("rule")) return STAT_ICONS.families;
    if (lower.includes("model")) return STAT_ICONS.models;
    return Object.values(STAT_ICONS)[index % Object.keys(STAT_ICONS).length];
  };

  root.innerHTML = stats
    .map(
      (stat, index) => `
        <article class="stat-card">
          <div class="stat-card-head">
            <span class="stat-icon" aria-hidden="true">${iconFor(stat.label, index)}</span>
            <div class="stat-label">${escapeHTML(stat.label)}</div>
          </div>
          <div class="stat-num">${escapeHTML(stat.value)}</div>
          ${stat.sub ? `<div class="stat-sub">${escapeHTML(stat.sub)}</div>` : ""}
        </article>
      `,
    )
    .join("");
}

/* -------------------------------------------------------------------------
   Leaderboard
   ------------------------------------------------------------------------- */

function sortedRows(rows) {
  return [...rows].sort((left, right) => {
    const leftCA = Number.isFinite(left.ca) ? left.ca : -Infinity;
    const rightCA = Number.isFinite(right.ca) ? right.ca : -Infinity;
    if (rightCA !== leftCA) return rightCA - leftCA;
    return String(left.model).localeCompare(String(right.model));
  });
}

function highlightValues(rows, key) {
  const values = rows
    .map((row) => row[key])
    .filter((value) => Number.isFinite(value));
  const unique = [...new Set(values)].sort((left, right) => right - left);
  return {
    best: unique[0],
    second: unique[1],
  };
}

function renderLeaderboardPanel(setting, metrics) {
  const rows = sortedRows(setting.rows || []);
  const highlights = Object.fromEntries(
    metrics
      .filter((metric) => metric.highlight)
      .map((metric) => [metric.key, highlightValues(rows, metric.key)]),
  );

  const groups = [];
  metrics.forEach((metric) => {
    const last = groups[groups.length - 1];
    if (last && last.label === metric.group) {
      last.span += 1;
    } else {
      groups.push({ label: metric.group, span: 1 });
    }
  });

  const groupHeader = groups
    .map((group) => `<th colspan="${group.span}" scope="colgroup">${escapeHTML(group.label)}</th>`)
    .join("");

  const metricHeader = metrics
    .map((metric) => {
      const arrow = metric.better === "high" ? " ↑" : "";
      return `<th scope="col" title="${escapeHTML(metric.full_label || metric.label)}">${escapeHTML(metric.label)}${arrow}</th>`;
    })
    .join("");

  const body = rows
    .map((row) => {
      const kind = row.kind === "proprietary" ? "proprietary" : "open";
      const kindLabel = kind === "proprietary" ? "Proprietary" : "Open-source";
      const metricCells = metrics
        .map((metric) => {
          const value = row[metric.key];
          let highlightClass = "";
          if (metric.highlight && Number.isFinite(value)) {
            if (value === highlights[metric.key]?.best) {
              highlightClass = " is-best";
            } else if (value === highlights[metric.key]?.second) {
              highlightClass = " is-second";
            }
          }
          const emptyClass = Number.isFinite(value) ? "" : " metric-value-empty";
          return `
            <td class="metric-value-cell">
              <span class="metric-value${highlightClass}${emptyClass}">${formatMetric(value)}</span>
            </td>
          `;
        })
        .join("");

      return `
        <tr>
          <td class="model-cell" data-label="Model">
            <div class="model-entry">
              <div class="model-topline">
                <strong class="model-name">${escapeHTML(row.model)}</strong>
              </div>
              <span class="model-kind model-kind-${kind}">${kindLabel}</span>
            </div>
          </td>
          ${metricCells}
        </tr>
      `;
    })
    .join("");

  return `
    <div class="leaderboard-setting-note">
      <p>${escapeHTML(setting.description || "")}</p>
      <span class="setting-note-pill">${escapeHTML(setting.note || "")}</span>
    </div>
    <div class="table-wrap">
      <table class="metric-table" aria-label="${escapeHTML(setting.label)} results">
        <thead>
          <tr class="group-row">
            <th rowspan="2" scope="col">Model</th>
            ${groupHeader}
          </tr>
          <tr class="metric-row">
            ${metricHeader}
          </tr>
        </thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  `;
}

function makeTabButton(label, id, controlsId, active) {
  const button = document.createElement("button");
  button.className = `tab-btn${active ? " active" : ""}`;
  button.type = "button";
  button.id = id;
  button.setAttribute("role", "tab");
  button.setAttribute("aria-controls", controlsId);
  button.setAttribute("aria-selected", active ? "true" : "false");
  button.tabIndex = active ? 0 : -1;
  button.textContent = label;
  return button;
}

function bindTabs(buttons, panels) {
  if (!buttons.length) return;

  const select = (index, focus = false) => {
    buttons.forEach((button, buttonIndex) => {
      const active = buttonIndex === index;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
      button.tabIndex = active ? 0 : -1;
    });
    panels.forEach((panel, panelIndex) => {
      panel.classList.toggle("active", panelIndex === index);
    });
    if (focus) buttons[index]?.focus();
  };

  buttons.forEach((button, index) => {
    button.addEventListener("click", () => select(index));
    button.addEventListener("keydown", (event) => {
      let nextIndex = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") {
        nextIndex = (index + 1) % buttons.length;
      } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
        nextIndex = (index - 1 + buttons.length) % buttons.length;
      } else if (event.key === "Home") {
        nextIndex = 0;
      } else if (event.key === "End") {
        nextIndex = buttons.length - 1;
      }
      if (nextIndex !== null) {
        event.preventDefault();
        select(nextIndex, true);
      }
    });
  });

  select(0);
}

function renderLeaderboard(board) {
  const tabsRoot = byId("leaderboard-tabs");
  const panelsRoot = byId("leaderboard-panels");
  if (!tabsRoot || !panelsRoot) return;

  if (!board || !Array.isArray(board.settings) || !board.settings.length) {
    tabsRoot.innerHTML = "";
    panelsRoot.innerHTML = `<div class="empty-state">Leaderboard data is unavailable.</div>`;
    return;
  }

  tabsRoot.innerHTML = "";
  panelsRoot.innerHTML = "";
  const buttons = [];
  const panels = [];

  board.settings.forEach((setting, index) => {
    const active = index === 0;
    const tabId = `setting-tab-${setting.id}`;
    const panelId = `setting-panel-${setting.id}`;
    const button = makeTabButton(setting.label, tabId, panelId, active);
    const panel = document.createElement("div");
    panel.className = `tab-panel${active ? " active" : ""}`;
    panel.id = panelId;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", tabId);
    panel.innerHTML = renderLeaderboardPanel(setting, board.metrics || []);

    tabsRoot.appendChild(button);
    panelsRoot.appendChild(panel);
    buttons.push(button);
    panels.push(panel);
  });

  bindTabs(buttons, panels);
}

/* -------------------------------------------------------------------------
   Takeaways
   ------------------------------------------------------------------------- */

function setupTakeawayCarousel(items, key, emptyLabel, startIndex = 0) {
  const stage = byId(`takeaway-${key}-stage`);
  const prevButton = byId(`takeaway-${key}-prev`);
  const nextButton = byId(`takeaway-${key}-next`);
  const counter = byId(`takeaway-${key}-counter`);
  const dots = byId(`takeaway-${key}-dots`);

  if (!stage || !prevButton || !nextButton || !counter || !dots) return;

  if (!Array.isArray(items) || !items.length) {
    stage.innerHTML = `<div class="empty-state">${escapeHTML(emptyLabel)}</div>`;
    prevButton.disabled = true;
    nextButton.disabled = true;
    counter.textContent = "0 / 0";
    dots.innerHTML = "";
    return;
  }

  let index = 0;

  function update() {
    const item = items[index];
    const pageLabel = String(startIndex + index + 1).padStart(2, "0");

    stage.innerHTML = `
      <article class="takeaway-card">
        <div class="takeaway-card-head">
          <span class="takeaway-page">${pageLabel}</span>
        </div>
        <p class="takeaway-statement">${escapeHTML(item.statement)}</p>
        <p class="takeaway-summary">${escapeHTML(item.summary)}</p>
      </article>
    `;

    counter.textContent = `${index + 1} / ${items.length}`;
    prevButton.disabled = index === 0;
    nextButton.disabled = index === items.length - 1;

    dots.innerHTML = items
      .map(
        (_, dotIndex) => `
          <button
            type="button"
            class="takeaway-dot${dotIndex === index ? " active" : ""}"
            data-index="${dotIndex}"
            aria-label="Go to takeaway ${dotIndex + 1}"
            aria-pressed="${dotIndex === index ? "true" : "false"}"
          ></button>
        `,
      )
      .join("");

    dots.querySelectorAll(".takeaway-dot").forEach((button) => {
      button.addEventListener("click", () => {
        index = Number(button.dataset.index);
        update();
      });
    });
  }

  prevButton.addEventListener("click", () => {
    if (index === 0) return;
    index -= 1;
    update();
  });

  nextButton.addEventListener("click", () => {
    if (index >= items.length - 1) return;
    index += 1;
    update();
  });

  update();
}

function renderTakeaways(takeaways) {
  const items = Array.isArray(takeaways) ? takeaways : [];
  const groups = {
    results: items.filter((item) => item.group === "results"),
    analysis: items.filter((item) => item.group === "analysis"),
    error: items.filter((item) => item.group === "error"),
  };

  let offset = 0;
  setupTakeawayCarousel(groups.results, "results", "Result takeaways are unavailable.", offset);
  offset += groups.results.length;
  setupTakeawayCarousel(groups.analysis, "analysis", "Analysis takeaways are unavailable.", offset);
  offset += groups.analysis.length;
  setupTakeawayCarousel(groups.error, "error", "Error-analysis takeaways are unavailable.", offset);
}

/* -------------------------------------------------------------------------
   Episode browser
   ------------------------------------------------------------------------- */

function renderActions(actions) {
  return (actions || [])
    .map((action) => {
      const rank = Number(action.strength_rank) || 1;
      const segments = [1, 2, 3, 4]
        .map(
          (step) =>
            `<span class="strength-segment${step <= rank ? " filled" : ""}" aria-hidden="true"></span>`,
        )
        .join("");
      return `
        <article class="action-card">
          <div class="action-card-head">
            <span class="action-rank">${String(rank).padStart(2, "0")}</span>
            <h4 class="action-name">${escapeHTML(action.display_name)}</h4>
          </div>
          <p>${escapeHTML(action.description)}</p>
          <div class="action-strength">
            <span class="strength-label">${escapeHTML(action.strength_label || "Intervention strength")}</span>
            <div class="strength-bar" role="img" aria-label="Intervention strength ${rank} of 4">${segments}</div>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderFields(fields) {
  const rows = (fields || [])
    .map(
      (field) => `
        <tr>
          <td>
            <span class="field-name">${escapeHTML(field.display_name)}</span>
            <code class="field-key">${escapeHTML(field.key)}</code>
          </td>
          <td class="field-description">${escapeHTML(field.description)}</td>
          <td class="field-unit">${escapeHTML(field.unit || "—")}</td>
          <td class="field-range">${formatRange(field.minimum, field.maximum)}</td>
        </tr>
      `,
    )
    .join("");

  return `
    <div class="fields-table-wrap">
      <table class="fields-table">
        <thead>
          <tr>
            <th scope="col">Field</th>
            <th scope="col">Description</th>
            <th scope="col">Unit</th>
            <th scope="col">Range</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function actionDisplayNames(actions) {
  return Object.fromEntries((actions || []).map((action) => [action.action_id, action.display_name]));
}

function renderUtilities(utilities, optimalAction, actionNames) {
  if (!utilities || typeof utilities !== "object") {
    return `<div class="empty-state" style="padding: 14px;">Reference utilities are unavailable for this case.</div>`;
  }

  const rows = Object.entries(utilities)
    .map(([actionId, rawValue]) => {
      const value = Number(rawValue);
      if (!Number.isFinite(value)) return "";
      const magnitude = Math.min(1, Math.abs(value));
      const side = value >= 0 ? "positive" : "negative";
      const optimal = actionId === optimalAction;
      const displayName = actionNames[actionId] || actionId;
      return `
        <div class="utility-row${optimal ? " is-optimal" : ""}" title="${escapeHTML(displayName)}">
          <span class="utility-label">${escapeHTML(actionId)}</span>
          <div class="utility-bar" aria-hidden="true">
            <span class="utility-fill ${side}" style="--magnitude: ${magnitude.toFixed(3)};"></span>
          </div>
          <span class="utility-value">${value.toFixed(2)}</span>
        </div>
      `;
    })
    .join("");

  return `
    <div class="utility-list">
      <div class="utility-list-head">
        <span>Reference utilities</span>
        <span>pre-change rule</span>
      </div>
      ${rows}
    </div>
  `;
}

function renderCases(cases, actions) {
  const actionNames = actionDisplayNames(actions);
  const cards = (cases || [])
    .map((caseItem) => {
      const payload = {
        context_id: caseItem.context_id,
        stratum: caseItem.stratum,
        raw_values: caseItem.raw_values,
        reference_utilities: caseItem.reference_utilities,
        reference_optimal_action: caseItem.reference_optimal_action,
      };
      return `
        <article class="case-card">
          <div class="case-head">
            <code class="case-id">${escapeHTML(caseItem.context_id)}</code>
            <span class="soft-pill">${escapeHTML(caseItem.stratum || "case")}</span>
          </div>
          <pre class="case-json">${escapeHTML(JSON.stringify(payload, null, 2))}</pre>
          ${renderUtilities(caseItem.reference_utilities, caseItem.reference_optimal_action, actionNames)}
        </article>
      `;
    })
    .join("");

  return cards || `<div class="empty-state">No sample cases are available for this episode.</div>`;
}

function renderRuleCard(rule, label, phase, changedFields) {
  const directionArrow = (direction) => (direction > 0 ? "↑" : "↓");
  const factors = (rule.relevant_fields || [])
    .map((factor) => {
      const tags = [];
      if (changedFields.replaced.has(factor.key)) tags.push("replaced");
      if (changedFields.reversed.has(factor.key)) tags.push("reversed");
      const tagMarkup = tags
        .map((tag) => `<span class="factor-tag">${tag}</span>`)
        .join("");
      return `
        <div class="factor-row${tags.length ? " is-changed" : ""}">
          <span class="factor-name">${escapeHTML(factor.display_name)}${tagMarkup}</span>
          <span class="factor-direction ${factor.direction > 0 ? "positive" : "negative"}">
            ${directionArrow(factor.direction)} ${escapeHTML(factor.direction_label)}
          </span>
        </div>
      `;
    })
    .join("");

  return `
    <div class="rule-card ${phase}">
      <p class="rule-card-label">${escapeHTML(label)}</p>
      <h4>${escapeHTML(rule.family_label)}</h4>
      <div class="rule-factors">${factors}</div>
    </div>
  `;
}

function renderTransition(transition, protocol) {
  const operations = transition.operations || [];
  const replaced = new Set();
  const reversed = new Set();

  operations.forEach((operation) => {
    if (operation.type === "field_replacement") {
      replaced.add(operation.outgoing_field);
      replaced.add(operation.incoming_field);
    } else if (operation.type === "direction_reversal") {
      reversed.add(operation.target_field);
    }
  });

  const changeItems = operations
    .map((operation) => {
      if (operation.type === "field_replacement") {
        return `
          <div class="change-item">
            <span class="change-icon" aria-hidden="true">↔</span>
            <div class="change-text">
              Replaced <strong>${escapeHTML(operation.outgoing_display_name)}</strong> with
              <strong>${escapeHTML(operation.incoming_display_name)}</strong> in the relevant factor set.
            </div>
          </div>
        `;
      }
      if (operation.type === "direction_reversal") {
        return `
          <div class="change-item">
            <span class="change-icon" aria-hidden="true">⇅</span>
            <div class="change-text">
              Reversed <strong>${escapeHTML(operation.target_display_name)}</strong> from
              <strong>${escapeHTML(operation.pre_direction_label)}</strong> to
              <strong>${escapeHTML(operation.post_direction_label)}</strong>.
            </div>
          </div>
        `;
      }
      return "";
    })
    .join("");

  const semantic = transition.semantic_change || {};
  const semanticBlock = semantic.trigger_event || semantic.change_reason
    ? `
      <div class="semantic-change">
        <span class="field-label">Semantic grounding</span>
        ${semantic.trigger_event ? `<p><strong>Trigger.</strong> ${escapeHTML(semantic.trigger_event)}</p>` : ""}
        ${semantic.change_reason ? `<p><strong>Why it is plausible.</strong> ${escapeHTML(semantic.change_reason)}</p>` : ""}
      </div>
    `
    : "";

  const totalBatches = protocol?.action_batch_count || 12;

  return `
    <details class="reveal-panel">
      <summary>
        <span>Reveal hidden rule</span>
        <span class="reveal-hint">pre-change → post-change</span>
      </summary>
      <div class="reveal-body">
        <p class="reveal-intro">
          Drift occurs after evaluation batch <strong>${transition.after_action_batch}</strong> of
          <strong>${totalBatches}</strong>. Both rules remain in the
          <strong>${escapeHTML(transition.pre_rule.family_label)}</strong> family. Directions are hidden
          during the interaction; this panel is for inspection.
        </p>
        <div class="rule-compare">
          ${renderRuleCard(transition.pre_rule, "Pre-change rule", "pre", { replaced, reversed })}
          ${renderRuleCard(transition.post_rule, "Post-change rule", "post", { replaced, reversed })}
        </div>
        <div class="change-list">${changeItems}</div>
        ${semanticBlock}
      </div>
    </details>
  `;
}

function renderEpisodePanel(episode) {
  return `
    <div class="scenario-card">
      <div class="scenario-badges">
        <span class="meta-pill">${escapeHTML(episode.domain_label)}</span>
        <span class="soft-pill">${escapeHTML(episode.rule_family_label)}</span>
        <span class="soft-pill">${escapeHTML(episode.branch)} branch</span>
        <span class="soft-pill">${escapeHTML(episode.template_id)}</span>
      </div>
      <p class="scenario-description">${escapeHTML(episode.scenario_description)}</p>
      <div class="role-block">
        <span class="field-label">Agent role</span>
        <p>${escapeHTML(episode.agent_role)}</p>
      </div>
    </div>

    <div class="explore-subhead">
      <h3>Actions by intervention strength</h3>
      <p>Four ordered actions, from the least to the most invasive.</p>
    </div>
    <div class="action-grid">${renderActions(episode.actions)}</div>

    <div class="explore-subhead">
      <h3>Field catalog</h3>
      <p>Eight observable fields; the hidden rule uses a subset.</p>
    </div>
    ${renderFields(episode.fields)}

    <div class="explore-subhead">
      <h3>Sample cases</h3>
      <p>Pre-change cases with reference utilities computed from the hidden rule.</p>
    </div>
    <div class="sample-grid">${renderCases(episode.cases, episode.actions)}</div>

    ${renderTransition(episode.transition, episode.interaction_protocol)}
  `;
}

function renderEpisodes(payload) {
  const tabsRoot = byId("domain-tabs");
  const panelsRoot = byId("domain-panels");
  const exploreRoot = byId("explore-root");
  const fallback = byId("explore-fallback");
  if (!tabsRoot || !panelsRoot || !exploreRoot || !fallback) return;

  const episodes = payload?.episodes;
  if (!Array.isArray(episodes) || !episodes.length) {
    throw new Error("episodes.json contains no episodes");
  }

  tabsRoot.innerHTML = "";
  panelsRoot.innerHTML = "";
  const buttons = [];
  const panels = [];

  episodes.forEach((episode, index) => {
    const active = index === 0;
    const tabId = `domain-tab-${episode.domain}`;
    const panelId = `domain-panel-${episode.domain}`;
    const button = makeTabButton(episode.domain_label, tabId, panelId, active);
    const panel = document.createElement("div");
    panel.className = `tab-panel domain-panel${active ? " active" : ""}`;
    panel.id = panelId;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", tabId);
    panel.innerHTML = renderEpisodePanel(episode);

    tabsRoot.appendChild(button);
    panelsRoot.appendChild(panel);
    buttons.push(button);
    panels.push(panel);
  });

  const utilityNote = byId("utility-note");
  if (utilityNote) {
    utilityNote.hidden = false;
    if (payload.utility_note) {
      utilityNote.textContent = payload.utility_note;
    }
  }

  fallback.hidden = true;
  exploreRoot.hidden = false;
  bindTabs(buttons, panels);
}

function showExploreFallback(error) {
  console.error(error);
  const exploreRoot = byId("explore-root");
  const fallback = byId("explore-fallback");
  const utilityNote = byId("utility-note");
  if (exploreRoot) exploreRoot.hidden = true;
  if (utilityNote) utilityNote.hidden = true;
  if (fallback) fallback.hidden = false;
}

/* -------------------------------------------------------------------------
   Citation
   ------------------------------------------------------------------------- */

async function copyTextToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.top = "-999px";
  textarea.style.left = "-999px";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();

  if (!copied) {
    throw new Error("Copy command was not available.");
  }
}

function setupCitationCopy() {
  const button = byId("copy-citation");
  const citation = byId("citation-bibtex");
  if (!button || !citation) return;

  const label = button.querySelector("span");
  const initialLabel = label?.textContent || "Copy";
  let resetTimer;

  button.addEventListener("click", async () => {
    const text = citation.textContent.trim();
    if (!text) return;

    button.disabled = true;
    try {
      await copyTextToClipboard(text);
      button.classList.add("copied");
      if (label) label.textContent = "Copied";
    } catch (error) {
      if (label) label.textContent = "Failed";
      console.error(error);
    } finally {
      button.disabled = false;
      clearTimeout(resetTimer);
      resetTimer = window.setTimeout(() => {
        button.classList.remove("copied");
        if (label) label.textContent = initialLabel;
      }, 1800);
    }
  });
}

/* -------------------------------------------------------------------------
   Boot
   ------------------------------------------------------------------------- */

function renderSiteDataError(error) {
  console.error(error);
  const stats = byId("stats-grid");
  if (stats) {
    stats.innerHTML = `<div class="empty-state" style="grid-column: 1 / -1;">Site data could not be loaded. Run <code>python3 website/scripts/build_data.py</code> to regenerate the bundle.</div>`;
  }

  const tabs = byId("leaderboard-tabs");
  const panels = byId("leaderboard-panels");
  if (tabs) tabs.innerHTML = "";
  if (panels) {
    panels.innerHTML = `<div class="empty-state">Leaderboard data is unavailable.</div>`;
  }

  ["results", "analysis", "error"].forEach((key) => {
    const stage = byId(`takeaway-${key}-stage`);
    if (stage) stage.innerHTML = `<div class="empty-state">Takeaways are unavailable.</div>`;
  });
}

async function init() {
  setupCitationCopy();

  try {
    const data = await loadJSON(SITE_DATA_URL);
    renderStats(data.stats || []);
    renderLeaderboard(data.leaderboard || null);
    renderTakeaways(data.takeaways || []);
  } catch (error) {
    renderSiteDataError(error);
  }

  try {
    const episodes = await loadJSON(EPISODES_URL);
    renderEpisodes(episodes);
  } catch (error) {
    showExploreFallback(error);
  }
}

init();
