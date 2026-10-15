export type Action = { action_id: string; label?: string; name?: string; description?: string };

export type EpisodeSummary = {
  episode_id: string;
  domain: string;
  rule_family: string;
  dynamic_type: DynamicType;
  sampling_branches: string[];
  after_action_batch: number;
  raw_field_count?: number;
  relevant_field_count: number;
  distractor_field_count: number;
  runs: string[];
};

export type DynamicType = "field_replacement" | "direction_reversal" | "compound";
export type DynamicScope = "all" | DynamicType;

export type RunSummary = {
  run_id: string;
  label: string;
  model_key: string;
  simulation_scenario?: string;
  simulation_description?: string;
  paired_static?: boolean;
  episode_ids: string[];
  metrics: Record<string, number | null>;
  batch_accuracy?: Array<{
    batch_number: number;
    accuracy: number | null;
    correct_cases: number;
    answered_cases: number;
    expected_cases: number;
  }>;
  batch_dynamic?: Record<string, Array<{
    batch_number: number;
    case_count: number;
    answered_cases: number;
    active_rule_accuracy: number | null;
    partial_credit_accuracy?: number | null;
    partial_credit_random_baseline?: number | null;
    static_accuracy?: number | null;
    old_rule_alignment: number | null;
    stale_policy_accuracy: number | null;
    flip_case_count?: number;
    nonflip_case_count?: number;
    flip_adoption_accuracy?: number | null;
    nonflip_stability_accuracy?: number | null;
    flipped_case_breakdown: {
      denominator: number;
      new_rule_adoption: number;
      stale_rule_adherence: number;
      neither_rule: number;
    };
  }>>;
  reasoning?: {
    parameter: string;
    value: string;
    thought_signature_calls: number;
    reasoning_summary_calls: number;
    total_calls: number;
  };
  hyperparameters?: Record<string, string | number | boolean | null>;
  breakdowns?: {
    by_family: Record<string, Record<string, number | null>>;
    by_domain: Record<string, Record<string, number | null>>;
  };
  views?: Record<DynamicScope, RunSummaryView>;
};

export type ProbeAllocationRow = {
  batch_number: number;
  probe_cases: number | null;
  action_utility_on_hand?: number | null;
};

export type RelevantFieldPhaseSummary = {
  episode_count: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
};

export type ChangeRecognitionEligibility = {
  eligible: number;
  total: number;
  rate: number | null;
};

export type RelevantFieldSummary = {
  pre_change: RelevantFieldPhaseSummary;
  post_change: RelevantFieldPhaseSummary;
  final: RelevantFieldPhaseSummary;
  paired_episode_count: number;
  changed_episode_count: number;
  unchanged_episode_count: number;
  changed_outcomes: {
    both_correct: number;
    pre_correct_final_incorrect: number;
    pre_incorrect_final_correct: number;
    both_incorrect: number;
  };
  unchanged_outcomes?: {
    both_correct: number;
    pre_correct_final_incorrect: number;
    pre_incorrect_final_correct: number;
    both_incorrect: number;
  };
  factor_recognition?: number | null;
  factor_recognition_pre?: number | null;
  factor_recognition_post?: number | null;
  factor_recognition_final?: number | null;
  change_recognition?: number | null;
  change_recognition_eligibility?: ChangeRecognitionEligibility | null;
  adaptation_episode_count?: number;
  adaptation_precision?: number | null;
  adaptation_recall?: number | null;
  change_atom_detection?: {
    field_replacement_atoms_detected: number;
    field_replacement_atoms_total: number;
    direction_reversal_atoms_detected: number;
    direction_reversal_atoms_total: number;
  };
  change_detection_outcomes?: {
    both_detected: number;
    field_replacement_only: number;
    direction_reversal_only: number;
    neither_detected: number;
    episode_count: number;
  };
};

export type RunSummaryView = {
  episode_count: number;
  metrics: Record<string, number | null> & {
    calibrated_accuracy?: number | null;
    partial_credit_accuracy?: number | null;
    partial_credit_random_baseline?: number | null;
  };
  first_learn_comparison?: {
    episode_count: number;
    dynamic_post_rule_accuracy: number | null;
    static_first_learn_accuracy: number | null;
    accuracy_delta: number | null;
    comparisons: Array<{
      episode_id?: string;
      dynamic_post_rule_accuracy: number | null;
      static_first_learn_accuracy: number | null;
      accuracy_delta: number | null;
      post_change_batch_count: number;
      static_reference_batch_count: number;
      comparison_batch_count: number;
    }>;
  };
  batch_accuracy: NonNullable<RunSummary["batch_accuracy"]>;
  batch_dynamic: NonNullable<RunSummary["batch_dynamic"]>;
  probe_by_batch?: ProbeAllocationRow[];
  probe_pre_dynamic_cases?: number | null;
  probe_post_dynamic_cases?: number | null;
  adaptation_curve?: Array<{
    batch_number: number;
    flip_case_count?: number;
    nonflip_case_count?: number;
    flip_adoption_accuracy?: number | null;
    nonflip_stability_accuracy?: number | null;
    flipped_case_breakdown?: {
      denominator: number;
      new_rule_adoption: number;
      stale_rule_adherence: number;
      neither_rule: number;
    };
  }>;
  breakdowns: NonNullable<RunSummary["breakdowns"]>;
  accuracy_breakdowns?: NonNullable<RunSummary["breakdowns"]>;
  utility_breakdowns?: NonNullable<RunSummary["breakdowns"]>;
  relevant_fields?: RelevantFieldSummary | null;
  branches?: Record<string, RunSummaryView>;
};

export type Manifest = {
  title: string;
  release: {
    name: string;
    slug: string;
    version: string;
  };
  dataset: string;
  benchmark_config: {
    schema_version: string;
    probe_budget: number;
    action_batch_count?: number;
    action_batch_size?: number;
    max_probe_rounds?: number;
    max_interactions?: number | null;
  };
  episode_count: number;
  family_counts: Record<string, number>;
  dynamic_type_counts: Record<DynamicType, number>;
  change_point_counts?: Record<string, number>;
  runs: RunSummary[];
  episodes: EpisodeSummary[];
};

export type FieldMeta = {
  type: "integer" | "decimal";
  minimum: number;
  maximum: number;
  unit?: string;
  description?: string;
  pressure_direction?: number;
};

export type CasePoint = {
  context_id: string;
  stratum: string;
  sampling_branch: string;
  phase?: "pre_dynamic" | "post_dynamic";
  family_diagnostic?: {
    pattern: string;
    group_id: string;
    role: string;
    fields: string[];
    [key: string]: unknown;
  } | null;
  raw_values: Record<string, number>;
  normalized_values: Record<string, number>;
  score: number;
  utilities: number[];
  oracle_action_index: number;
  oracle_action_id: string;
};

export type ProbePoint = {
  context_id: string;
  interaction: number;
  probe_round: number;
  before_action_batch: number;
  rule_phase: "pre_change" | "post_change";
  valid: boolean;
  mode: string;
  base_context_id?: string;
  changes?: Record<string, number>;
  returned_utilities: Array<number | null>;
  raw_values: Record<string, number>;
  normalized_values: Record<string, number>;
  score: number;
  utilities: number[];
  oracle_action_index: number;
  source_model?: string;
};

export type FieldCurve = {
  field_id: string;
  reference_value: number;
  samples: Array<{
    x: number;
    normalized_x: number;
    feasible: boolean;
    score: number | null;
    utilities: Array<number | null>;
  }>;
};

export type Heatmap = {
  x_field: string;
  y_field: string;
  size: number;
  region: { x: [number, number]; y: [number, number] } | null;
  cells: Array<{ x: number; y: number; xi: number; yi: number; feasible: boolean; score: number | null }>;
};

export type MechanismSummary = {
  family: string;
  detail: Record<string, unknown>;
  domain_rationale: string;
};

export type Episode = {
  episode_id: string;
  schema_version: string;
  setting: string;
  taxonomy: Record<string, number | string>;
  scenario: {
    domain: string;
    template_id: string;
    agent_role: string;
    actions: Action[];
  };
  interaction_spec: Record<string, number | null>;
  fields: Record<string, FieldMeta>;
  relevant_fields: string[];
  distractor_fields: string[];
  constraints: unknown[];
  mechanism: MechanismSummary;
  mechanism_raw: Record<string, unknown>;
  mechanism_views: Record<"pre_change" | "post_change", {
    relevant_fields: string[];
    mechanism: MechanismSummary;
    mechanism_raw: Record<string, unknown>;
    field_curves: FieldCurve[];
    heatmap: Heatmap | null;
  }>;
  dynamic?: {
    event: {
      after_action_batch: number;
      type?: DynamicType;
      dynamic_type?: DynamicType;
      dynamic_components?: DynamicType[];
      transition_id?: string;
      outgoing_field?: string;
      incoming_field?: string;
      change_event?: string;
      decision_role?: string;
      target_field?: string;
      stable_relevant_field?: string;
      permanent_distractor?: string;
      pre_direction?: number;
      post_direction?: number;
      decision_semantics_before?: string;
      decision_semantics_after?: string;
      why_direction_reverses?: string;
      operational_trigger?: string;
      why_outgoing_stops_controlling?: string;
      authority_before?: string;
      authority_after?: string;
      why_incoming_authoritative?: string;
      effective_scope?: string;
      expected_duration?: string;
      ending_condition?: string;
      realism_rationale?: string;
      operations?: Record<string, unknown>[];
      sampling_design: {
        candidate_slot_id: string;
        available_post_branches: string[];
      };
    };
    pre_dynamic: {
      relevant_fields: string[];
      mechanism: MechanismSummary;
      mechanism_raw: Record<string, unknown>;
    };
    post_dynamic: {
      relevant_fields: string[];
      mechanism: MechanismSummary;
      mechanism_raw: Record<string, unknown>;
    };
    population_change: {
      flip_rate: number;
      action_transition_matrix: number[][];
      mean_score_shift: number;
      mean_absolute_score_shift: number;
    };
  };
  reference_values: Record<string, number>;
  system_prompt: string;
  cases: CasePoint[];
  field_curves: FieldCurve[];
  heatmap: Heatmap | null;
  generation_diagnostics: Record<string, unknown>;
};

export type ActionResult = {
  context_id: string;
  action_id: string | null;
  utility: number;
  stratum: string;
  family_diagnostic?: CasePoint["family_diagnostic"];
  correct: boolean;
  regret: number;
  selected_action_index: number | null;
  oracle_action_index: number;
  unanswered: boolean;
  score: number;
  raw_values: Record<string, number>;
  oracle?: {
    optimal_action_id?: string;
    utilities?: Record<string, number>;
  };
};

export type TraceEvent = {
  request_id?: string;
  interaction: number;
  timestamp?: string;
  status?: string;
  request?: { role: string; content: string };
  request_messages?: Array<{ role: string; content: string }>;
  response?: { tool: string; arguments: Record<string, unknown> };
  raw_response?: string;
  reasoning_content?: string;
  reasoning_signature_present?: boolean;
  reasoning_available?: boolean;
  provider_raw_response?: unknown;
  result?: Record<string, unknown>;
  error?: string;
  error_type?: string;
  usage?: Record<string, number>;
  latency_seconds?: number;
  valid?: boolean | null;
  provider_trace?: Record<string, unknown> | null;
  provider_wire_request_characters?: number;
  canonical_visible_history_characters?: number;
  static_prompt_characters?: number;
  task_spec_characters?: number;
  latest_state_card_characters?: number;
  history_event_count?: number;
  provider_continuation_present?: boolean;
  reasoning_history_policy?: string;
  request_context?: {
    history_policy?: string;
    event_mapping?: Array<{
      source_event_id?: string;
      visible_event_id?: string | null;
      disposition?: string;
    }>;
    wire_event_groups?: string[][];
    wire_request?: {
      messages?: Array<Record<string, unknown>>;
      contents?: Array<Record<string, unknown>>;
    };
  };
};

export type ConversationArtifact = {
  schema_version: string;
  system_instruction?: string;
  system_prompt_hash: string;
  task_spec: Record<string, unknown>;
  events: Array<Record<string, unknown>>;
};

export type PersistentHistoryArtifact = {
  schema_version: string;
  history_policy: "eval_case_hidden" | "outcome_only";
  system_instruction?: string;
  task_spec: Record<string, unknown>;
  events: Array<Record<string, unknown>>;
  event_mapping?: Array<{
    source_event_id?: string;
    visible_event_id?: string | null;
    disposition?: string;
  }>;
};

export type RunBranch = {
  episode_id: string;
  run_id?: string;
  run_label?: string;
  model_key?: string;
  condition: "dynamic" | "static";
  setting_id?: string;
  sampling_branch?: string;
  source_sampling_branch?: string;
  enable_probe_skip?: boolean;
  least_probe_count?: number | null;
  history_policy?: "eval_case_hidden" | "eval_case_given" | "outcome_only" | "full_history" | "legacy_eval_case_given" | "legacy_full_history";
  history_policy_version?: string | null;
  history_view_status?: "available" | "eval_case_given_only" | "legacy_eval_case_given_only" | "full_history_only" | "legacy_full_history_only";
  run_status: string;
  run_success: boolean;
  termination: Record<string, unknown> | null;
  error?: string;
  metrics: Record<string, unknown>;
  probes: ProbePoint[];
  action_results: ActionResult[];
  tool_trace: TraceEvent[];
  conversation?: ConversationArtifact | null;
  agent_visible_history?: PersistentHistoryArtifact | null;
  initial_user_message?: string | null;
  relevant_fields_assessments?: Record<string, unknown>;
  trace_schema_version: string;
};

export type RunEpisode = {
  episode_id: string;
  run_id?: string;
  run_label?: string;
  model_key?: string;
  simulation_scenario?: string;
  simulation_description?: string;
  fork_after_batch: number;
  shared_prefix_interactions: number;
  run_status: string;
  run_success: boolean;
  trace_schema_version: string;
  branches: Record<string, RunBranch>;
};
