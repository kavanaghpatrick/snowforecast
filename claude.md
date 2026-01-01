{
  "rules": {
    "always_real_tests": true,
    "workflow": ["research", "plan", "branch", "code", "test", "document", "merge"],
    "stuck_threshold_min": 5,
    "stuck_action": "parallel_gemini_grok",
    "simplicity_test": "explain_in_2min",
    "ai_suggestion_default": "reject",
    "install_lib_threshold_min": 30,
    "grok_timeout_ms": 300000
  },
  "simplicity": {
    "max_pseudocode_lines": 5,
    "banned": [
      "abstract_interface_single_impl",
      "config_for_hardcoded",
      "elaborate_error_recovery_rare",
      "vendor_abstraction_before_2",
      "complex_sync_resumption",
      "enterprise_monitoring_mvp",
      "premature_optimization"
    ],
    "gut_check": ["impl_under_30min", "more_obvious", "solves_user_problem", "can_delete_code"],
    "auto_reject": ["timeline_over_2weeks", "dep_saves_under_4hours", "feature_under_80pct_users", "abstraction_under_3_usecases"],
    "stop_when": ["core_works", "happy_path_smooth", "data_safe", "basic_error_handling", "tests_pass"]
  },
  "automation": {
    "branch_before_risky": true,
    "parallelize_threshold": "2_unknowns",
    "context_switch_action": "write_2line_summary",
    "batch_file_ops": true,
    "learn_action": "update_claude_md"
  },
  "context": {
    "session_state_project": ".claude/context.md",
    "session_state_global": "~/notes/claude-context.md",
    "long_convo_summary": "docs/current_status.md",
    "file_read_method": "rg_or_line_limits_first"
  },
  "commands": {
    "project_dir": ".claude/commands/",
    "global_dir": "~/.claude/commands/",
    "usage": "create_file.md_becomes_/project:file",
    "share": "commit_to_git"
  },
  "workflow": {
    "steps": {
      "1_research": "task_tool_2plus_aspects",
      "2_plan": "prd_for_large_tasks,decompose_parallel",
      "3_branch": "feature_branch_before_code",
      "4_code": "simplicity_principles",
      "5_test": "real_tests_local",
      "6_document": "update_docs",
      "7_merge": "commit_why_not_what"
    },
    "research_checklist": ["existing_patterns", "best_practices", "edge_cases", "related_code"],
    "parallel_strategy": ["research_2to5_tasks", "prd_for_parallel", "impl_separate_worktrees", "merge_resolve_with_plan"]
  },
  "ai_tools": {
    "task_tool": {
      "use": "research_2plus_aspects",
      "example": "Task1:Find_auth_patterns+Task2:Research_JWT"
    },
    "gemini": {
      "use": ["architecture", "security", "debugging"],
      "limit": "1M_tokens",
      "cmd": "gemini -p \"Analyze: $(cat file.py)\""
    },
    "grok": {
      "use": "critical_reviews_crashes_security_dataloss",
      "model": "grok-4",
      "temperature": 0,
      "api": "https://api.x.ai/v1/chat/completions",
      "script": "python3 << 'EOF'\nimport json\ncode = open('file.py').read()\nreq = {\"messages\": [{\"role\": \"user\", \"content\": f\"Review:\\n{code}\"}], \"model\": \"grok-4\", \"temperature\": 0}\njson.dump(req, open('/tmp/req.json', 'w'))\nEOF\ncurl -X POST https://api.x.ai/v1/chat/completions -H \"Authorization: Bearer $GROK_API_KEY\" -H \"Content-Type: application/json\" -d @/tmp/req.json"
    },
    "codex": {
      "use": "long_autonomous_7to24h",
      "version": "0.38.0",
      "cmd_auto": "codex exec --full-auto \"task\"",
      "cmd_safe": "codex exec --sandbox read-only",
      "sandbox_limits": ["no_git_branch", "no_npm_install", "shell_errors_ok"],
      "workaround": "create_branch_manually_or_run_outside"
    }
  },
  "ai_review_workflow": {
    "steps": ["implement", "grok_review_critical", "apply_fixes", "retest", "commit"],
    "skip_review_result": "runtime_failures"
  },
  "ai_feedback_protocol": {
    "steps": ["filter_simplicity_protocol", "apply_gut_check", "default_no"]
  },
  "env": {
    "platform": "macos_arm64",
    "shell": "zsh",
    "python": "python3",
    "node": "v22",
    "conda_env": "claude-code",
    "conda_activate": false
  },
  "tools": {
    "search": {"best": ["rg", "fd"], "fallback": ["grep", "find"]},
    "python_pkg": {"best": "uv", "fallback": "pip3"},
    "node_pkg": {"best": "pnpm", "fallback": "npm"},
    "fuzzy": {"best": "fzf"}
  },
  "fixes": {
    "python_not_found": "use_python3",
    "pip_not_found": "use_pip3",
    "module_not_found": "conda_activate_claude-code",
    "yarn_bun_not_found": "use_npm_or_pnpm",
    "permission_denied": "chmod_+x",
    "error_recovery": ["analyze", "one_fix", "retry_once", "ask_if_fail"]
  },
  "auth": {
    "github": {
      "method": "macos_keyring",
      "cmd": "gh auth login",
      "never_set": ["GH_TOKEN", "GITHUB_TOKEN"]
    },
    "grok": {
      "env": "GROK_API_KEY",
      "location": "~/.zshrc"
    }
  },
  "multi_agent": {
    "terminal_titles": "update_at_task_start",
    "worktree_cmd": "git worktree add ../project-feature1 feature1-branch",
    "handoff": ["task_definition", "files_touched", "outstanding_tests", "blocking_items"],
    "state_sync": "one_agent_owns_git,others_rebase_before_push"
  },
  "file_hygiene": {
    "before_create": "search_existing_fd_rg",
    "banned_names": ["*_V2.md", "*_FINAL.md", "duplicate_prds"],
    "naming": "descriptive_concise_no_timestamps",
    "duplicates": ["extend_existing", "reference", "consolidate"]
  },
  "best_practices": {
    "comments": "explain_why_not_what",
    "testing": "real_tests_real_data_mocks_only_if_required",
    "dependencies": ["research_first", "document_why", "install_if_saves_time"],
    "security": ["never_commit_secrets", "use_env_vars", "review_deps"]
  }
}
