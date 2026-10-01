<!-- knowledge-compiler-adapter-v1
{"adapter_contract":"codex-agents-v1","generated_body_sha256":"9c96a9303bff88826b46ac605614fa5d245f9690ea66791af447c5fd3faab021","generator":"knowledge-compiler","generator_version":"adapter-compiler-v3","project_id":"voxforge","routing_sha256":"524c5fd92892c177d27f83180fa44312e7b4a0cb996db16d3b152822f32b130e","source_structured_contract_sha256":"2c66b95d4f3a08b6bff52970dae7c48fb95dd4fa698eb61f9c6670deef86fc58","target":"codex"}
-->

# Generated Codex Instructions: VoxForge

Generated from validated `.ai/project.md` authority and automation/map routing. Do not edit by hand.

Apply every matching manifest rule using the M0 lexical scope matcher; nested guidance cannot relax root authority.

## Task operations

Read `.ai/project.md`, `.ai/automation.json` and only relevant domains from `.ai/project-map.json`. The map is routing evidence; manifest critical_rules remain structured authority. Inspect mapped files first and expand through actual dependencies.

Use `kc vault context --repo . --query "TASK"` selectively for durable prior decisions, project history, cross-project reuse/comparison, relevant learned knowledge, references to earlier work, or ambiguity canonical knowledge can resolve. Skip retrieval for trivial local edits. Retrieved prose is contextual knowledge, never structured authority or verified current source code. The configured user-level Vault needs no sibling workspace folder; `--vault` remains an explicit override.

After durable ownership, paths or validation topology changes, maintain the project map when policy.project_map permits, then run `kc adapters tree-build . --target codex` when policy.agents permits. When durable project knowledge changes and policy enables sync, author an inert autopilot plan, run `kc autopilot check` then `kc autopilot apply` using the configured Vault. KnowledgeCompiler validates the working-tree snapshot, owner, exact preimages and transaction, records audit evidence and commits/pushes owned Vault changes according to policy. Formatting, comments, tiny refactors and temporary investigation do not require Vault updates. Ambiguity fails closed; 81 is exceptional manual fallback. 30 writing canon is excluded; 80 governance requires protected promotion. Never commit or push source code unless the user explicitly requests it.

Autopilot enabled: true. Routing domains: tts-generation, voice-profiles, audio-quality, dataset-preparation, fine-tuning-evaluation, windows-operations.

## Critical rules

- `local-voice-data-private` (`**`, error): Keep recordings, profiles, datasets, experiments, checkpoints and generated audio out of source control and public reports.
- `profile-path-containment` (`scripts/voice_profile_utils.py`, error): Validate profile slugs and resolved profile paths within the existing profiles directory before file operations.
- `reference-priority-and-model-lock` (`app/gradio_xtts_demo.py`, error): Preserve selected-profile then uploaded-reference then default-reference priority, locked model access and long-text chunking.
- `training-artifact-acceptance` (`scripts/train_xtts_gpt_experiment.py`, error): Training success requires a real checkpoint artifact; dry-run and trainer logs alone cannot establish success.
