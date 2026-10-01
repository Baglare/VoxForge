---
{"critical_rules":[{"id":"local-voice-data-private","kind":"sensitive-area","scope":"**","severity":"error","statement":"Keep recordings, profiles, datasets, experiments, checkpoints and generated audio out of source control and public reports."},{"id":"profile-path-containment","kind":"invariant","scope":"scripts/voice_profile_utils.py","severity":"error","statement":"Validate profile slugs and resolved profile paths within the existing profiles directory before file operations."},{"id":"reference-priority-and-model-lock","kind":"invariant","scope":"app/gradio_xtts_demo.py","severity":"error","statement":"Preserve selected-profile then uploaded-reference then default-reference priority, locked model access and long-text chunking."},{"id":"training-artifact-acceptance","kind":"invariant","scope":"scripts/train_xtts_gpt_experiment.py","severity":"error","statement":"Training success requires a real checkpoint artifact; dry-run and trainer logs alone cannot establish success."}],"manifest_version":1,"project_id":"voxforge","project_name":"VoxForge","schema":"project-ai-manifest-v1"}
---
# Purpose

Local Windows Turkish TTS experiment environment using XTTS-v2, Gradio, file-based voice profiles and experimental fine-tuning.

# Repository Map

- tts-generation: app/gradio_xtts_demo.py, scripts/text_chunking_utils.py, scripts/audio_concat_utils.py
- voice-profiles: scripts/voice_profile_utils.py, scripts/create_voice_profile.py, scripts/delete_voice_profile.py, scripts/recreate_voice_profile.py
- audio-quality: scripts/audio_preprocessing_utils.py, scripts/audio_quality_utils.py, scripts/compare_reference_quality.py
- dataset-preparation: scripts/init_finetune_dataset.py, scripts/generate_recording_plan.py, scripts/build_metadata_from_recording_plan.py, scripts/validate_finetune_dataset.py, scripts/finetune_readiness_report.py
- fine-tuning-evaluation: scripts/export_xtts_finetune_dataset.py, scripts/train_xtts_gpt_experiment.py, scripts/evaluate_xtts_finetuned_checkpoint.py, scripts/evaluate_xtts_checkpoint_matrix.py, scripts/compare_finetune_experiments.py
- windows-operations: scripts/smoke_check.py, requirements.txt

# Architecture

app/gradio_xtts_demo.py owns UI orchestration and cached model access. scripts owns reusable audio, profile, dataset, training and evaluation utilities; PowerShell runners expose workflows. Reference priority is selected profile, uploaded audio, default sample. Long text is chunked before WAV concatenation. Fine-tuning preparation in Gradio runs readiness/export/dry-run; real training remains an explicit CLI operation.

# Validation Notes

Use syntax checks for changed Python files and scripts/smoke_check.py only when dependency/audio checks are requested. Model inference, GPU training and manual listening are separate acceptance gates; technical quality scores do not establish perceptual quality.

# Sensitive Areas

Local voice recordings, profiles, datasets, experiment outputs and checkpoints remain outside Git. Profile slugs and resolved paths must stay below the profile owner. Model access retains its existing lock.

# Non-goals

No public service, cloud storage or account system. Voice profile creation is not model fine-tuning; dry-run/log success is not checkpoint success.
