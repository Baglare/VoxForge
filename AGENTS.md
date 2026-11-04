# Project instructions: VoxForge

Görevle ilgili kaynaklardan başla, mevcut mimariyi ve kullanıcı değişikliklerini koru.
Yalnız hızlı, yerel ve hedefli doğrulama yap; çalıştırılmayan kontrolleri belirt.
Kullanıcı istemedikçe commit/push veya harici mutation yapma.

## Critical rules

- `local-voice-data-private` (`**`, error): Keep recordings, profiles, datasets, experiments, checkpoints and generated audio out of source control and public reports.
- `profile-path-containment` (`scripts/voice_profile_utils.py`, error): Validate profile slugs and resolved profile paths within the existing profiles directory before file operations.
- `reference-priority-and-model-lock` (`app/gradio_xtts_demo.py`, error): Preserve selected-profile then uploaded-reference then default-reference priority, locked model access and long-text chunking.
- `training-artifact-acceptance` (`scripts/train_xtts_gpt_experiment.py`, error): Training success requires a real checkpoint artifact; dry-run and trainer logs alone cannot establish success.
