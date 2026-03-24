import json
from pathlib import Path

models_dir = Path('../backend/trained_models')
files = sorted(models_dir.glob('*'))
print('=== SAVED FILES ===')
for f in files:
    print(f'  {f.name}  {f.stat().st_size//1024} KB')

print()
print('=== TRAINING METRICS SUMMARY ===')
metrics = json.load(open(models_dir / 'training_metrics.json'))
rows = sorted(metrics.items(), key=lambda x: -x[1].get('roc_auc', 0))
for name, m in rows:
    print(f'{name} -> acc={m.get("accuracy",0):.4f}  auc={m.get("roc_auc",0):.4f}  features={m.get("features",[])}')

print()
print('=== FEATURE REGISTRY ===')
fr = json.load(open(models_dir / 'feature_registry.json'))
for group, feats in fr.items():
    print(f'{group}: {feats}')
