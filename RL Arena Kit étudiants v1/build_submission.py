"""Construire des archives reproductibles avec ou sans checkpoint."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from student_agent import MODEL_VERSION


def build(cold=False):
    root = Path(__file__).resolve().parent
    source = (root/'student_agent.py').read_text(encoding='utf-8')
    model = root/'artifacts'/'policy.json'
    if not cold and model.stat().st_size > 10_000_000:
        raise ValueError('Le fichier de données dépasse la limite de 10 Mo')
    if not cold:
        data = json.loads(model.read_text(encoding='utf-8'))
        if data.get('version') != MODEL_VERSION:
            raise ValueError('La remise nécessite une table Q-learning compatible avec cet agent')
    if cold:
        signature = 'model_path=MODEL_PATH, seed=0, load=True, adaptive=False,'
        if source.count(signature) != 1:
            raise ValueError('Signature de MyAgent modifiée : vérifier l\'export sans modèle')
        source = source.replace(signature,signature.replace('load=True','load=False'),1)
    compile(source,'student_agent.py','exec')
    target = root/'artifacts'/('submission_cold.zip' if cold else 'submission.zip')
    target.parent.mkdir(exist_ok=True)
    with ZipFile(target,'w',ZIP_DEFLATED) as archive:
        archive.writestr('student_agent.py',source)
        if not cold:
            archive.write(model,'artifacts/policy.json')
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cold',action='store_true',help='table Q vide, aucun checkpoint chargé')
    args = parser.parse_args()
    print(build(args.cold))


if __name__ == '__main__':
    main()
