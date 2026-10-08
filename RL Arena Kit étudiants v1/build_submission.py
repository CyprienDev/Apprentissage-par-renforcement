"""Construire des archives reproductibles avec ou sans checkpoint."""
import argparse
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED


def build(cold=False):
    root = Path(__file__).resolve().parent
    source = (root/'student_agent.py').read_text(encoding='utf-8')
    if cold:
        signature = 'model_path=MODEL_PATH, seed=0, load=True, adaptive=True,'
        if source.count(signature) != 1:
            raise ValueError('Signature de MyAgent modifiée : vérifier l\'export sans modèle')
        source = source.replace(signature,signature.replace('load=True','load=False'),1)
    compile(source,'student_agent.py','exec')
    target = root/'artifacts'/('submission_cold.zip' if cold else 'submission.zip')
    target.parent.mkdir(exist_ok=True)
    with ZipFile(target,'w',ZIP_DEFLATED) as archive:
        archive.writestr('student_agent.py',source)
        for name in ('RAPPORT.md','SOUMISSION.md'):
            archive.write(root/name,name)
        if not cold:
            archive.write(root/'artifacts'/'policy.json','artifacts/policy.json')
        for directory in ('standard','stress'):
            path = root/'artifacts'/'v3'/directory/'evaluation.json'
            if path.exists():
                archive.write(path,str(path.relative_to(root)))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cold',action='store_true',help='table vide, aucun checkpoint lu par défaut')
    args = parser.parse_args()
    print(build(args.cold))


if __name__ == '__main__':
    main()
