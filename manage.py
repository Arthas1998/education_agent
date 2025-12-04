import argparse
import os
import sys

# Ensure project root importable
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import create_app


def preconvert_all(app):
    pm = app.pdf_manager
    for cid, meta in list(pm.courses.items()):
        print(f"Converting course {cid} - {meta.title} ({meta.total_pages} pages)")
        try:
            pm.ensure_images(cid)
        except Exception as e:
            print(f"Error converting {cid}: {e}")


def run_server(host: str, port: int, debug: bool):
    app = create_app()
    app.run(host=host, port=port, debug=debug)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    runp = sub.add_parser('run')
    runp.add_argument('--host', default='0.0.0.0')
    runp.add_argument('--port', type=int, default=19980)
    runp.add_argument('--debug', action='store_true')

    pre = sub.add_parser('preproc')

    args = parser.parse_args()

    if args.cmd == 'run':
        run_server(args.host, args.port, args.debug)
    elif args.cmd == 'preproc':
        app = create_app()
        preconvert_all(app)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
