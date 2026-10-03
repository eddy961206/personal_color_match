"""Optional Python-stdlib launcher. The application itself runs entirely in the browser."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse

class LocalHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='color, me local-only launcher')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / 'web'
    handler = partial(LocalHandler, directory=str(root))
    print(f'브라우저에서 http://127.0.0.1:{args.port} 를 열어 줘. 종료: Ctrl+C')
    try:
        with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
            server.serve_forever()
    except KeyboardInterrupt:
        print('\n종료했어.')
