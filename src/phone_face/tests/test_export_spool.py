import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ftp_upload_faces import sync_once
from src.phone_face.export_spool import publish_gallery, retire_gallery


class ExportSpoolTests(unittest.TestCase):
    def test_gallery_pngs_upload_one_rank_per_poll(self):
        session, person = 'a' * 32, 'b' * 12
        png = b'\x89PNG\r\n\x1a\n'
        sent = []

        class FakeFTP:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def connect(self, *_args, **_kwargs): pass
            def login(self, *_args): pass
            def set_pasv(self, *_args): pass
            def cwd(self, *_args): pass
            def mkd(self, *_args): pass
            def storbinary(self, command, stream, **_kwargs):
                sent.append((command, stream.read()))
            def rename(self, *_args): pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spool, state = root / 'spool', root / 'state'
            images = [
                {'image': png + b'profile', 'score': 100, 'frontal': False},
                {'image': png + b'front', 'score': 80, 'frontal': True},
            ]
            self.assertTrue(publish_gallery(spool, session, person, images[:1]))
            self.assertTrue(publish_gallery(spool, session, person, images))
            with patch.dict(os.environ, {'FTP_HOST': 'localhost',
                                           'FTP_USER': 'test', 'FTP_PASSWORD': 'test'}), \
                    patch('scripts.ftp_upload_faces.FTP', FakeFTP):
                self.assertEqual(sync_once(spool, None, '/faces', state), (1, 1))
                self.assertEqual([item[0] for item in sent], ['STOR .01.png.uploading'])
                self.assertEqual(sync_once(spool, None, '/faces', state), (1, 1))
                self.assertEqual([item[0] for item in sent],
                                 ['STOR .01.png.uploading', 'STOR .02.png.uploading'])
                self.assertEqual(sync_once(spool, None, '/faces', state), (1, 0))

            retire_gallery(spool, session, person)
            self.assertEqual(list(spool.glob('*/*/manifest.json')), [])


if __name__ == '__main__':
    unittest.main()
