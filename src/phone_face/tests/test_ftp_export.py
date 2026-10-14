import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from scripts import ftp_upload_faces
from src.phone_face.export_spool import publish_faces


class FakeFTP:
    files = {}
    directories = {'/'}

    def __enter__(self):
        self.current = '/'
        return self

    def __exit__(self, *_):
        return False

    def connect(self, *_args, **_kwargs):
        return None

    def login(self, *_args):
        return None

    def set_pasv(self, *_args):
        return None

    def cwd(self, path):
        from ftplib import error_perm
        destination = '/' if path == '/' else self.current.rstrip('/') + '/' + path
        if destination not in self.directories:
            raise error_perm('missing directory')
        self.current = destination

    def mkd(self, path):
        self.directories.add(self.current.rstrip('/') + '/' + path)

    def storbinary(self, command, handle, **_kwargs):
        self.files[self.current + '/' + command.removeprefix('STOR ')] = handle.read()

    def rename(self, source, target):
        self.files[self.current + '/' + target] = self.files.pop(self.current + '/' + source)


class FTPExportTests(unittest.TestCase):
    def test_placeholder_config_is_rejected_and_environment_can_override(self):
        with tempfile.TemporaryDirectory() as directory:
            example = Path(directory) / 'ftp.py'
            example.write_text("HOST = 'xxx'\nftp.login('xxx', 'xxx')\n")
            with self.assertRaisesRegex(ValueError, 'xxx'):
                ftp_upload_faces.credentials_from_config(example)
            with patch.dict('os.environ', {'FTP_HOST': 'localhost',
                                           'FTP_USER': 'example',
                                           'FTP_PASSWORD': 'example'}):
                self.assertEqual(ftp_upload_faces.credentials_from_config(example),
                                 ('localhost', 'example', 'example'))

    def test_only_complete_groups_upload_and_retries_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = root / 'ftp.py'
            example.write_text("HOST = '127.0.0.1'\nftp.login('demo', 'secret')\n")
            spool = root / 'spool'
            session, person = 'a' * 32, 'b' * 12
            images = [{'image': b'\xff\xd8' + bytes([i]), 'score': i + 1}
                      for i in range(5)]
            FakeFTP.files, FakeFTP.directories = {}, {'/'}
            with patch.object(ftp_upload_faces, 'FTP', FakeFTP):
                self.assertTrue(publish_faces(spool, session, person, images[:1]))
                self.assertEqual(ftp_upload_faces.sync_once(spool, example, '/faces'), (1, 1))
                self.assertEqual(len(FakeFTP.files), 1)
                self.assertEqual(ftp_upload_faces.sync_once(spool, example, '/faces'), (1, 0))
                self.assertTrue(publish_faces(spool, session, person, images[:2]))
                self.assertEqual(ftp_upload_faces.sync_once(spool, example, '/faces'), (1, 1))
                self.assertEqual(len(FakeFTP.files), 2)
                self.assertEqual(ftp_upload_faces.sync_once(spool, example, '/faces'), (1, 0))
                self.assertTrue(publish_faces(spool, session, person, images))
                for expected in (3, 4, 5):
                    self.assertEqual(ftp_upload_faces.sync_once(spool, example, '/faces'), (1, 1))
                    self.assertEqual(len(FakeFTP.files), expected)
                self.assertEqual(ftp_upload_faces.sync_once(spool, example, '/faces'), (1, 0))
            self.assertEqual(len(FakeFTP.files), 5)
            state = json.loads((root / 'ftp_upload_state' / f'{session}-{person}.json').read_text())
            self.assertRegex(state['remote_dir'],
                             r'^/faces/session_\d{8}_\d{6}/person_01$')
            self.assertIn(state['remote_dir'] + '/05.jpg', FakeFTP.files)

    def test_existing_uuid_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = root / 'ftp.py'
            example.write_text("HOST = '127.0.0.1'\nftp.login('demo', 'secret')\n")
            spool = root / 'spool'
            session, person = 'c' * 32, 'd' * 12
            images = [{'image': b'\xff\xd8' + bytes([i]), 'score': i + 1}
                      for i in range(3)]
            publish_faces(spool, session, person, images[:2])
            state_root = root / 'state'
            state_root.mkdir()
            legacy_dir = f'/faces/{session}/{person}'
            (state_root / f'{session}-{person}.json').write_text(
                json.dumps({'remote_dir': legacy_dir, 'images': {}}))
            FakeFTP.files, FakeFTP.directories = {}, {'/'}
            with patch.object(ftp_upload_faces, 'FTP', FakeFTP):
                self.assertEqual(ftp_upload_faces.sync_once(spool, example, '/faces', state_root), (1, 1))
                self.assertEqual(ftp_upload_faces.sync_once(spool, example, '/faces', state_root), (1, 1))
            self.assertIn(legacy_dir + '/01.jpg', FakeFTP.files)
            self.assertIn(legacy_dir + '/02.jpg', FakeFTP.files)


if __name__ == '__main__':
    unittest.main()
