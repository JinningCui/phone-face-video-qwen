import time
import unittest

from src.phone_face.live import LatestFrameReader


class LatestFrameReaderTests(unittest.TestCase):
    def test_slow_consumer_gets_recent_frame_instead_of_backlog(self):
        class Capture:
            def __init__(self):
                self.value = 0

            def read(self):
                time.sleep(0.003)
                self.value += 1
                return (self.value <= 20, self.value)

        reader = LatestFrameReader(Capture(), 0)
        reader.thread.join(timeout=1)
        ok, frame, sequence = reader.next(0, timeout=.1)
        self.assertTrue(ok)
        self.assertEqual((frame, sequence), (20, 20))
        self.assertEqual(reader.next(sequence, timeout=.01), (False, None, sequence))
        reader.close()


if __name__ == '__main__':
    unittest.main()
