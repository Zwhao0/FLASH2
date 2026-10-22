import pathlib,struct,tempfile,unittest
from flash.local_controller import Ring,aggregate
class RingTest(unittest.TestCase):
 def test_committed_records_wrap_and_mode(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'ring';data=bytearray(64+2*128)
   data[:8]=b'FLASHNCL';struct.pack_into('<6I',data,8,1,64,128,2,0,3);struct.pack_into('<Q',data,32,3)
   for seq in [2,3]:
    off=64+(seq-1)%2*128
    struct.pack_into('<7Q',data,off,seq,100,200,1000,42,128,0)
    struct.pack_into('<8i',data,off+56,7,0,1,-1,3,4,3,0)
   p.write_bytes(data);ring=Ring(p);rows=ring.read()
   self.assertEqual([r['seq'] for r in rows],[2,3]);self.assertEqual(ring.lost,1)
   ring.mode(1);self.assertEqual(struct.unpack_from('<I',ring.mem,24)[0],1)
   self.assertEqual(aggregate(rows)[0]['avg_us'],1);ring.close()
if __name__=='__main__':unittest.main()

class MonitorSkipTest(unittest.TestCase):
 def test_monitor_history_is_not_reparsed_as_profile_data(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'ring';data=bytearray(64+2*128);data[:8]=b'FLASHNCL'
   struct.pack_into('<6I',data,8,1,64,128,2,0,3);struct.pack_into('<Q',data,32,3);p.write_bytes(data)
   ring=Ring(p);ring.skip();self.assertEqual(ring.read(),[]);self.assertEqual(ring.lost,0);ring.close()
