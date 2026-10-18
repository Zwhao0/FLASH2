import json
import unittest
from flash.history import advance_history

class RollingHistoryTest(unittest.TestCase):
    def test_consumes_new_iterations_after_redis_window_rolls(self):
        rows=[{'iteration':i,'time_ms':100.0} for i in range(550)]
        history,new,last,gap=advance_history([],rows[:500],-1)
        self.assertEqual((len(history),last,gap),(500,499,0))
        history,new,last,gap=advance_history(history,[json.dumps(x) for x in rows[50:]],last)
        self.assertEqual([x['iteration'] for x in new],list(range(500,550)))
        self.assertEqual((len(history),history[0]['iteration'],last,gap),(500,50,549,0))

    def test_repeated_poll_does_not_duplicate_history(self):
        rows=[{'iteration':i,'time_ms':100.0} for i in range(10)]
        history,new,last,gap=advance_history(rows,rows,9)
        self.assertEqual(history,rows)
        self.assertEqual((new,last,gap),([],9,0))

    def test_reports_a_real_consumer_overrun(self):
        history,new,last,gap=advance_history([], [{'iteration':i,'time_ms':100.0} for i in range(100,600)], 49)
        self.assertEqual((gap,len(history),last),(50,500,599))
