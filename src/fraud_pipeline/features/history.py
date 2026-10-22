"""Rolling median of previously observed completed amounts."""
import bisect
from collections import deque


class RollingMedian:
    def __init__(self, seconds):
        self.window = seconds * 1000000
        self.rows, self.values = deque(), []

    def add(self, at, amount):
        self.rows.append((at, amount))
        bisect.insort(self.values, amount)

    def at(self, now):
        while self.rows and self.rows[0][0] < now - self.window:
            _, amount = self.rows.popleft()
            self.values.pop(bisect.bisect_left(self.values, amount))
        n = len(self.values)
        return n, (self.values[(n - 1)//2] + self.values[n//2])/2 if n else 0.


class RollingMean:
    """Mean of completed amounts in [now-window, now), observed causally."""
    def __init__(self, seconds):
        self.window=seconds*1000000
        self.rows=deque()
        self.total=0

    def add(self,at,amount):
        self.rows.append((at,amount));self.total+=amount

    def at(self,now):
        while self.rows and self.rows[0][0]<now-self.window:
            _,amount=self.rows.popleft();self.total-=amount
        n=len(self.rows)
        return n,self.total/n if n else 0.
