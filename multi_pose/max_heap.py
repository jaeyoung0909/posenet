import math

def half(k):
    return math.floor(k / 2)

class MaxHeap():
    def __init__(self, maxSize, getElementValue):
        self.priorityQueue = [] 
        self.numberOfElements = -1
        self.getElementValue = getElementValue
    
    def enqueue(self, x):
        self.numberOfElements += 1
        self.priorityQueue[self.numberOfElements] = x 
        self.swim(self.numberOfElements)
    
    def empty(self):
        return self.numberOfElements == -1
    
    def size(self):
        return self.numberOfElements + 1
    
    def all(self):
        return self.priorityQueue[0: self.numberOfElements + 1]
    
    def max(self):
        return self.priorityQueue[0]
    
    def swim(self, k):
        while( k>0 and self.less(half(k), k)):
            self.exchange(k, half(k))
            k = half(k)
    
    def sink(self, k):
        while (2 * k <= self.numberOfElements):
            j = 2*k
            if j < self.numberOfElements and self.less(j, j+1):
                j+= 1
            if not self.less(k, j):
                break 
            self.exchange(k, j)
            k=j

    def getValueAt(self, i):
        return self.getElementValue(self.priorityQueue[i])
    
    def less(self, i, j):
        return self.getValueAt(i) < self.getValueAt(j)
    
    def exchange(self, i, j):
        t = self.priorityQueue[i]
        self.priorityQueue[i] = self.priorityQueue[j]
        self.priorityQueue[j] = t