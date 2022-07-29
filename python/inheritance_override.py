class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, val):
        self.value += val
    

class UpgradeCalculator(Calculator):
    def minus(self, val):
        self.value -= val

# inherit Calculator class and override 'add' method
class MaxLimitCalculator(Calculator):
    def add(self, val):
        self.value += val
        if self.value >= 100:
           self.value = 100    # why is this 'return 100' code incorrect? 


cal = UpgradeCalculator()
cal.add(10)
cal.minus(7)
print(cal.value)

cal2 = MaxLimitCalculator()
cal2.add(50)
cal2.add(60)
print(cal2.value)