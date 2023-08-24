import math

class DistanceBW2points():
    
    def __init__(self, first_X, first_Y, second_X, second_Y):
        self.x1, self.y1, self.x2, self.y2 = first_X, first_Y, second_X, second_Y
    
    def getDistance(self):
        x = pow(self.x2-self.x1, 2)
        y = pow(self.y2-self.y1, 2)
        return math.sqrt(x + y)



def main():
    pass

if __name__ == "__main__":
    main()