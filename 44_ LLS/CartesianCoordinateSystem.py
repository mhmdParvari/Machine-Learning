import cv2 as cv
import numpy as np

class CooSystem:
    def __init__(self, width, height, magnifier):
        # self.max_x = max_x
        # self.max_y = max_y
        self.magnifier = magnifier
        self.width = width
        self.height = height
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.field = np.zeros((self.height, self.width, 3), np.uint8)

        axis_x = [[0, self.center_y], [self.width-1, self.center_y]]
        axis_y = [[self.center_x, 0], [self.center_x, self.height-1]]
        # cv.drawContours(self.field, np.array([axis_x]), -1, (255,255,255), 2)
        # cv.drawContours(self.field, np.array([axis_y]), -1, (255,255,255), 2)
        cv.line(self.field, axis_x[0], axis_x[1], (255,255,255), 2)
        cv.line(self.field, axis_y[0], axis_y[1], (255,255,255), 2)
        

        for idx, i in enumerate(range(self.center_x + self.magnifier, self.width, self.magnifier)):
            cv.circle(self.field, (i, self.center_y), 4, (255,255,255), -1, cv.LINE_AA)
            cv.putText(self.field, str(idx+1), (i-2,self.center_y+20), cv.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv.circle(self.field, (self.width - i, self.center_y), 4, (255,255,255), -1, cv.LINE_AA)
            cv.putText(self.field, str(-(idx+1)), (self.width - i-6,self.center_y+20), cv.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            
        for idx, i in enumerate(range(self.center_y + self.magnifier, self.height, self.magnifier)):
            cv.circle(self.field, (self.center_x, i), 4, (255,255,255), -1, cv.LINE_AA)
            cv.putText(self.field, str(-(idx+1)), (self.center_x -30, i+5), cv.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            
            cv.circle(self.field, (self.center_x, self.height - i), 4, (255,255,255), -1, cv.LINE_AA)
            cv.putText(self.field, str(idx+1), (self.center_x -20, self.height - i + 5), cv.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)



    def plot_line_with_equation(self, a, b, c, color): # ax1 + bx2 = c
        points = []
        x1 = - (self.center_x / self.magnifier)
        x2 = (c - a * x1) / b
        x1 *= self.magnifier
        x2 *= self.magnifier
        points.append([round(self.center_x + x1) , round(self.center_y - x2)])

        x1 = self.center_x / self.magnifier
        x2 = (c - a * x1) / b
        x1 *= self.magnifier
        x2 *= self.magnifier
        points.append([round(self.center_x + x1) , round(self.center_y - x2)])
        
        # cv.drawContours(self.field, np.array([points]), -1, (0, 255, 0), 2)
        cv.line(self.field, points[0], points[1], color, 2)


    def plot_points(self, points, color):
        for point in points:
            x = round(self.center_x + point[0] * self.magnifier)
            y = round(self.center_y - (point[1] * self.magnifier))

            cv.circle(self.field, (x, y), 4, color, -1, cv.LINE_AA)

    def plot_point(self, x, y):
        x = round(self.center_x + x * self.magnifier)
        y = round(self.center_y - (y * self.magnifier))
        cv.circle(self.field, (x, y), 4, (255,0,0), -1, cv.LINE_AA)

    def show(self):
        # cv.imshow('h', self.field)
        cv.imwrite('h.jpg', cv.cvtColor(self.field, cv2.COLOR_BGR2HSV))

    def get_img(self):
        return self.field
