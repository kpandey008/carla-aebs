# Implementation of PID control


class PID():
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

    def step(self, target, feedback):
        error = target - feedback
        delta_error = error - self.last_error
        self.PTerm = self.Kp * error
        self.ITerm += self.Ki * error
        self.DTerm = self.Kd * delta_error 
        self.last_error = error

        self.output = max(0.0, self.PTerm + self.ITerm + self.DTerm)
        self.output = min(1.0, self.output)
        
        return self.output
