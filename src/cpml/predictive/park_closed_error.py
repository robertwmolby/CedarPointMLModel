class ParkClosedError(Exception):
    """ Raised when attempt to predict park attendance is made on a day the park is closed"""