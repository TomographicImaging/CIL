
class InPlaceError(Exception):
    "Raised when the object should not be used in-place"
    def __init__(self,  message="Object should not be used in-place"):
        self.message = message
        super().__init__(self.message)
