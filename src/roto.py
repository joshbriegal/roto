class RoTo:

    def __init__(self, timeseries, flux, flux_errors, **kwargs):

        self.timeseries = timeseries
        self.flux = flux
        self.flux_errors = flux_errors

        self.methods = []


    def __call__(self, *args):

        periods = [method() for method in self.methods]




    