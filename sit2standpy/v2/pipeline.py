__all__ = ['Sequential']


class Sequential:
    def __init__(self):
        """
        Sequential model for procssing a pipeline for IMU analysis

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency of the IMU data provided to the model

        Methods
        -------
        add(process)
            Add a processing step to the pipeline.
        predict(data)
            Predict/run the processing pipeline on the input data
        """
        self.procs = []

    def add(self, process):
        """
        Add a processing step to the pipeline.

        Parameters
        ----------
        process : class
            A instantiated process class, that has a `predict` method that takes in data as a dictionary or path to
            a HDF file(see :class:`~gaitpy.pipeline.Sequential.predict`)
        """

        self.procs.append(process)

    def predict(self, data):
        """
        Predict/run the processing pipeline on the input data

        Parameters
        ----------
        data : {str, dict}
            Either a H5 file path (string), or a dictionary. Both the H5 format and the dictionary must follow the
            below format.  The h5 file or dictionary will be modified in-place.

        Notes
        -----
        The layout for the input H5 file or the dictionary must be as follows:

        * Sensors
            * Lumbar
                * Accelerometer
                * Unix Time [units=seconds]
        
        """

        for proc in self.procs:
            proc.predict(data)
