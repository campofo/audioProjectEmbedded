from models import Session, Log
from sqlalchemy import desc


class FileLogger:
    """
    A class used to log events into the database and retrieve log entries.

    Attributes
    ----------
    session : Session
        A SQLAlchemy session object for interacting with the database.

    Methods
    -------
    log_event(description, audio_file, spectrogram_file)
        Logs a new event to the database.
    get_logs()
        Retrieves all log entries from the database.
    """

    def __init__(self):
        """
        Initializes the FileLogger with a new database session.
        """
        self.session = Session()

    def log_event(self, description, audio_file, spectrogram_file):
        """
        Logs a new event to the database.

        Parameters
        ----------
        description : str
            A description of the event.
        audio_file : str
            The filename of the associated audio file.
        spectrogram_file : str
            The filename of the associated spectrogram file.
        """
        new_log = Log(description=description, audio_file=audio_file, spectrogram_file=spectrogram_file)
        self.session.add(new_log)
        self.session.commit()

    def get_logs(self):
        """
        Retrieves all log entries from the database.

        Returns
        -------
        list of Log
            A list of all log entries, ordered by the newest first.
        """
        return self.session.query(Log).order_by(desc(Log.id)).all()
