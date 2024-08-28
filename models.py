from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# Base class for declarative class definitions.
Base = declarative_base()


class Log(Base):
    """
    A class used to represent a log entry in the database.

    Attributes
    ----------
    id : int
        The primary key of the log entry.
    description : str
        The description of the log entry.
    audio_file : str
        The filename of the associated audio file.
    spectrogram_file : str
        The filename of the associated spectrogram file.
    """
    __tablename__ = 'logs'

    id = Column(Integer, primary_key=True)
    description = Column(String)
    audio_file = Column(String)
    spectrogram_file = Column(String)


# Set up the database
engine = create_engine('sqlite:///logs.db')
Base.metadata.create_all(engine)

# Create a configured "Session" class
Session = sessionmaker(bind=engine)
