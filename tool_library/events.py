import datetime


class Event:
    """
    A class to represent an event.
    """

    def __init__(self, time, tool, event_type, message):
        """
        Initialize the Event object.

        :param time: Time of the event
        :param event_type: Type of the event (string)
        :param message: Message associated with the event
        """
        self.time = time
        self.type = event_type
        self.message = message
        self.tool = tool

    def __str__(self):
        """
        String representation of the Event object.
        """
        return f"Time: {self.time}, Type: {self.type}, Tool: {self.tool}, Message: {self.message}"


class EventLog:
    """
    A class to represent a log of events.
    """

    def __init__(self):
        """
        Initialize the EventLog object.
        """
        self.last_check_time = datetime.datetime.now()
        self.events = []

    def add_event(self, event_type, tool, message):
        """
        Add a new event to the log.

        :param event_type: Type of the event (string)
        :param message: Message associated with the event
        """
        event_time = datetime.datetime.now()
        new_event = Event(event_time, tool, event_type, message)

        self.events.append(new_event)

    def find_events(self, minutes_ago=0):
        """
        Find all events since a certain time.

        :param minutes_ago: Time in minutes to look back for events. Default is 0 which means since the last check.
        :return: List of events that occurred since the specified time.
        """
        current_time = datetime.datetime.now()
        if minutes_ago == 0:
            last_check_time = self.last_check_time

        else:
            # Calculate the time to look back
            last_check_time = current_time - datetime.timedelta(minutes=minutes_ago)

        self.last_check_time = current_time

        # Filter events that occurred after the last check time
        return [event for event in self.events if event.time >= last_check_time]


if __name__ == "__main__":
    # Example Usage
    event_log = EventLog()
    event_log.add_event("Warning", "PC", "Low battery")
    event_log.add_event("Error", "PC", "System failure")

    # Query for events since 5 minutes ago
    recent_events = event_log.find_events(5)
    for event in recent_events:
        print(event)

    event_log.add_event("Error", "PC", "System failure again")

    print("Since?")

    recent_events = event_log.find_events(0)
    for event in recent_events:
        print(event)
