import json
import logging
import queue

class MessageAnnouncer:
    """ A queuing mechanism for multiple listeners (clients) to receive announcements from this server.
        Code adapted from github.com/MaxHalford/flask-sse-no-deps.
    """

    def __init__(self):
        self.listeners = []

    def listen(self):
        self.listeners.append(queue.Queue(maxsize=5))
        return self.listeners[-1]

    def announce(self, event:str, data_dict:dict):
        """
        Takes a dict of values and sends it in an event to all listeners. 
        """
        logging.getLogger("messenger").info(f"Announcing event '{event}'")
        # We go in reverse order because we might have to delete an element, which will shift the
        # indices backward
        msg_str = format_dict_sse(event=event, data_dict=data_dict)
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg_str)
            except queue.Full:
                del self.listeners[i]


def format_dict_sse(event, data_dict: dict) -> str:
    """Formats a string and an event name in order to follow the event stream convention.
    Example output:
    >>> 'event: Jackson 5\\ndata: {"abc": 123}\\n\\n'
    """
    data = json.dumps(data_dict)
    msg = f'data: {data}\n\n'
    if event is not None:
        msg = f'event: {event}\n{msg}'
    return msg
