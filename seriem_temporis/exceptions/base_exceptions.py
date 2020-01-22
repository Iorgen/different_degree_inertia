class GeneratorException(Exception):
    """

    """
    def __init__(self, generate_status, message):
        self.generate_status = generate_status
        self.message = message
