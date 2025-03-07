import sys

from ..utils.logging import logger


class PhishingDetectionException(Exception):
    """
    Custom exception class for handling errors in the phishing website detection project.
    """

    def __init__(self, error_message: str, error_detail: sys):
        """
        Args:
            error_message (str): Error message to be displayed.
            error_detail (sys): System details of the error (e.g., traceback).
        """
        super().__init__(error_message)
        self.error_message = PhishingDetectionException.get_detailed_error_message(
            error_message, error_detail
        )

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: sys) -> str:
        """
        Constructs a detailed error message including file name, line number, and error message.
        If traceback is not available, returns a generic error message.
        """
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error occurred in file: [{file_name}] at line: [{line_number}]. Error: [{error_message}]"
        else:
            return f"Error occurred. Error: [{error_message}]"

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return str(self)


# Example usage (for testing)
if __name__ == "__main__":
    try:
        logger.info("Entering the try block")
        a = 1 / 0  # This will raise a ZeroDivisionError
    except Exception as e:
        raise PhishingDetectionException(str(e), sys)
