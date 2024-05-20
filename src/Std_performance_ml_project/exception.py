import sys

#Custom error message for Custom Exception Function
def error_message_detail(error, error_detail:sys):
    _,_,exec_tb=error_detail.exc_info()
    filename=exec_tb.tb_frame.f_code.co_filename
    error_message="Error occurred in python script name: [{0}] line numeber: [{1}] error message: [{2}] ".format(filename,exec_tb.tb_lineno,str(error))
    return error_message

# Custom Exception Funtion
def CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super.__init__(error_message)
        self.error_message=error_message(error_message,error_detail)
    
    def __str__(self):
        return str(self.error_message)
