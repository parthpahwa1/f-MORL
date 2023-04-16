class ContinuousParameters:

    __LOG_SIG_MAX = 2
    __LOG_SIG_MIN = -10
    __VALUE_SCALING = 1000
    __epsilon = 1e-6

    __slots__ = ()

    @staticmethod
    def get_value_scaling():
        return ContinuousParameters.__VALUE_SCALING
    
    @staticmethod
    def get_log_sig_max():
        return ContinuousParameters.__LOG_SIG_MAX
    
    @staticmethod
    def get_log_sig_min():
        return ContinuousParameters.__LOG_SIG_MIN
    
    @staticmethod
    def get_epsilon():
        return ContinuousParameters.__epsilon
    