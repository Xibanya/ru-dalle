class Colors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'


def print_cyan(msg):
    print(f"{Colors.OKCYAN}{msg}{Colors.ENDC}")


def print_blue(msg):
    print(f"{Colors.OKBLUE}{msg}{Colors.ENDC}")


def print_green(msg):
    print(f"{Colors.OKGREEN}{msg}{Colors.ENDC}")


def print_warn(msg):
    print(f"{Colors.WARNING}{msg}{Colors.ENDC}")