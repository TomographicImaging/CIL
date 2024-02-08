def find_key(dic, val):
    """return the key of dictionary dic given the value"""
    return [k for k, v in dic.items() if v == val][0]


def message(cls, msg, *args):
    msg = "{0}: " + msg
    for i in range(len(args)):
        msg += " {%d}" %(i+1)
    args = list(args)
    args.insert(0, cls.__name__ )

    return msg.format(*args )
