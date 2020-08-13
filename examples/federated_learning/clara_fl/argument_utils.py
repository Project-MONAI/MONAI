from distutils.util import strtobool


def parse_var(s):
    items = s.split('=')
    key = items[0].strip()  # we remove blanks around keys, as is logical
    value = ''
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return key, value


def parse_vars(items):
    """
    To convert a list of key=value pairs into a dictionary.
    :param items: ['a=1', 'b=2', 'c=3']
    :return: {'a': '1', 'b': '2', 'c': '3'}
    """
    d = {}
    if items:
        for item in items:
            key, value = parse_var(item)

            # d[key] = value
            try:
                d[key] = int(value)
            except ValueError:
                pass
                try:
                    d[key] = float(value)
                except ValueError:
                    pass
                    try:
                        d[key] = bool(strtobool(str(value)))
                    except ValueError:
                        pass
                        d[key] = value
    return d
