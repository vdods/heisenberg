def prod (iterable, initial=None):
    if initial != None:
        return reduce(lambda a,b:a*b, iterable, initial)
    else:
        return reduce(lambda a,b:a*b, iterable)
