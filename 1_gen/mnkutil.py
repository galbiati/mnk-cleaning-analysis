def bits2boards(num):
    s = '{0:b}'.format(num)
    return '0'*(36-len(s)) + s