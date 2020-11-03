class C(object):
    def __init__(self):
        self._x = 0

    @property
    def x(self):
        print('getting')
        return self._x

    @x.setter
    def set_x(self, value):
        print('setting')
        self._x = value


if __name__ == '__main__':
    c = C()
    print(c.x)
    c.x = 10
    print(c.x)