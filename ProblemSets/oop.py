
class Backpack(object):

    def __init__(self, name, color, max_size = 5):
        self.name = name
        self.contents = []
        self.color = color
        self.max_size = max_size

    def put(self, item):

        if len(self.contents) < self.max_size:
            self.contents.append(item)
        else:
            print("No Room!")


    def take(self, item):
        self.contents.remove(item)

    def dump(self):
       self.contents.clear()


class Jetpack(Backpack):

    def __init__(self, name, color, max_size = 2, fuel = 10):
        Backpack.__init__(self, name, color, max_size)
        self.fuel = fuel

    def fly(self, x):
        fly = (self.fuel - (0.15 * x))
        if fly < 0:
            print("Not enough fuel!")
        else:
            return fly


    def dump(self):
        Backpack.dump(self)
        self.fuel = 0
