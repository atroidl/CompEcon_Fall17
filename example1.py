data = range(4)

def display():
    print ("Data:", data)

if __name__ == "__main__":
    display()
    print ("This file was executed from the command line or an interpreter.")
else:
    print ("This file was imported.")
