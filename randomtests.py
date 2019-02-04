import sys, termios, tty, os, time
 
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
 
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
 
button_delay = 0.01
 
while True:
    char = getch()
 
    if (char == "p"):
        print("Stop!")
        exit(0)
 
    if (char == "a"):
        print("Left pressed")
        time.sleep(button_delay)
 
    elif (char == "d"):
        print("Right pressed")
        time.sleep(button_delay)
 
    elif (char == "w"):
        print("Up pressed")
        time.sleep(button_delay)
 
    elif (char == "s"):
        print("Down pressed")
        time.sleep(button_delay)
 
    elif (char == " "):
        print("Jump pressed")
        time.sleep(button_delay)