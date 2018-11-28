from utils import loading, print_rotation, myThread, print_time
import threading 

# loading()

# print_rotation()

# Create new threads
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)

# Start new Threads
thread1.start()
thread2.start()

print("Exiting Main Thread")