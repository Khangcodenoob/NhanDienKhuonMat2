import tkinter as tk
import subprocess
from tkinter import simpledialog

def run_add_faces():
    # Ask the user to input a name via a dialog box
    name = simpledialog.askstring("Input", "Enter the name:")
    
    if name:  # Only proceed if the user provided a name
        # Run the add_faces.py script, passing the name as an argument
        subprocess.run(["python", "add_faces.py", name])

def run_test():
    subprocess.run(["python", "test.py"])

# Create the main window
window = tk.Tk()
window.title("Face Recognition")

# Create buttons
add_faces_button = tk.Button(window, text="Add Faces", command=run_add_faces)
add_faces_button.pack(pady=10)

test_button = tk.Button(window, text="Test Faces", command=run_test)
test_button.pack(pady=10)

# Run the application
window.mainloop()
