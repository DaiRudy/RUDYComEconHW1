import time
import subprocess

def run_and_time(script_path):
    """
    Call an external Python script and count its running time (in seconds).
    Returns the running time of the script.
    """
    start = time.time()
    # If necessary, you can change it to "python3" or an absolute path
    subprocess.run(["python", script_path], check=True)
    end = time.time()
    
    return end - start

if __name__ == "__main__":
    # Three scripts that need to be executed
    # Change Directory if applicable
    scripts = [
        r"C:\Users\19782\OneDrive - University of Florida\Dropbox_Backup\Course\Fang\hw1\HW1.3.2.py",
        r"C:\Users\19782\OneDrive - University of Florida\Dropbox_Backup\Course\Fang\hw1\HW1.4.1.py",
        r"C:\Users\19782\OneDrive - University of Florida\Dropbox_Backup\Course\Fang\hw1\HW1.4.2.py"
    ]

    # Execute one by one and print the running time
    for script in scripts:
        elapsed = run_and_time(script)
        print(f"Script: {script}\nTime used: {elapsed:.4f} seconds\n")
