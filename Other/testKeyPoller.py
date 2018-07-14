from keyPoller import KeyPoller
with KeyPoller() as keyPoller:
    while True:
        c = keyPoller.poll()
        if not c is None:
            if c == "c":
                break
            print("Pressed the letter", c)