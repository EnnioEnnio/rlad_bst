from enum import Enum
from typing import Optional

import numpy as np


class Command(Enum):
    right = 0  # Move the pointer right
    left = 1  # Move the pointer right
    push = 2  # Add pointer to pointers
    pop = 3  # Remove last pointer
    compare = 4
    mark = 5  # Put the current command pointer on stack
    jump = 6  # Jump with the commandpointer to the last stack entry
    swap = 7
    isnotend = 8
    # Check we have not reached the end of the array and set it as skipflag
    isnotstart = 9  # Check we have not reached the beginning of the array and set it as skipflag # noqa: E501
    isnotequal = (
        10  # Checks that the pointers do not overlap and set it as skipflag
    )
    drop = 11  # Remove the last stack entry
    swapright = (
        12  # Swaps the element the pointer points to, with the one right of it
    )
    compareright = 13  # Checks if data at last pointer is smaller equal to the data right of it and set it as skipflag # noqa: E501


def check_sorted(data):
    # TODO: Add our own sorted check
    for i in range(len(data) - 1):
        if data[i] > data[i + 1]:
            return False
    return True


# write a function that returns a one hot encoding of an int given labels
def getcode(command: Command):
    onehot = np.zeros((len(Command)), dtype=bool)
    onehot[command.value] = 1
    return onehot


class SortMachine:
    def __init__(self, program: np.array, data: list, verbosity=2):
        self.program: np.array = program
        self.data: list = data
        self.pointers: list = [0]
        self.stack: list = []
        self.skipflag: bool = False
        self.commandpointer: int = 0
        self.lastcommand: np.array = np.zeros((1), dtype=bool)
        self.lastconditional: Optional[bool] = None
        self.execcost: int = 0
        self.storage: int = -1

        active = True
        print("Initial")
        print("program: ", "\n".join([self.getlabel(cmd) for cmd in program]))
        self.printsortmachine()
        print("Running")
        while active:
            active = self.transition_once()
            if verbosity >= 1:
                self.printsortmachine(verbosity)

    def transition_with_command(self, cmd):
        self.lastconditional = None
        if self.skipflag:
            self.skipflag = False
            return
        self.execcost += 1

        getattr(self, Command(np.argmax(cmd)).name)()

    def right(self):
        if self.pointers[-1] < len(self.data) - 1:
            self.pointers[-1] += 1

    def left(self):
        if self.pointers[-1] > 0:
            self.pointers[-1] -= 1

    def push(self):
        self.pointers.append(self.pointers[-1])

    def pop(self):
        if len(self.pointers) > 1:
            self.pointers.pop()

    def compare(self):
        self.skipflag = self.data[self.pointers[-1]] <= self.storage

    def mark(self):
        self.stack.append(self.commandpointer - 1)

    def jump(self):
        if len(self.stack) != 0:
            self.commandpointer = self.stack.pop()

    def swap(self):
        temp = self.storage
        self.storage = self.data[self.pointers[-1]]
        self.data[self.pointers[-1]] = temp

    def isnotend(self):
        self.skipflag = self.pointers[-1] >= len(self.data) - 1

    def isnotstart(self):
        self.skipflag = self.pointers[-1] == 0

    def isnotequal(self):
        if len(self.pointers) > 1:
            self.skipflag = self.pointers[-1] == self.pointers[-2]

    def drop(self):
        if len(self.stack) > 0:
            self.stack.pop()
        self.skipflag = False

    def swapright(self):
        temp = self.data[self.pointers[-1] + 1]
        self.data[self.pointers[-1] + 1] = self.data[self.pointers[-1]]
        self.data[self.pointers[-1]] = temp

    def compareright(self):
        self.skipflag = (
            self.data[self.pointers[-1]] <= self.data[self.pointers[-1] + 1]
        )

    def transition_once(self):
        if check_sorted(self.data) or self.commandpointer == len(self.program):
            return False
        cmd = self.program[self.commandpointer]
        self.lastcommand = cmd
        self.commandpointer += 1
        self.transition_with_command(cmd)
        return True

    def highlight_nth_element(self):
        # Convert the integer array into a string (space-separated)
        arr_str = " ".join(map(str, self.data))

        # Get the nth element to highlight
        nth_element = str(self.data[self.pointers[-1]])

        # Highlight the nth element by adding square brackets around it
        highlighted_str = arr_str.replace(nth_element, f"[{nth_element}]")

        return highlighted_str

    def printsortmachine(self, verbosity=1):
        print("---")
        if verbosity >= 2:
            print("data: ", self.highlight_nth_element())
        print("pointers: ", self.pointers)
        print("stack: ", self.stack)
        print("skipflag: ", self.skipflag)
        print("commandpointer: ", self.commandpointer)
        print("lastcommand: ", self.getlabel(self.lastcommand))
        print("lastconditional: ", self.lastconditional)
        print("execcost: ", self.execcost)
        print("---")

    def getlabel(self, onehot):
        idx = np.argmax(onehot)
        return Command(idx).name if onehot[idx] else "invalid"
