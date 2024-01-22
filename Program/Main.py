#
# MINIMUM LOCAL REGISTER ALLOCATION 
#
# INSTANCE: Sequence of instructions forming a basic block (no jumps), number N of available registers, cost $S_i$ (for $1\le i\le N$) of loading or storing register i. 
# SOLUTION: A register allocation for the instruction sequence.
# MEASURE: The total loading and storing cost for executing the instructions using the register allocation.
#

import sys;
import enum;
import numpy as np;
import matplotlib.pyplot as plt;

#
# usagetype_t
#
class usagetype_t(enum.Enum):
    READ  = 1;
    WRITE = 2;

    def __str__(self):
        if (self == usagetype_t.READ):
            return "READ";
        elif (self == usagetype_t.WRITE):
            return "WRITE";
    
    def __repr__(self):
        return self.__str__();

#
# Instance
#
# n - number of registers.
# s - cost of loading or storing register i.
# p - program, sequence of what is read/written at each instruction (step).
#     p[i] = [x, usagetype_t]
#
class Instance:
    def __init__(self, n, s, p):
        self.n = n;
        self.s = s;
        self.p = p;
    
    def __str__(self):
        return "n = {}\ns = {}\np = {}".format(self.n, self.s, self.p);

#
# ReadInstance
#
# Instance format:
# N                    - number of registers
# S_1 S_2 ... S_N      - cost of loading or storing register i
# [MOV|ADD|SUB] X1, X2 - program instructions
# ...
#
def ReadInstance(name):
    instance = None;

    try:
        with open("Document/"+name, "r") as f:
            n = int(f.readline());
            s = [int(x) for x in f.readline().split()];
            p = [];
            for line in f:
                instructionlist = line.split();
                instruction     = instructionlist[0];
                if (instruction.upper() == "MOV" or instruction.upper() == "ADD" or instruction.upper() == "SUB" or instruction.upper() == "MUL"):
                    argument = [ instructionlist[1].rsplit(',')[0], instructionlist[2].rsplit(',')[0], instructionlist[3] ];
                    if (not argument[0].isdigit()):
                        p.append((int(argument[0][1:]), usagetype_t.READ));
                    if (not argument[1].isdigit()):
                        p.append((int(argument[1][1:]), usagetype_t.READ));
                    if (argument[2].isdigit()):
                        raise Exception("Erroneous program instruction. Destination register cannot be a constant.");
                    p.append((int(argument[2][1:]), usagetype_t.WRITE));

            instance = Instance(n, s, p);
    except IOError:
        print("Instance \"{}\" was not found.".format(name));
        sys.exit(1);
    return instance;

#
# BruteForce
#
# Brute force method.
#
def BruteForce():
    instance = ReadInstance("Example 1.asm");
    print(instance);

    # Register configuration after each instruction.
    allocation = np.full((len(instance.p), instance.n), -1, dtype=int);
    
    # Generate all possible register allocations, count the cost of each one.
    # Find the minimum cost.
    mincost = sys.maxsize;
    n       = instance.n;
    stop_i  = 5;
    stack   = [(instance.p[0], 0, 0)]; # Stack of instructions to be processed. Contains the instruction, the placement of the register (index of register), and the index of the instruction in the program.
    while (len(stack) > 0):
        argument      = stack.pop();
        instruction   = argument[0];
        step_register = instruction[0];
        step_usage    = instruction[1];
        placement     = argument[1];
        i             = argument[2];
        
        # Set all register configurations after i to -1, meaning they are not used.
        for j in range(i, len(instance.p)):
            for k in range(0, n):
                allocation[j][k] = -1;

        # Copy previous allocation as the starting allocation for this step.
        if (i > 0):
            for j in range(0, n):
                allocation[i][j] = allocation[i-1][j];
        allocation[i][placement] = step_register;
        #print("Depth: {}\n Try: {} at {}\nAllocation: {}\n\n".format(i, instruction, placement, allocation));

        # Find place in allocation for next instruction.
        # -1 means register was never used before.
        # If it's not -1, then it's already in the allocation.
        if (i+1 < len(instance.p) and (i < stop_i)):
            next_instruction = instance.p[i+1];
            next_register    = next_instruction[0];
            next_i           = i+1;

            for j in range(0, n):
                if (allocation[i][j] == -1):
                    stack.append((next_instruction, j, next_i)); # does have cost
                elif (allocation[i][j] == next_register):
                    stack.append((next_instruction, j, next_i)); # no cost
                else:
                    # Some register has to be freed.
                    # Add all possible moves to the stack.
                    for k in range(0, n):
                        stack.append((next_instruction, k, next_i));


    print(allocation);

#
# Main
#
# Main entry point.
#
def Main():
    BruteForce();
    pass;

if __name__ == "__main__":
    Main();
