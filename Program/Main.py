#
# MINIMUM LOCAL REGISTER ALLOCATION 
#
# INSTANCE: Sequence of instructions forming a basic block (no jumps), number N of available registers, cost $S_i$ (for $1\le i\le N$) of loading or storing register i. 
# SOLUTION: A register allocation for the instruction sequence.
# MEASURE: The total loading and storing cost for executing the instructions using the register allocation.
#

import sys;
import numpy as np;
import matplotlib.pyplot as plt;

#
# Instance
#
# n - number of registers.
# s - cost of loading or storing register i.
# p - program, sequence of what is read/written at each instruction (step).
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
                        p.append([argument[0][1:], 'R']);
                    if (not argument[1].isdigit()):
                        p.append([argument[1][1:], 'R']);
                    if (argument[2].isdigit()):
                        raise Exception("Erroneous program instruction. Destination register cannot be a constant.");
                    p.append([argument[2][1:], 'W']);

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
    pass;

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
