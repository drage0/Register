#
# MINIMUM LOCAL REGISTER ALLOCATION 
#
# INSTANCE: Sequence of instructions forming a basic block (no jumps), number N of available registers, cost $S_i$ (for $1\le i\le N$) of loading or storing register i. 
# SOLUTION: A register allocation for the instruction sequence.
# MEASURE: The total loading and storing cost for executing the instructions using the register allocation.
#

import os
import sys;
import enum;
import numpy as np;
import matplotlib.pyplot as plt;
from copy import deepcopy;

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
# PrintLineSeparator
#
def PrintLineSeparator():
    print("----------------\n");

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
        with open("Document/"+name+".asm", "r") as f:
            n = int(f.readline());
            s = [int(x) for x in f.readline().split()];
            p = [];
            for line in f:
                instructionlist = line.split();
                instruction     = instructionlist[0];
                if (instruction.upper() == "ADD" or instruction.upper() == "SUB" or instruction.upper() == "MUL"):
                    argument = [ instructionlist[1].rsplit(',')[0], instructionlist[2].rsplit(',')[0], instructionlist[3] ];
                    if (not argument[0].isdigit()):
                        p.append((int(argument[0][1:]), usagetype_t.READ));
                    if (not argument[1].isdigit()):
                        p.append((int(argument[1][1:]), usagetype_t.READ));
                    if (argument[2].isdigit()):
                        raise Exception("Erroneous program instruction. Destination register cannot be a constant.");
                    p.append((int(argument[2][1:]), usagetype_t.WRITE));
                elif (instruction.upper() == "MOV"):
                    argument = [ instructionlist[1].rsplit(',')[0], instructionlist[2] ];
                    if (not argument[0].isdigit()):
                        p.append((int(argument[0][1:]), usagetype_t.READ));
                    if (argument[1].isdigit()):
                        raise Exception("Erroneous program instruction. Destination register cannot be a constant.");
                    p.append((int(argument[1][1:]), usagetype_t.WRITE));

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
def BruteForce(bake, name):
    instance_name         = name;
    instance_filepath_txt = f"Document/{instance_name} brute force.txt";
    instance              = ReadInstance(instance_name);
    result                = (None, 0);
    history               = [];

    print(instance);
    PrintLineSeparator();

    # Forcibly bake if the solution file doesn't yet exist.
    bake = (bake or not os.path.isfile(instance_filepath_txt));

    if (bake):
        print(f"Bake file {instance_filepath_txt}...");

        # Register configuration after each instruction.
        allocation = np.full((len(instance.p), instance.n), -1, dtype=int);
        
        #
        # Generate all possible register allocations, count the cost of each one.
        # Find the minimum cost.
        # 
        # Stack:
        #  [0] - Instruction (register, usage) . Contains the instruction, the placement of the register (index of register), cost, and the index of the instruction in the program.
        #  [1] - Placement of the register.
        #  [2] - Total cost after entering program instruction number 'i' ([3]).
        #  [3] - Index of the program instruction.
        #
        mincost           = sys.maxsize;
        mincostallocation = None;
        n                 = instance.n;
        stop_i            = -1;
        stack             = [];
        iteration         = 0;
        for i in range(0, n):
            stack.append((instance.p[0], i, instance.s[i], 0));
        while (len(stack) > 0):
            argument      = stack.pop();
            instruction   = argument[0];
            step_register = instruction[0];
            step_usage    = instruction[1];
            placement     = argument[1];
            cost          = argument[2];
            i             = argument[3];
            iteration     = 1+iteration;
            
            if (iteration % 100_000 == 0):
                history.append((iteration, mincost));
                print("Iteration: {}".format(iteration));
                print("Minimum cost: {}".format(mincost));
                PrintLineSeparator();
            
            #
            # No need to continue if the cost is already higher than the minimum cost.
            #
            if (cost > mincost):
                continue;

            # Set all register configurations after i to -1, meaning they are not used.
            for j in range(i, len(instance.p)):
                allocation[j, :] = -1
            
            # Copy previous allocation as the starting allocation for this step.
            if (i > 0):
                allocation[i] = allocation[i-1].copy()
            allocation[i][placement] = step_register;
            
            #
            # Find place in allocation for next instruction.
            # -1 means register was never used before.
            # If it's not -1, then it's already in the allocation.
            #
            if (i+1 < len(instance.p) and (stop_i < 0 or i < stop_i)):
                next_instruction = instance.p[i+1];
                next_register    = next_instruction[0];
                next_i           = i+1;

                #  Random permutation of random(0, n)
                randomlist = np.random.permutation(n);

                for j in randomlist:
                    if (allocation[i][j] == -1):
                        stack.append((next_instruction, j, cost+instance.s[j], next_i)); # Added cost of loading the resgister.
                        break;
                    elif (allocation[i][j] == next_register):
                        if (step_usage == usagetype_t.WRITE):
                            stack.append((next_instruction, j, cost+instance.s[j], next_i)); # Cost of writing to the register.
                        else:
                            stack.append((next_instruction, j, cost, next_i)); # No cost of reading from the register (it's already loaded).
                        break;
                    else:
                        # Some register has to be freed.
                        # Add all possible moves to the stack.
                        randomlist_2 = np.random.permutation(n);
                        for k in randomlist_2:
                            stack.append((next_instruction, k, cost+instance.s[j]+instance.s[k], next_i)); # Added cost of storing the register and loading another.
            else:
                if (cost < mincost):
                    mincost = cost;
                    mincostallocation = np.copy(allocation);
                    print("New minimum cost: {}".format(mincost));
                    print(allocation);
                    PrintLineSeparator();

        result = (mincost, mincostallocation);

        # Write result to a file.
        with open(instance_filepath_txt, "w") as f:
            f.write("Iteration, Minimum cost\n");
            for entry in history:
                f.write("{}, {}\n".format(entry[0], entry[1]));
            f.write("# Register configuration:\n");
            f.write(f"Cost: {result[0]}\n".format(result[0]));
            f.write(f"Allocation: ");
            for i in range(0, len(result[1])):
                f.write(" ".join([str(x) for x in result[1][i]]) + " ");
    else:
        # Read result from a file.
        with open(instance_filepath_txt, "r") as f:
            n = instance.n;
            p = len(instance.p);

            f.readline();
            for line in f:
                if (line[0] == '#'):
                    break;
                history.append([int(x) for x in line.split(',')]);
            mincost    = int(f.readline().split(':')[1]);
            allocation = np.array([int(x) for x in f.readline().split()[1:]]).reshape((p, n));
            
            result = (history[-1][1], allocation);

    print("Result:");
    print(result);
    PrintLineSeparator();

    #
    # Plot history.
    #
    plt.plot([x[0] for x in history], [x[1] for x in history]);
    plt.yticks(np.arange(0, max([x[1] for x in history])+1, 1));
    plt.locator_params(nbins=20);
    plt.xlim(left=0);
    plt.xlabel("Iteration");
    plt.ylabel("Minimum cost");
    plt.title("History (brute force)");
    plt.show();



#
# Main
#
# Main entry point.
#
def Main():
    bake = False;
    name = sys.argv[1];

    BruteForce(bake, name);

if __name__ == "__main__":
    Main();
