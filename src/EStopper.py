#!/usr/bin/env python3

class EarlyStopper:
    def __init__(self, patience=1, min_change=0, mode='min'):
        self.patience = patience
        self.min_change = min_change
        self.mode = mode
        self.counter = 0
        self.min_variable = float('inf')
        self.max_variable = float('-inf')

    def stop_early(self, variable):
        if self.mode == 'min':
            if variable < self.min_variable:
                self.min_variable = variable
                self.counter = 0
            elif (variable - self.min_variable) > self.min_change:
                self.counter += 1
                if self.counter > self.patience:
                    return True

        elif self.mode == 'max':
            if variable > self.max_variable:
                self.max_variable = variable
                self.counter = 0
            elif self.max_variable > (self.min_change + variable):
                self.counter += 1
                if self.counter > self.patience:
                    return True

        return False