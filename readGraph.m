function [N1, N2, E1, E2] = readGraph()
    N1 = dlmread('nodes1.txt');
    N2 = dlmread('nodes2.txt')'
    E1 = dlmread('edges1.txt');
    E2 = dlmread('edges2.txt');
end