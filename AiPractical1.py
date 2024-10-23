#Breadth first search

from collections import deque

def bfs_shortest_path(graph, start, goal):
    queue = deque([[start]])
    visited = set()
    
    while queue:
        path = queue.popleft()
        node = path[-1]
        
        if node == goal:
            return path
        
        if node not in visited:
            visited.add(node)
            
            for neighbor in graph.get(node, []):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return None

graph = { 'A': ['B', 'C'], 'B': ['A', 'D', 'E'], 'C': ['A', 'F'], 'D': ['B'], 'E': ['B', 'F'], 'F': ['C', 'E'] }

shortest_path = bfs_shortest_path(graph, 'A', 'F')
print("Shortest Path from 'A' to 'F':", shortest_path)

#output
#Shortest Path from 'A' to 'F': ['A', 'C', 'F']





#iterative Depth first Search

def iterative_dfs(graph, start, goal):
    stack = [[start]]
    
    visited = set()
    
    while stack:
        path = stack.pop()
        node = path[-1]
        
        if node == goal:
            return path
        
        if node not in visited:
            visited.add(node)
            
            for neighbor in reversed(graph.get(node, [])):
                new_path = list(path)
                new_path.append(neighbor)
                stack.append(new_path)
                
    return None

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}


# Find path from A to F
dfs_path = iterative_dfs(graph, 'A', 'F')
print("Path from 'A' to 'F' using IDFS:", dfs_path)


#output
#Path from 'A' to 'F' using IDFS: ['A', 'B', 'E', 'F']

