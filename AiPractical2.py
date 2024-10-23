
#implement the A* Search algorithm

import heapq

romania_map = {
    'Arad': {'Zerind': 75, 'Timisoara': 118, 'Sibiu': 140},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Pitesti': 97, 'Craiova': 146},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

class Node:
    def __init__(self, city, cost, parent=None):
        self.city = city
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost

def astar_search(graph, start, goal):
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, start)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.city == goal.city:
            return construct_path(current_node)

        closed_set.add(current_node.city)

        for neighbor, distance in graph[current_node.city].items():
            if neighbor not in closed_set:
                new_node = Node(neighbor, current_node.cost + distance, current_node)
                heapq.heappush(open_list, new_node)

    return None 

def construct_path(node):
    path = []
    while node:
        path.append(node.city)
        node = node.parent
    return path[::-1]  

start_node = Node('Arad', 0)
goal_node = Node('Bucharest', 0)
path = astar_search(romania_map, start_node, goal_node)

if path:
    print("Path found:", path)
else:
    print("No path found")
    
#Output
#Path found: ['Arad', 'Sibiu', 'Rimnicu Vilcea', 'Pitesti', 'Bucharest']


#2
graph = {
    'A': {'B': 1, 'C': 3}, 
    'B': {'A': 1, 'D': 1, 'E': 5}, 
    'C': {'A': 3, 'F': 4},
    'D': {'B': 1, 'E': 2, 'G': 7}, 
    'E': {'B': 5, 'D': 2, 'G': 2}, 
    'F': {'C': 4, 'G': 1},
    'G': {'D': 7, 'E': 2, 'F': 1}
}

heuristic = {'A': 7, 'B': 6, 'C': 2, 'D': 3, 'E': 1, 'F': 1, 'G': 0}

# RBFS Algorithm
def rbfs(graph, node, goal, heuristic, path, cost_so_far, f_limit):
    if node == goal:
        return path, cost_so_far

    neighbors = [(cost_so_far + graph[node][neighbor] + heuristic[neighbor], neighbor) 
                 for neighbor in graph.get(node, {})]
    if not neighbors:
        return None, float('inf')

    while neighbors:
        neighbors.sort()
        best_f, best_node = neighbors[0]
        
        if best_f > f_limit:
            return None, best_f
        
        alternative = neighbors[1][0] if len(neighbors) > 1 else float('inf')
        result, best_f = rbfs(graph, best_node, goal, heuristic, path + [best_node], 
                              cost_so_far + graph[node][best_node], min(f_limit, alternative))
        if result:
            return result, best_f
        
        neighbors[0] = (best_f, best_node)
    return None, float('inf')

def recursive_best_first_search(graph, start, goal, heuristic):
    return rbfs(graph, start, goal, heuristic, [start], 0, float('inf'))

# Execute RBFS
path, cost = recursive_best_first_search(graph, 'A', 'G', heuristic)
print(f"RBFS Path: {path}, Cost: {cost}")

#output
#RBFS Path: ['A', 'B', 'D', 'E', 'G'], Cost: 6


    
