# coding: utf-8

"""
This code is part of the course 'Innovative Programmiermethoden für Industrieroboter' (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import networkx as nx
import matplotlib.pyplot as plt


def aStarVisualize(planner, solution, ax = None, nodeSize = 300):
    graph = planner.graph
    collChecker = planner._collisionChecker
    # get a list of positions of all nodes by returning the content of the attribute 'pos'
    pos = nx.get_node_attributes(graph,'pos')
    color = nx.get_node_attributes(graph,'color')
    
    # get a list of degrees of all nodes
    #degree = nx.degree_centrality(graph)
    
    # draw graph (nodes colorized by degree)
    open_nodes = [node for node,attribute in graph.nodes(data=True) if attribute['status']=="open"]
    draw_nodes = nx.draw_networkx_nodes(graph, pos, node_color='#FFFFFF', nodelist=open_nodes, ax = ax, node_size=nodeSize)
    draw_nodes.set_edgecolor("b")
    open_nodes = [node for node,attribute in graph.nodes(data=True) if attribute['status']=="closed"]
    draw_nodes = nx.draw_networkx_nodes(graph, pos, node_color='#0000FF', nodelist=open_nodes, ax = ax, node_size=nodeSize)
    #nx.draw_networkx_nodes(graph, pos,  cmap=plt.cm.Blues, ax = ax, node_size=nodeSize)
    nx.draw_networkx_edges(graph,pos,
                               edge_color='b',
                               width=3.0
                            )
    
    collChecker.drawObstacles(ax)
    
    # draw nodes based on solution path
    Gsp = nx.subgraph(graph,solution)
    nx.draw_networkx_nodes(Gsp,pos,
                            node_size=nodeSize,
                             node_color='g')
        
    # draw edges based on solution path
    nx.draw_networkx_edges(Gsp,pos,alpha=0.8,edge_color='g',width=10,arrows=True)
 
    nx.draw_networkx_nodes(graph,pos,nodelist=[solution[0]],
                           node_size=300,
                           node_color='#00dd00',  ax = ax)
    nx.draw_networkx_labels(graph,pos,labels={solution[0]: "S"},  ax = ax)


    nx.draw_networkx_nodes(graph,pos,nodelist=[solution[-1]],
                                   node_size=300,
                                   node_color='#DD0000',  ax = ax)
    nx.draw_networkx_labels(graph,pos,labels={solution[-1]: "G"},  ax = ax)


"""def aStarVisualizeWspace(planner, solution, ax = None, nodeSize = 100):
    #Draw graph, obstacles and solution in a axis environment of matplotib.
    
    # get a list of positions of all nodes by returning the content of the attribute 'pos'
    graph = planner.graph
    collChecker = planner._collisionChecker

    collChecker.drawObstacles(ax)
    
    
    pos = nx.get_node_attributes(graph,'pos')
    # todo extract from pos the first two dimensions only for drawing in workspace
    pos2D = dict()
    for key in pos.keys():
        pos2D[key] = (pos[key][0], pos[key][1])
        
    pos = pos2D
    
    # draw graph (nodes colorized by degree)
    nx.draw_networkx_nodes(graph, pos, ax = ax, node_size=nodeSize)
    nx.draw_networkx_edges(graph,pos,
                                ax = ax
                                 )
    Gcc = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    G0=graph.subgraph(Gcc[0])# = largest connected component

    # how largest connected component
    nx.draw_networkx_edges(G0,pos,
                               edge_color='b',
                               width=3.0, ax = ax
                            )

    
    # draw nodes based on solution path
    Gsp = nx.subgraph(graph,solution)
    nx.draw_networkx_nodes(Gsp,pos,
                            node_size=nodeSize*1.5,
                             node_color='g',  ax = ax)
        
    # draw edges based on solution path
    nx.draw_networkx_edges(Gsp,pos,alpha=0.8,edge_color='g',width=10,  ax = ax)
        
    # draw start and goal
    if "start" in graph.nodes(): 
        nx.draw_networkx_nodes(graph,pos,nodelist=["start"],
                                   node_size=nodeSize*1.5,
                                   node_color='#00dd00',  ax = ax)
        nx.draw_networkx_labels(graph,pos,labels={"start": "S"},  ax = ax)


    if "goal" in graph.nodes():
        nx.draw_networkx_nodes(graph,pos,nodelist=["goal"],
                                   node_size=nodeSize*1.5,
                                   node_color='#DD0000',  ax = ax)
        nx.draw_networkx_labels(graph,pos,labels={"goal": "G"},  ax = ax)"""

def aStarVisualizeWspace(planner, solution, ax = None, nodeSize = 100):
    graph = planner.graph
    collChecker = planner._collisionChecker
    robot = collChecker.robot # On accède à l'objet ShapeRobot

    collChecker.drawObstacles(ax) # Dessine les obstacles et le robot une fois
    
    pos = nx.get_node_attributes(graph,'pos')
    pos2D = {key: (val[0], val[1]) for key in pos.keys()} # Projection 2D pour NetworkX
    
    # Dessin du graphe (Correction pour DiGraph)
    Gcc = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    G0 = graph.subgraph(Gcc[0])
    nx.draw_networkx_edges(G0, pos2D, edge_color='b', width=1.0, ax=ax, alpha=0.1)

    # DESSIN DES EMPREINTES DU ROBOT RECTANGLE
    if solution:
        for i, node_id in enumerate(solution):
            # On récupère la pose complète (x, y, theta) du nœud
            current_pose = graph.nodes[node_id]['pos']
            robot.setTo(current_pose) # Positionne et FAIT TOURNER le rectangle
            
            # On dessine le rectangle avec une transparence (alpha)
            # On n'en dessine qu'un sur trois pour ne pas surcharger le graphique
            if i % 3 == 0 or i == len(solution)-1:
                robot.drawRobot(ax, color='green', alpha=0.2) 

        # Dessin de la ligne de solution
        Gsp = nx.subgraph(graph, solution)
        nx.draw_networkx_edges(Gsp, pos2D, edge_color='g', width=5, ax=ax)