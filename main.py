import json

import networkx as nx
from matplotlib import pyplot as plt
from openai import OpenAI
import keyboard
import community

TREE_FILENAME = "word_tree.json"

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


def get_completion_for_words(word_a, word_b):
    completion = client.chat.completions.create(
        model="local-model",  # this field is currently unused
        messages=[
            {
                "role": "system",
                "content": "Tell a logic word that comes thinking on 2 given words for exemple:\n"
                           "Input: New York + Terrorism\n"
                           "Output: 9/11\n"
                           "Your response should be only one word but for exemple for the words like Nuclear Weapon,"
                           " you can split word"
            },
            {"role": "user", "content": "USA + City"},
            {"role": "assistant", "content": "New York"},
            {"role": "user", "content": f"{word_a} + {word_b}"},
        ],
        temperature=0.7,
    )

    return completion.choices[0].message.content


def get_completion_for_words_recursive(words_array, words_found):
    if words_found is None:
        words_found = {}

    new_words_array = words_array.copy()

    for i in range(len(words_array)):
        for j in range(i + 1, len(words_array)):
            word_a = words_array[i]
            word_b = words_array[j]

            if keyboard.is_pressed('s'):  # Check if 's' key is pressed
                print("Stopping...")
                return words_found

            if word_a not in words_found or (word_b not in words_found[word_a] and (
                    word_b not in words_found or word_a not in words_found[word_b])):
                result = get_completion_for_words(word_a, word_b)

                # If the result is the same as one of the inputs, do not add it to the tree
                if result is word_a or result is word_b:
                    print(f"Result is the same as one of the inputs: {result}")
                    continue

                # If the result is too large (more than 2 words), do not add it to the tree
                if len(result.split(" ")) > 2:
                    print(f"Result too large: {result}")
                    continue

                print(f"{word_a} + {word_b} = {result}")
                words_found[word_a] = words_found.get(word_a, {})
                words_found[word_a][word_b] = result
                words_found[word_b] = words_found.get(word_b, {})
                words_found[word_b][word_a] = result

                # Remove the used word from the words_array

                if word_a in new_words_array:
                    new_words_array.remove(word_a)

                if word_b in new_words_array:
                    new_words_array.remove(word_b)

                # Add the result to the words_array
                new_words_array.append(result)

    if len(new_words_array) < 4:
        # If there are less than 4 words in the array, add the words that are in words_array until there are 4 words
        new_words_array = new_words_array + [word for word in words_array if word not in new_words_array]

    words_found = get_completion_for_words_recursive(new_words_array, words_found)

    return words_found


def load_word_tree():
    try:
        with open(TREE_FILENAME, 'r') as file:
            word_tree = json.load(file)
        return word_tree
    except FileNotFoundError:
        print(f"File '{TREE_FILENAME}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the word tree: {e}")
        return None


def save_word_tree(word_tree):
    with open(TREE_FILENAME, 'w') as file:
        json.dump(word_tree, file)


def display_tree(tree):
    G = nx.Graph()
    for word_a in tree:
        for word_b in tree[word_a]:
            G.add_edge(word_a, tree[word_a][word_b], label=word_b)

    # Compute the partition using the Louvain method
    partition = community.best_partition(G)

    # Create a layout
    pos = nx.spring_layout(G, k=0.3, iterations=200, seed=42)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=list(partition.values()),
                           cmap=plt.colormaps.get_cmap('viridis'))
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    # Add labels to nodes
    nx.draw_networkx_labels(G, pos, font_size=5)

    plt.axis("off")
    plt.show()


def get_last_results(words_found, n=4):
    if words_found is None or len(words_found) == 0:
        return None

    return list(words_found.keys())[-n:]


def display_related_degree_graph(input_word, words_found):
    if input_word not in words_found:
        print(f"Word '{input_word}' not found in the word tree.")
        return

    G = nx.Graph()

    # Helper function to recursively add related words and their degrees
    def add_related_words(word, path=None, visited=None):
        if path is None:
            path = [input_word]
        if visited is None:
            visited = set()
        if word not in words_found:
            return
        visited.add(word)
        for related_word, result_word in words_found[word].items():
            if related_word not in path and related_word not in visited:
                new_path = path + [related_word]
                G.add_edge(path[-1], related_word)
                add_related_words(result_word, new_path, visited)

    # Add related words and their degrees
    add_related_words(input_word)

    # Calculate shortest path lengths using BFS
    shortest_path_lengths = nx.single_source_shortest_path_length(G, input_word)

    # Create a layout
    pos = nx.spring_layout(G, k=0.3, iterations=200, seed=42)

    # Assign a unique color to each degree level
    max_degree = max(shortest_path_lengths.values())
    cmap = plt.get_cmap('viridis', max_degree + 1)  # Create a colormap with the required number of colors
    node_colors = {node: cmap(distance) for node, distance in shortest_path_lengths.items()}

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=75, node_color=[node_colors[node] for node in G.nodes()])
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Create legend for degree levels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) for i in range(max_degree + 1)]
    labels = [f'Degree {i}' for i in range(max_degree + 1)]

    plt.legend(handles, labels, loc='upper right', title='Degree Level')

    plt.title(f"Related Words Degree Graph for '{input_word}'")
    plt.axis("off")
    plt.show()


def main():
    words_found = load_word_tree()
    last_result = get_last_results(words_found)

    base_words = ["Fire", "Wind", "Water", "Earth"]

    if last_result is not None:
        base_words = last_result

    # words_res = get_completion_for_words_recursive(base_words, words_found)
    display_tree(words_found)

    display_related_degree_graph("Fire", words_found)

    save_word_tree(words_found)


if __name__ == "__main__":
    main()
