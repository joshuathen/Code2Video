from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: Defining the Planar Graph", [
            "Planar graphs avoid edge crossings.", 
            "Vertices, edges, and faces define structure.", 
            "Faces include the outer infinite region."
        ])
        
        # Load assets
        vertex_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/vertex.svg"
        edge_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/edge.svg"
        
        # Create a simple planar graph: square
        nodes = VGroup(*[SVGMobject(vertex_svg, color=WHITE) for _ in range(4)])
        edges = VGroup(
            SVGMobject(edge_svg, color=WHITE),
            SVGMobject(edge_svg, color=WHITE),
            SVGMobject(edge_svg, color=WHITE),
            SVGMobject(edge_svg, color=WHITE)
        )
        
        # Position nodes
        positions = ["B2", "B4", "D4", "D2"]
        for i, pos in enumerate(positions):
            self.place_at_grid(nodes[i], pos, scale_factor=0.5)
            
        # Position edges (simplified as lines for geometric connections)
        # Using placeholder lines to connect the SVGs properly
        graph_edges = VGroup(
            Line(nodes[0].get_center(), nodes[1].get_center(), color=WHITE),
            Line(nodes[1].get_center(), nodes[2].get_center(), color=WHITE),
            Line(nodes[2].get_center(), nodes[3].get_center(), color=WHITE),
            Line(nodes[3].get_center(), nodes[0].get_center(), color=WHITE)
        )
        
        graph = VGroup(graph_edges, nodes)
        self.place_in_area(graph, 'A3', 'F6', scale_factor=1.0)
        
        # Adding labels
        node_labels = Text("V, E, F", font_size=20, color=WHITE)
        self.place_at_grid(node_labels, 'C3', scale_factor=0.9)
        self.add(node_labels)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(graph))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(
            nodes.animate.set_color("#FF5733"),
            run_time=2
        )
        self.play(
            nodes.animate.set_color(WHITE),
            run_time=2
        )

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.play(
            graph_edges.animate.set_color("#33FF57"),
            run_time=2
        )
        self.play(
            graph_edges.animate.set_color(WHITE),
            run_time=2
        )
