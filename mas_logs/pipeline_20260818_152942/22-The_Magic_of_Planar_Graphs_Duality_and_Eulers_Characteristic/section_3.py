from manim import *

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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The dual graph G* transforms graph properties.",
            "Place a node in each graph face.",
            "Connect nodes sharing an edge boundary.",
            "Honeycomb duals form a triangular lattice.",
            "Duality shifts perspective to graph faces."
        ]
        self.setup_layout("The Concept of Duality", lecture_lines)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/honeycomb.svg]
        # Load asset
        honeycomb_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/honeycomb.svg")
        
        # === Animation for Lecture Line 1 ===
        # Draw a graph G with vertices and edges in #FF6600
        graph_g = honeycomb_svg.copy().set_color("#FF6600")
        self.place_in_area(graph_g, 'B2', 'E5', scale_factor=0.9)
        self.play(FadeIn(graph_g))
        self.lecture[0].set_color("#FF6600")

        # === Animation for Lecture Line 2 ===
        # Place a dot in each face of G to represent dual vertices, colored #00FFFF
        # Centroid dot label
        dual_node = Dot(color="#00FFFF")
        self.place_at_grid(dual_node, 'C3')
        
        label = Text("Dual Vertex", font_size=20, color="#00FFFF")
        self.place_at_grid(label, 'C4', scale_factor=0.6)
        
        self.play(FadeIn(dual_node), FadeIn(label))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Connect dual vertices across edges of G to form dual graph G*. 
        # Color edges of G* in #FFFF00
        dual_line1 = Line(self.grid["C3"], self.grid["B3"], color="#FFFF00")
        dual_line2 = Line(self.grid["C3"], self.grid["D3"], color="#FFFF00")
        self.play(Create(dual_line1), Create(dual_line2))
        self.lecture[2].set_color("#FFFF00")

        # === Animation for Lecture Line 4 ===
        # Honeycomb duals form a triangular lattice.
        # Place at E3 as suggested
        honeycomb_dual = honeycomb_svg.copy().set_color("#FFFF00")
        self.place_at_grid(honeycomb_dual, 'E3', scale_factor=0.7)
        self.play(FadeIn(honeycomb_dual))
        self.lecture[3].set_color("#FF6600")

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.wait(2)
