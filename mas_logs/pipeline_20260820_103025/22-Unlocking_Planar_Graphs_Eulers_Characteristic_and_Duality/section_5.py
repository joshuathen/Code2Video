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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "VLSI circuits use planarity to prevent shorts.",
            "Topology maps simplify complex pathfinding tasks.",
            "Euler's formula remains a powerful geometric tool."
        ]
        self.setup_layout("Real-world Application", lecture_lines)
        
        # Colors for lines
        colors = ["#00FF00", "#FFFF00", "#FFFFFF"]

        # === Animation for Lecture Line 1 ===
        # Represent a simplified circuit as a planar graph
        circuit = VGroup()
        nodes = VGroup(*[Dot(color=BLUE) for _ in range(4)])
        # Manually create lines in VGroup to allow scaling
        lines = VGroup(*[Line(np.array([0, 0, 0]), np.array([1, 1, 0]), color="#00FF00") for _ in range(4)])
        circuit.add(nodes, lines)
        self.place_in_area(circuit, 'B2', 'D3', scale_factor=1.2)
        self.add(circuit)
        self.lecture[0].set_color(colors[0])
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Show duality/pathing
        dual_nodes = VGroup(*[Dot(color=RED) for _ in range(2)])
        self.place_at_grid(dual_nodes[0], 'C2', scale_factor=0.8)
        self.place_at_grid(dual_nodes[1], 'D2', scale_factor=0.8)
        dual_line = Line(dual_nodes[0].get_center(), dual_nodes[1].get_center(), color="#FFFF00")
        self.add(dual_nodes, dual_line)
        self.lecture[1].set_color(colors[1])
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Euler's formula representation
        formula = MathTex(r"V - E + F = 2", color="#FFFFFF")
        self.place_at_grid(formula, 'B5', scale_factor=1.0)
        self.add(formula)
        self.lecture[2].set_color(colors[2])
        self.wait(2)
