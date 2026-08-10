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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Duality Relationship", [
            "Dual vertices equal original faces.",
            "Original vertices equal dual faces.",
            "Edges match in both graphs."
        ])
        
        # Load asset - although empty/none, following protocol
        # placeholder = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # Original graph
        orig_vertices = [Dot(color=YELLOW) for _ in range(4)]
        orig_edges = [Line(orig_vertices[0], orig_vertices[1], color=WHITE),
                      Line(orig_vertices[1], orig_vertices[2], color=WHITE),
                      Line(orig_vertices[2], orig_vertices[3], color=WHITE),
                      Line(orig_vertices[3], orig_vertices[0], color=WHITE),
                      Line(orig_vertices[0], orig_vertices[2], color=WHITE)]
        orig_graph = VGroup(*orig_edges, *orig_vertices)
        
        # Dual graph
        dual_vertices = [Dot(color="#3357FF") for _ in range(3)]
        dual_edges = [Line(dual_vertices[0], dual_vertices[1], color="#FF5733"),
                      Line(dual_vertices[1], dual_vertices[2], color="#FF5733")]
        dual_graph = VGroup(*dual_edges, *dual_vertices)
        
        # Layouts - updated based on crit
        self.place_in_area(orig_graph, 'B1', 'E3', scale_factor=0.8)
        self.place_in_area(dual_graph, 'B4', 'E6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(orig_graph), Create(dual_graph))
        self.lecture[0].set_color("#3357FF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(Flash(orig_graph[1:5], color=YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF5733")
        highlight_edges = VGroup(orig_edges[0], dual_edges[0])
        self.play(highlight_edges.animate.set_color("#FF5733"))
        self.wait(1)
