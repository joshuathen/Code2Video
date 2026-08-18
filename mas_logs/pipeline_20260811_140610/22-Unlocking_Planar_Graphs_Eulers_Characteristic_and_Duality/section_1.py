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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisites & Definitions", [
            "A planar graph embeds without crossing edges.",
            "Faces include every enclosed region.",
            "The exterior region counts as a face."
        ])
        
        # Load Polyhedron Asset
        poly = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/polyhedron.svg", color=WHITE)
        self.place_in_area(poly, 'A2', 'C5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(poly))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Representing faces/structure visually
        nodes = [[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]]
        v = VGroup(*[Dot(n, color="#FF0000") for n in nodes])
        e = VGroup(Line(nodes[0], nodes[1], color="#00FF00"), Line(nodes[1], nodes[2], color="#00FF00"), Line(nodes[2], nodes[3], color="#00FF00"), Line(nodes[3], nodes[0], color="#00FF00"))
        face = Polygon(*nodes, fill_opacity=0.3, color="#0000FF")
        graph = VGroup(face, e, v)
        self.place_in_area(graph, 'D2', 'E5', scale_factor=0.6)
        self.play(FadeIn(graph))
        self.lecture[1].set_color("#0000FF")

        # === Animation for Lecture Line 3 ===
        summary = Text("Exterior counts as face", font_size=20, color=WHITE)
        self.place_at_grid(summary, 'D4', scale_factor=0.9)
        self.play(Write(summary))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
