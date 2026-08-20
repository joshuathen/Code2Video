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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Euler’s Characteristic Formula", [
            "Euler's formula states V - E + F = 2.",
            "This holds for any connected planar graph.",
            "The exterior region counts as one face.",
            "Example: A triangle has V=3, E=3, F=2.",
            "Formula result: 3 - 3 + 2 = 2."
        ])
        
        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"V - E + F = 2", color=WHITE)
        self.place_at_grid(formula, "B4", scale_factor=1.2)
        formula_label = Text("Euler's Formula", font_size=20)
        formula_label.next_to(formula, UP)
        
        # Load asset
        triangle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg")
        self.place_at_grid(triangle, "A1", scale_factor=0.3)
        
        self.play(Write(formula), Write(formula_label), FadeIn(triangle))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        graph = Graph(
            [1, 2, 3, 4],
            [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
            layout="circular"
        )
        graph.set_color("#FF6600")
        self.place_at_grid(graph, "D3", scale_factor=0.6)
        self.play(FadeIn(graph))
        self.lecture[1].set_color("#FF6600")

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        v_label = Text("V=4", color="#00FF00", font_size=24)
        e_label = Text("E=6", color="#FF6600", font_size=24)
        f_label = Text("F=4", color="#0000FF", font_size=24)
        
        v_label.next_to(graph, UP, buff=0.1)
        e_label.next_to(graph, RIGHT, buff=0.1)
        f_label.next_to(graph, DOWN, buff=0.1)
        
        self.play(Write(v_label), Write(e_label), Write(f_label))
        self.lecture[3].set_color("#00FF00")

        # === Animation for Lecture Line 5 ===
        res = MathTex(r"4 - 6 + 4 = 2", color=WHITE)
        self.place_at_grid(res, "F5", scale_factor=0.9)
        self.play(FadeIn(res))
        self.lecture[4].set_color("#FFFFFF")
        
        # Second asset usage
        triangle_ref = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg")
        self.place_at_grid(triangle_ref, "F2", scale_factor=0.2)
        self.play(FadeIn(triangle_ref))
        
        self.wait(2)
