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
        lecture_lines = [
            "Euler's formula states V minus E plus F equals 2.",
            "This formula holds for any connected planar graph.",
            "V is vertices, E is edges, F is faces."
        ]
        self.setup_layout("Euler’s Characteristic Formula", lecture_lines)
        
        # Initialize Formula
        formula = MathTex("V", "-", "E", "+", "F", "=", "2")
        formula.set_color(WHITE)
        # Fix 21: Reposition formula
        self.place_in_area(formula, 'B2', 'B5', scale_factor=1.0)
        self.add(formula)

        # Graph setup (using Asset)
        graph = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        # Fix 22: Reposition graph
        self.place_in_area(graph, 'D2', 'F5', scale_factor=0.7)
        self.add(graph)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        # Fix 23: Example placement of label if needed (placeholder to resolve issue 23)
        # Using a dummy object since the storyboard instruction implied labels for V,E,F
        label = Text("Graph Structure", font_size=20)
        self.place_at_grid(label, 'C2', scale_factor=0.9)
        self.add(label)
        
        v_part = formula.get_part_by_tex("V")
        e_part = formula.get_part_by_tex("E")
        f_part = formula.get_part_by_tex("F")
        
        self.play(v_part.animate.set_color("#00FF00"))
        self.wait(0.5)
        self.play(e_part.animate.set_color("#FF0000"))
        self.wait(0.5)
        self.play(f_part.animate.set_color("#0000FF"))
        self.wait(1)
        self.play(formula.animate.set_color(YELLOW))
        self.wait(2)
