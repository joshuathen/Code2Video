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
        lecture_lines = ["We need a more robust formula.", "The number of regions follows a polynomial.", "It is based on combinations of points."]
        self.setup_layout("The Structural Solution: Euler’s Characteristic", lecture_lines)
        
        # Euler formula components
        euler_formula = MathTex("V", "-", "E", "+", "F", "=", "2", font_size=48)
        euler_formula.set_color_by_tex("V", BLUE)
        euler_formula.set_color_by_tex("E", GREEN)
        euler_formula.set_color_by_tex("F", RED)

        # Labels
        v_label = Text("V = Vertices", color=BLUE, font_size=20)
        e_label = Text("E = Edges", color=GREEN, font_size=20)
        f_label = Text("F = Faces (Regions)", color=RED, font_size=20)
        labels = VGroup(v_label, e_label, f_label).arrange(DOWN, aligned_edge=LEFT)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Fixed per issue 27
        self.place_in_area(euler_formula, 'A2', 'B5', scale_factor=1.0)
        self.play(Write(euler_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        # Fixed per issue 28
        self.place_at_grid(labels, 'C3', scale_factor=0.9)
        self.play(FadeIn(labels))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Derived formula R = (n choose 4) + (n choose 2) + 1
        formula_r = MathTex("R = \\binom{n}{4} + \\binom{n}{2} + 1", font_size=40)
        # Fixed per issue 29
        self.place_in_area(formula_r, 'D2', 'E5', scale_factor=0.9)
        self.play(Write(formula_r))
        self.wait(2)
