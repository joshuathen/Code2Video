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
        lecture_lines = ["Boundary conditions provide system constraints.", "Guitar strings fixed at both ends.", "Free-hanging ropes move without anchors."]
        self.setup_layout("Boundary Conditions: The 'Rules' of the System", lecture_lines)
        
        # Elements
        rope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rope.svg")
        string_line = Line(start=self.grid["C3"], end=self.grid["C4"], color=WHITE)
        dot_left = Dot(string_line.get_start(), color=WHITE)
        dot_right = Dot(string_line.get_end(), color=WHITE)
        domain = VGroup(string_line, dot_left, dot_right)
        
        # Use place_in_area as requested
        self.place_in_area(string_line, 'C3', 'C4', scale_factor=1.0)
        
        label_left = MathTex("u(0, t) = 0", font_size=30, color=BLUE)
        label_right = MathTex("u(L, t) = 0", font_size=30, color=BLUE)
        formulas = VGroup(label_left, label_right).arrange(RIGHT, buff=1.0)
        self.place_in_area(formulas, 'B3', 'B4', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(rope))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(domain), Write(formulas))
        self.play(
            dot_left.animate.set_color(RED),
            dot_right.animate.set_color(RED),
            run_time=0.5
        )
        self.play(
            dot_left.animate.set_color(WHITE),
            dot_right.animate.set_color(WHITE),
            run_time=0.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
