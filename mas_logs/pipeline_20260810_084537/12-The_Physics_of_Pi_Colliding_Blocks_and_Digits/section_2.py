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
        lecture_lines = ["Elastic collisions conserve momentum and energy.", "Energy defines the system's total speed.", "[Asset: energy_conservation_vector_field]"]
        self.setup_layout("Prerequisite Physics: Conservation Laws", lecture_lines)
        
        # Setup objects
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg]
        momentum_eq = MathTex(r"p_1 + p_2 = p_1' + p_2'", color=WHITE)
        block_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg", color=GREEN, fill_opacity=0.5)
        block_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg", color=BLUE, fill_opacity=0.5)
        arrow_a = Arrow(start=ORIGIN, end=RIGHT*1.5, color=GREEN)
        arrow_b = Arrow(start=ORIGIN, end=LEFT*1.5, color=BLUE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(arrow_a, 'C2')
        self.place_at_grid(arrow_b, 'C6')
        self.place_at_grid(block_a, 'C3')
        self.place_at_grid(block_b, 'C5')
        self.play(FadeIn(VGroup(block_a, block_b, arrow_a, arrow_b)))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.place_in_area(momentum_eq, 'A2', 'A5', scale_factor=0.9)
        self.play(Write(momentum_eq))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Using simple primitives as placeholder for [Asset: energy_conservation_vector_field]
        vector_field = VGroup(*[Line(start=ORIGIN, end=UP*0.5, color=RED).shift(i*0.5*RIGHT) for i in range(5)])
        self.place_in_area(vector_field, 'D2', 'E5', scale_factor=0.7)
        
        # Animate interaction
        self.play(
            block_a.animate.shift(RIGHT*1.6),
            block_b.animate.shift(LEFT*1.6),
            arrow_a.animate.shift(RIGHT*1.6),
            arrow_b.animate.shift(LEFT*1.6),
            FadeIn(vector_field)
        )
        self.wait(1)
