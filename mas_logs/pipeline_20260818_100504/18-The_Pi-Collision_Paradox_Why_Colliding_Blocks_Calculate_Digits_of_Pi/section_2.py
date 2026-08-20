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
        self.setup_layout("Prerequisite: Elastic Collisions and Conservation", [
            "Elastic collisions conserve momentum and kinetic energy.",
            "Equal masses simply exchange velocities upon impact.",
            "Treat them as passing through each other."
        ])
        
        # Elements
        momentum_formula = MathTex(r"p = mv", font_size=36, color=WHITE)
        kinetic_energy_formula = MathTex(r"K = \frac{1}{2}mv^2", font_size=36, color="#FF33FF")
        
        # Applying requested layout changes
        self.place_in_area(momentum_formula, 'A3', 'B5', scale_factor=0.9)
        self.place_in_area(kinetic_energy_formula, 'C3', 'D5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Write(momentum_formula))
        self.play(momentum_formula.animate.set_color("#33CCFF"), Indicate(momentum_formula, scale_factor=1.2))
        self.play(Write(kinetic_energy_formula))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Collision shapes
        c1 = Circle(radius=0.3, color=BLUE, fill_opacity=0.8)
        c2 = Circle(radius=0.3, color=RED, fill_opacity=0.8)
        v1 = Arrow(start=ORIGIN, end=RIGHT*0.5, color=BLUE).next_to(c1, UP, buff=0.1)
        v2 = Arrow(start=ORIGIN, end=LEFT*0.5, color=RED).next_to(c2, UP, buff=0.1)
        collision_shapes = VGroup(c1, c2, v1, v2)
        
        # Position them
        self.place_in_area(collision_shapes, 'E2', 'F5', scale_factor=0.8)
        
        self.play(Create(c1), Create(c2), Create(v1), Create(v2))
        self.play(c1.animate.shift(RIGHT*1.2), c2.animate.shift(LEFT*1.2), 
                  v1.animate.shift(RIGHT*1.2), v2.animate.shift(LEFT*1.2))
        self.play(FadeOut(collision_shapes))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.wait(1)
