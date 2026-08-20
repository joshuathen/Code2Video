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
        self.setup_layout("Prerequisite Physics: Conservation Laws", ["Collisions conserve momentum.", "Collisions conserve kinetic energy.", "Mass ratios define velocities."])
        
        # Reveal lecture lines sequentially
        self.play(FadeIn(self.lecture[0]))
        
        # === Animation for Lecture Line 1 ===
        momentum_eq = MathTex(r"m_1v_1 + m_2v_2 = \text{const.}", color=BLUE)
        self.place_at_grid(momentum_eq, 'A2', scale_factor=0.8)
        self.play(Write(momentum_eq))
        self.lecture[0].set_color(BLUE)
        
        self.play(FadeIn(self.lecture[1]))
        
        # === Animation for Lecture Line 2 ===
        energy_eq = MathTex(r"\frac{1}{2}m_1v_1^2 + \frac{1}{2}m_2v_2^2 = \text{const.}", color=GREEN)
        self.place_at_grid(energy_eq, 'B2', scale_factor=0.7)
        self.play(Write(energy_eq))
        self.lecture[1].set_color(GREEN)
        
        self.play(FadeIn(self.lecture[2]))
        
        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg]
        ball1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg", color=RED)
        ball2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg", color=YELLOW)
        ball1_label = MathTex("m_1", font_size=24).next_to(ball1, UP, buff=0.1)
        ball2_label = MathTex("m_2", font_size=24).next_to(ball2, UP, buff=0.1)
        
        balls = VGroup(ball1, ball2, ball1_label, ball2_label)
        self.place_in_area(balls, 'D2', 'F5', scale_factor=0.6)
        
        self.play(Create(ball1), Create(ball2), Write(ball1_label), Write(ball2_label))
        self.lecture[2].set_color(YELLOW)
        
        # Animate momentum exchange
        self.play(ball1.animate.shift(RIGHT * 0.5), ball2.animate.shift(LEFT * 0.5), run_time=1)
        
        self.wait(2)
