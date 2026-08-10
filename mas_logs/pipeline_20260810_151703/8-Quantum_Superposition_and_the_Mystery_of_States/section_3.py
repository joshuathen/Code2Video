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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Measurement Problem and Collapse", [
            "Measurement forces the wave function to collapse.",
            "The probability depends on the squared amplitudes.",
            "The vector snaps to a basis pole."
        ])
        
        # Assets
        eye = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg", color=WHITE)
        sensor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg", color=WHITE)

        # Define objects
        wave_vector = Arrow(start=ORIGIN, end=UP*1.5+RIGHT*1.5, color=BLUE)
        basis_0 = Arrow(start=ORIGIN, end=UP*2, color=GREEN, buff=0)
        basis_1 = Arrow(start=ORIGIN, end=RIGHT*2, color=RED, buff=0)
        
        # Labels
        labels = VGroup(
            MathTex(r"|\psi\rangle").next_to(wave_vector.get_end(), UP+RIGHT),
            MathTex(r"|0\rangle").next_to(basis_0.get_end(), UP),
            MathTex(r"|1\rangle").next_to(basis_1.get_end(), RIGHT)
        )
        
        # Place assets and vectors
        self.place_at_grid(eye, 'B5', scale_factor=0.8)
        self.place_at_grid(sensor, 'E5', scale_factor=0.8)
        self.place_in_area(VGroup(wave_vector, basis_0, basis_1, labels), 'C4', 'E5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(wave_vector), FadeIn(basis_0), FadeIn(basis_1), FadeIn(labels), FadeIn(eye))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        prob_rect = Rectangle(height=0.5, width=1.5, color=YELLOW, fill_opacity=0.3)
        self.place_at_grid(prob_rect, 'D2', scale_factor=0.8)
        
        self.play(Create(prob_rect), FadeIn(sensor))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)
        
        # collapse_animation represented by rotating wave_vector to basis_0
        self.play(
            Rotate(wave_vector, angle=-PI/4, about_point=ORIGIN),
            FadeOut(prob_rect)
        )
        self.wait(2)
