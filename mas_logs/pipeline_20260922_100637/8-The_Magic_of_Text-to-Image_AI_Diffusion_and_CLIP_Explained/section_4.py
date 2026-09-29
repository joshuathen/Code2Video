from manim import *
import numpy as np

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
        lecture_lines = [
            "Diffusion adds noise to destroy original images.",
            "The forward process maps data to chaos.",
            "Reverse diffusion learns to undo this noise.",
            "We step-by-step refine noise into clarity.",
            "This math recovers structure from randomness."
        ]
        self.setup_layout("The Artist: The Diffusion Process", lecture_lines)
        
        # Assets
        photo = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg")
        noise = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/noise.svg")
        
        # Initial placement
        self.place_in_area(photo, "B2", "E5", scale_factor=0.6)
        
        # Noise dots for simulation
        noise_dots = VGroup(*[Dot(point=np.random.normal(0, 0.5, 3), color=WHITE, radius=0.03) for _ in range(200)])

        # Math formula
        formula = MathTex(r"x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon", font_size=32)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF33A1"))
        self.add(photo)
        self.play(Transform(photo, noise))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        # Using noise icon here
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#33A1FF"))
        self.play(Transform(noise, photo))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF33"))
        self.place_in_area(noise_dots, "B1", "E3", scale_factor=0.6)
        self.play(Create(noise_dots))
        self.play(FadeOut(noise_dots))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF8833"))
        self.place_in_area(formula, "C4", "D6", scale_factor=1.0)
        self.play(Write(formula))
        self.add(photo)
        self.wait(2)
