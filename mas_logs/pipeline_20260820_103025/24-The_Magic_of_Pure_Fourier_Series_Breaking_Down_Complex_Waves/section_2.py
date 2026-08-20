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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Mathematical Prerequisites: Orthogonality", [
            "Waves are vectors in function space.",
            "Orthogonality means they don't overlap.",
            "Think of perpendicular geometric axes."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Visualize two perpendicular vectors in bright blue (#00BFFF) and yellow (#FFFF00)
        vec1 = Line(ORIGIN, RIGHT*1.5, color="#00BFFF", stroke_width=6).add_tip()
        vec2 = Line(ORIGIN, UP*1.5, color="#FFFF00", stroke_width=6).add_tip()
        axes_group = VGroup(vec1, vec2)
        
        # Update based on Feedback 31/29
        self.place_in_area(axes_group, "A4", "B6", scale_factor=0.5)
        self.play(Create(axes_group))
        self.lecture[0].set_color("#00BFFF")

        # === Animation for Lecture Line 2 ===
        # Show a sine wave (orange #FF8C00) and a cosine wave (purple #9370DB)
        ax = Axes(x_range=[0, 2*PI, 0.5], y_range=[-1.5, 1.5, 1], axis_config={"include_ticks": False}).scale(0.5)
        sin_wave = ax.plot(lambda x: np.sin(x), color="#FF8C00")
        cos_wave = ax.plot(lambda x: np.cos(x), color="#9370DB")
        waves = VGroup(ax, sin_wave, cos_wave)
        
        # Add Asset
        violin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/violin.svg")
        waves.add(violin)
        
        # Update based on Feedback 31/30
        self.place_in_area(waves, "D4", "F6", scale_factor=0.6)
        self.play(Create(ax), Create(sin_wave), Create(cos_wave), FadeIn(violin))
        self.lecture[1].set_color("#9370DB")

        # === Animation for Lecture Line 3 ===
        # Fade out overlapping parts, highlight 'cancellation'
        self.play(FadeOut(sin_wave), FadeOut(cos_wave), FadeOut(violin))
        rect = SurroundingRectangle(axes_group, color=WHITE, buff=0.2)
        self.play(Create(rect))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
