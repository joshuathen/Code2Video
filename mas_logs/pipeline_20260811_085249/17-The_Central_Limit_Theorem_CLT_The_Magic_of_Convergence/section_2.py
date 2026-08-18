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
        self.setup_layout("The Problem: Randomness in the Wild", [
            "Real-world data is often not normally distributed.",
            "Many distributions are skewed, uniform, or discrete.",
            "We need to predict outcomes despite this randomness."
        ])
        
        # Animations
        # Compass icon
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, 'A6', scale_factor=0.3)
        
        # Ruler icon
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        self.place_at_grid(ruler, 'F6', scale_factor=0.3)

        # Create chaotic scatter plot - fix 23: D1-F6
        dots = VGroup(*[Dot(point=np.random.uniform(-1, 1, 3) * 1.5, radius=0.04) for _ in range(50)])
        self.place_in_area(dots, 'D1', 'F6', scale_factor=0.6)
        
        # Trend line
        line = Line(start=self.grid['F1'], end=self.grid['A6'], color=WHITE)
        
        # Error bars
        error_bars = VGroup()
        for dot in dots:
            bar = Line(dot.get_center() + UP*0.1, dot.get_center() + DOWN*0.1, stroke_width=1)
            error_bars.add(bar)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(compass), FadeIn(dots))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(Create(line))
        line.set_color("#33FFF5")
        self.lecture[1].set_color("#33FFF5")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(error_bars), FadeIn(ruler))
        error_bars.set_color("#F533FF")
        self.lecture[2].set_color("#F533FF")
        
        self.wait(2)
