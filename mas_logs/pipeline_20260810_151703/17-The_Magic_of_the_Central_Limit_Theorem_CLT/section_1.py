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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Hook: The Chaos of Individuals", [
            "Individual data points are often chaotic and unpredictable.",
            "Consider a uniform distribution of squirrel jumps.",
            "There is no clear pattern in the noise."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Create scattered points and squirrel icons
        np.random.seed(42)
        squirrel_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg")
        
        # Create a group of dots and a few squirrel icons
        points = VGroup()
        for _ in range(50):
            dot = Dot(color="#FF9900", radius=0.05)
            dot.move_to(np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]))
            points.add(dot)
            
        icons = VGroup()
        for _ in range(5):
            icon = squirrel_icon.copy()
            icon.set_color("#FF9900")
            icon.move_to(np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]))
            icons.add(icon)
        
        group = VGroup(points, icons)
        self.place_in_area(group, "C2", "F5", scale_factor=0.8)
        
        self.play(FadeIn(group))
        self.lecture[0].set_color("#FF9900")

        # === Animation for Lecture Line 2 ===
        label = Text("Uniform Distribution: Individual Squirrels", font_size=20, color=WHITE)
        self.place_at_grid(label, "B2", scale_factor=0.6)
        self.play(Write(label))
        self.lecture[1].set_color("#FF9900")

        # === Animation for Lecture Line 3 ===
        self.play(Indicate(group, color="#FF9900"))
        self.lecture[2].set_color("#FF9900")
        self.wait(2)
