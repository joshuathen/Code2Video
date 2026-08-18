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
        self.setup_layout("Prerequisite Review: The Normal Distribution", 
                          ["The Normal Distribution is symmetric.", 
                           "It is often called the Bell Curve.", 
                           "Many phenomena follow this predictable pattern."])
        
        # Axes and Curve
        axes = Axes(x_range=[-4, 4, 1], y_range=[0, 1, 0.2], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="#FFFFFF")
        dist_group = VGroup(axes, curve)
        
        # Mean line and SD
        mean_line = DashedLine(axes.c2p(0, 0), axes.c2p(0, 0.4), color="#FFFF00")
        sd_line = Line(axes.c2p(0, 0.3), axes.c2p(1, 0.3), color="#00FF00")
        
        label = Text("Normal Dist", color="#FF0000", font_size=20)
        
        # Asset: Bell icon
        bell_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg")
        bell_icon.set_color(WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        self.place_in_area(dist_group, 'A3', 'C6', scale_factor=0.55)
        self.play(Create(dist_group))
        
        # Place bell icon near the curve
        self.place_at_grid(bell_icon, 'B3', scale_factor=0.5)
        self.play(FadeIn(bell_icon))
        
        self.place_at_grid(label, 'E4', scale_factor=0.8 * 0.7) # Applying B020 scale factor adjustment
        self.play(Write(label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#ADD8E6"))
        self.play(Create(mean_line))
        self.play(Create(sd_line))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#ADD8E6"))
        dots = VGroup(*[Dot(axes.c2p(np.random.normal(0, 0.8), 0.05), radius=0.03, color=WHITE) for _ in range(30)])
        self.play(FadeIn(dots))
        self.wait(2)
