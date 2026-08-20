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
        self.setup_layout("Prerequisite Review: The Orthogonality of Sine/Cosine", 
                          ["Orthogonality means mathematical independence of functions.", 
                           "Sine and cosine waves are mutually independent.", 
                           "This independence allows us to extract signal components."])
        
        # Assets
        sine_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg")
        cosine_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cosine.svg")
        
        # Animations
        # === Animation for Lecture Line 1 ===
        axes = Axes(x_range=[0, 2*PI, PI/2], y_range=[-1.5, 1.5], x_length=4, y_length=2.5)
        sin1 = axes.plot(lambda x: np.sin(x), color="#FF00FF")
        sin2 = axes.plot(lambda x: np.sin(2*x), color="#FF00FF")
        plot_group = VGroup(axes, sin1, sin2, sine_icon)
        self.place_in_area(plot_group, 'C3', 'E6', scale_factor=0.75)
        
        self.play(Create(axes), Create(sin1), Create(sin2), FadeIn(sine_icon))
        self.lecture[0].set_color("#FF00FF")

        # === Animation for Lecture Line 2 ===
        prod_curve = axes.plot(lambda x: np.sin(x) * np.cos(2*x), color="#FFFF00")
        area = axes.get_area(prod_curve, [0, 2*PI], color="#FFFF00", opacity=0.3)
        self.play(
            ReplacementTransform(VGroup(sin1, sin2, sine_icon), VGroup(prod_curve, cosine_icon)), 
            FadeIn(area)
        )
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.play(area.animate.set_color("#FF0000"), area.animate.set_opacity(0))
        self.lecture[2].set_color("#FF0000")
        self.wait(1)
