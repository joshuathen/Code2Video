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
        self.setup_layout("Visualizing the Bell Curve", [
            "Two key parameters emerge: mean and variance.", 
            "The mean of means equals population mean.", 
            "Variance shrinks as sample size grows."
        ])
        
        # Load assets
        curve_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg")
        bell_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg")
        
        # Bell curve elements
        axes = Axes(x_range=[-4, 4, 1], y_range=[0, 1, 0.2], axis_config={"include_tip": False})
        bell_curve = axes.plot(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color=WHITE)
        axes_group = VGroup(axes, bell_curve)
        self.place_in_area(axes_group, 'B2', 'E5', scale_factor=0.45)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(curve_icon, 'A2', scale_factor=0.5)
        self.play(FadeIn(curve_icon), Create(bell_curve), run_time=2)
        self.play(self.lecture[0].animate.set_color("#87CEEB"))

        # === Animation for Lecture Line 2 ===
        mean_line = Line(axes.c2p(0, 0), axes.c2p(0, 0.4), color="#00FF00")
        # Creating a green box as requested by critic 31
        green_bar = Rectangle(height=0.5, width=0.1, color="#00FF00", fill_opacity=1)
        self.place_at_grid(green_bar, 'D4', scale_factor=0.6)
        self.play(Create(mean_line), FadeIn(green_bar))
        self.play(self.lecture[1].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 3 ===
        # Creating a variance box as requested by critic 32
        variance_box = Rectangle(height=0.5, width=0.5, color="#FFD700", fill_opacity=1)
        self.place_at_grid(variance_box, 'D5', scale_factor=0.5)
        self.place_at_grid(bell_icon, 'F4', scale_factor=0.5)
        
        # Create shaded region
        shaded_area = axes.get_area(bell_curve, x_range=[-1, 1], color="#FFD700", opacity=0.3)
        self.play(FadeIn(shaded_area), FadeIn(variance_box), FadeIn(bell_icon))
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        self.wait(2)
