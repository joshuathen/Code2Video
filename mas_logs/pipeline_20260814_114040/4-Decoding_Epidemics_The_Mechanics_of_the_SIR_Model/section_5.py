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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Adjusting transmission variables can flatten curves.",
            "Social distancing reduces the infection peak significantly.",
            "Predictive models guide effective public health policy."
        ]
        self.setup_layout("Simulation & Application: 'Flattening the Curve'", lecture_lines)
        
        # Load asset
        person_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg"
        
        # Create graphs representing infection curves
        axes = Axes(x_range=[0, 10, 1], y_range=[0, 10, 2], axis_config={"include_tip": False}).scale(0.6)
        curve_high = axes.plot(lambda x: 8 * np.exp(-(x - 4)**2 / 2), color=RED)
        curve_low = axes.plot(lambda x: 4 * np.exp(-(x - 5)**2 / 4), color=GREEN)
        
        # Create population icon (mobject)
        person = SVGMobject(person_asset).scale(0.2)
        
        # Layout according to feedback
        self.place_in_area(axes, "C4", "E6", scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(RED))
        self.play(Create(axes), Create(curve_high))
        self.place_at_grid(person.copy(), "B4", scale_factor=0.5)
        self.play(FadeIn(person))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Transform(curve_high, curve_low))
        distancing_icon = person.copy().set_color(YELLOW)
        self.place_at_grid(distancing_icon, "B5", scale_factor=0.5)
        self.play(FadeIn(distancing_icon))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        target_icon = person.copy().set_color(GREEN)
        self.place_at_grid(target_icon, "E5", scale_factor=0.6)
        self.play(FadeIn(target_icon))
        self.wait(2)
