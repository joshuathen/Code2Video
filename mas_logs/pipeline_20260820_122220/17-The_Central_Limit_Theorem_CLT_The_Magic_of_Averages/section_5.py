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
        self.setup_layout("Summary & Key Takeaway", [
            "Start with any distribution shape.",
            "Collect repeated samples and calculate means.",
            "Plot these means on a chart.",
            "Observe the emerging normal bell curve.",
            "Larger samples result in a tighter curve."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sample.svg]
        shape_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sample.svg", color=WHITE)
        self.place_at_grid(shape_icon, 'B5', scale_factor=0.6)
        self.play(FadeIn(shape_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        dots = VGroup(*[Dot(color=GREEN) for _ in range(5)])
        self.place_in_area(dots, 'B4', 'B6', scale_factor=1.0)
        self.play(FadeIn(dots))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/chart.svg]
        chart_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chart.svg", color=GREEN)
        axes = Axes(x_range=[-2, 2, 1], y_range=[0, 1, 0.5], x_length=3, y_length=2).add_coordinates()
        axes_group = VGroup(axes, chart_icon)
        self.place_in_area(axes_group, 'D3', 'E5', scale_factor=0.8)
        self.play(Create(axes), FadeIn(chart_icon))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        curve = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-2, 2], color=YELLOW)
        self.place_in_area(curve, 'D3', 'E5', scale_factor=0.8)
        self.play(Create(curve))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg]
        dice_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg", color=YELLOW)
        meter = Rectangle(width=3, height=0.5, color=WHITE)
        fill = Rectangle(width=2.5, height=0.4, color=RED).set_fill(RED, opacity=0.8).align_to(meter, LEFT)
        meter_group = VGroup(meter, fill, dice_icon)
        self.place_at_grid(meter_group, 'E3', scale_factor=0.7)
        self.play(Create(meter), FadeIn(fill), FadeIn(dice_icon))
        self.play(fill.animate.set_width(2.5))
