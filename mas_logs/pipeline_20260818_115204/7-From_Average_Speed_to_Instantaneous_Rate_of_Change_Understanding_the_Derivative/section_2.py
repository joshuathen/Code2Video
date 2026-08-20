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
        self.setup_layout("Visualizing the Slope: The Secant Line", [
            "A secant line connects two points.",
            "Its slope shows average rate of change.",
            "Slide one point closer to shrink the gap."
        ])

        # Define the curve (position-time, rocket-like)
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], x_length=4, y_length=4)
        curve = axes.plot(lambda x: 0.25 * x**3, color=WHITE)
        self.place_in_area(axes, "B1", "E6")
        self.add(axes, curve)

        # Assets
        pt_a_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg").set_color(BLUE)
        pt_b_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg").set_color(RED)

        # Points on curve
        val_a = ValueTracker(1.0)
        val_b = ValueTracker(3.0)
        
        def get_point_a(): return axes.c2p(val_a.get_value(), 0.25 * val_a.get_value()**3)
        def get_point_b(): return axes.c2p(val_b.get_value(), 0.25 * val_b.get_value()**3)

        pt_a = pt_a_icon.copy().scale(0.2).add_updater(lambda m: m.move_to(get_point_a()))
        pt_b = pt_b_icon.copy().scale(0.2).add_updater(lambda m: m.move_to(get_point_b()))
        
        secant = always_redraw(lambda: Line(get_point_a(), get_point_b(), color="#FFFF00"))

        # Slope Triangle
        slope_triangle = always_redraw(lambda: Polygon(
            get_point_a(),
            [get_point_b()[0], get_point_a()[1], 0],
            get_point_b(),
            color="#00FFFF", fill_opacity=0.2
        ))
        slope_label = Text("Slope", font_size=20, color="#00FFFF")
        self.place_at_grid(slope_label, "A6")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.add(pt_a, pt_b, secant)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.add(slope_triangle, slope_label)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.play(val_b.animate.set_value(1.5), run_time=3)
        self.wait(1)
