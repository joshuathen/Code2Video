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
        self.setup_layout("Application: Visualizing the Slope", [
            "Observe the slope changing along the curve.",
            "The derivative is the tangent line's slope.",
            "This connects algebra to geometry. [Asset: tangent_slider_anim]",
            "Like a drone measuring climb rate.",
            "Instantaneous changes visualized in real time. [Asset: climb_rate_anim]"
        ])
        
        # Setup Axes and Parabola
        axes = Axes(x_range=[-2, 2], y_range=[-1, 4], axis_config={"include_tip": True}).scale(0.5)
        curve = axes.plot(lambda x: x**2, color=BLUE)
        axes_and_curve = VGroup(axes, curve)
        self.place_in_area(axes_and_curve, "C2", "E6", scale_factor=0.75)
        self.add(axes_and_curve)

        # Point for tangent
        t = ValueTracker(-1.5)
        dot = Dot(color=YELLOW)
        dot.add_updater(lambda m: m.move_to(axes.c2p(t.get_value(), t.get_value()**2)))
        
        def get_tangent():
            val = t.get_value()
            pt = axes.c2p(val, val**2)
            slope = 2 * val
            return Line(start=pt + LEFT * 0.2 + LEFT * 0.2 * slope, end=pt + RIGHT * 0.2 + RIGHT * 0.2 * slope, color="#00FF00")
            
        tangent = always_redraw(get_tangent)
        self.add(dot, tangent)

        # Assets
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg")
        drone.add_updater(lambda m: m.move_to(axes.c2p(t.get_value(), t.get_value()**2)))
        
        # Placeholder for asset-specific logic from storyboard
        tangent_slider_anim = VGroup(tangent, dot)
        self.place_at_grid(tangent_slider_anim, "C2", scale_factor=0.6)
        
        climb_rate_anim = drone
        self.place_at_grid(climb_rate_anim, "F2", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        self.play(t.animate.set_value(1.5), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_opacity(1)
        self.add(drone)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_opacity(1)
        self.wait(2)
