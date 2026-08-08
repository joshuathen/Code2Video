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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Limiting Process: Getting Closer", [
            "Slide the second point toward the first.",
            "Observe the secant line rotating.",
            "It flattens into a tangent line."
        ])

        # Setup Graph and Points
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.25 * x**2, x_range=[0, 4])
        
        # Fix: axes positioning
        self.place_in_area(axes, 'B2', 'F6', scale_factor=0.5)
        curve.move_to(axes.get_center()) # Ensure curve stays with axes
        
        self.add(axes)
        self.add(curve)

        t1 = 1.0
        # Trackers
        t2_tracker = ValueTracker(3.5)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Loading SVG, if not found, use circle
        try:
            point_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        except:
            point_icon = Circle(radius=0.1, color=WHITE, fill_opacity=1)

        p1 = Dot(axes.c2p(t1, 0.25 * t1**2), color=WHITE)
        
        # Use point_icon for p2
        p2 = point_icon.copy()
        p2.scale(0.5)
        
        def update_p2(m):
            m.move_to(axes.c2p(t2_tracker.get_value(), 0.25 * t2_tracker.get_value()**2))
            
        p2.add_updater(update_p2)

        secant_line = always_redraw(lambda: axes.get_secant_slope_group(
            t1, curve, dx=t2_tracker.get_value()-t1, secant_line_color=GREEN, dx_line_color=YELLOW
        ))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.play(Create(p1), Create(p2), Create(secant_line))
        self.play(t2_tracker.animate.set_value(1.1), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#32CD32")
        self.play(t2_tracker.animate.set_value(1.05), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        self.play(t2_tracker.animate.set_value(1.001), run_time=1)
        self.wait(1)
