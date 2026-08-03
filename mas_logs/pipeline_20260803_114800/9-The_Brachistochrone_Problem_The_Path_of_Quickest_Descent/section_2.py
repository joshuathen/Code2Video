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
        # Setup layout
        self.setup_layout("Prerequisite Knowledge: Speed and Energy", [
            "Conservation of energy dictates speed based on vertical drop.",
            "Velocity equals the square root of 2gh.",
            "Steeper starts grant more speed early in the descent."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display the energy-to-speed formula v = sqrt(2gy) in white (#FFFFFF)
        self.lecture[0].set_color(WHITE)
        formula = MathTex(r"v = \sqrt{2gy}", color=WHITE)
        # Resolved issue 29: Use scale factor 1.0 and position A4-B6
        self.place_in_area(formula, 'A4', 'B6', scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Falling marble [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg]
        # with its velocity vector (#FF8C00) growing.
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#FF8C00") 
        
        # Resolved issue 25: Integrate SVG asset
        try:
            marble = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg")
            marble.set_color(WHITE)
        except:
            # Fallback if asset is missing
            marble = Circle(radius=0.15, color=WHITE, fill_opacity=1)
            
        self.place_at_grid(marble, 'C1', scale_factor=0.4)
        start_pos = marble.get_center().copy()
        
        # Velocity vector
        velocity_vector = Arrow(UP, DOWN, color="#FF8C00", buff=0, stroke_width=4)
        velocity_vector.set_opacity(0)
        
        y_tracker = ValueTracker(0)
        
        # Updaters: Keep them simple to avoid timeouts
        marble.add_updater(lambda m: m.move_to(start_pos + DOWN * y_tracker.get_value()))
        
        def vector_updater(v):
            val = y_tracker.get_value()
            if val <= 0.05:
                v.set_opacity(0)
                return
            v.set_opacity(1)
            curr_pos = marble.get_center()
            # Scaling: length = sqrt(y) * constant
            v_len = np.sqrt(val) * 0.8
            v.put_start_and_end_on(curr_pos, curr_pos + DOWN * v_len)
            
        velocity_vector.add_updater(vector_updater)
        
        self.add(marble, velocity_vector)
        # Fall 3 grid units down
        self.play(y_tracker.animate.set_value(3.0), run_time=2.0, rate_func=linear)
        self.wait(0.5)
        
        marble.clear_updaters()
        velocity_vector.clear_updaters()

        # === Animation for Lecture Line 3 ===
        # Highlight the steep start of the curve with bright pulse (#FFFACD).
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#FFFACD") 
        
        # Create a brachistochrone-like curve using smooth points for speed
        curve_pts = [
            np.array([0, 0, 0]),
            np.array([0.2, -1.0, 0]),
            np.array([1.0, -1.8, 0]),
            np.array([2.5, -2.0, 0])
        ]
        curve = VMobject(color=WHITE)
        curve.set_points_smoothly(curve_pts)
        
        # Highlight segment for the "steeper start"
        highlight_pts = [
            np.array([0, 0, 0]),
            np.array([0.2, -1.0, 0]),
            np.array([0.4, -1.3, 0])
        ]
        highlight_segment = VMobject(color="#FFFACD", stroke_width=6)
        highlight_segment.set_points_smoothly(highlight_pts)
        
        viz_group = VGroup(curve, highlight_segment)
        # Resolved issue 30: Use scale factor 0.8 and position C1-F4
        self.place_in_area(viz_group, 'C1', 'F4', scale_factor=0.8)
        
        self.play(Create(curve))
        self.play(Create(highlight_segment))
        # Pulse effect using there_and_back to be efficient
        self.play(
            highlight_segment.animate.set_stroke(width=12),
            run_time=0.5,
            rate_func=there_and_back
        )
        self.wait(1.5)
