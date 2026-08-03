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
        # Show a falling marble [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg]
        # with its velocity vector (#FF8C00) growing.
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#ADFF2F") 
        
        start_pt = self.grid['C1']
        
        # Resolved issue 25: Integrate SVG asset
        try:
            marble = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg")
            marble.set_color("#ADFF2F")
        except Exception:
            # Fallback if asset is missing or broken
            marble = Circle(radius=0.2, color="#ADFF2F", fill_opacity=1)
            
        self.place_at_grid(marble, 'C1', scale_factor=0.3)
        
        y_tracker = ValueTracker(0)
        
        # Create vector. Use Arrow and update its points.
        velocity_vector = Arrow(start_pt, start_pt + DOWN*0.01, buff=0, color="#FF8C00", stroke_width=4)
        
        def update_marble(m):
            m.move_to(start_pt + DOWN * y_tracker.get_value())

        def update_vector(v):
            curr_y = y_tracker.get_value()
            curr_pos = start_pt + DOWN * curr_y
            # v = sqrt(2gy). We'll scale it so it doesn't get too long.
            v_len = np.sqrt(max(curr_y, 0)) * 0.8
            # Update arrow points
            v.put_start_and_end_on(curr_pos, curr_pos + DOWN * v_len)
            # Visibility logic
            if v_len < 0.05:
                v.set_opacity(0)
            else:
                v.set_opacity(1)

        marble.add_updater(update_marble)
        velocity_vector.add_updater(update_vector)
        
        self.play(FadeIn(marble), FadeIn(velocity_vector))
        # Total distance is 3 grid units (C to F)
        self.play(y_tracker.animate.set_value(3), run_time=2.0, rate_func=linear)
        self.wait(0.5)
        
        marble.clear_updaters()
        velocity_vector.clear_updaters()

        # === Animation for Lecture Line 3 ===
        # Highlight the steep start of the curve with bright pulse (#FFFACD).
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#FFFACD") 
        
        # Simple curve to represent 'steeper start'
        # Using a small t-offset to avoid slope singularity at t=0
        curve = ParametricFunction(
            lambda t: np.array([t * 1.5, - (t**0.5) * 2, 0]),
            t_range=[0.001, 1.5],
            color=WHITE
        )
        
        highlight_segment = ParametricFunction(
            lambda t: np.array([t * 1.5, - (t**0.5) * 2, 0]),
            t_range=[0.001, 0.4],
            color="#FFFACD",
            stroke_width=6
        )
        
        viz_group = VGroup(curve, highlight_segment)
        # Resolved issue 30: Use scale factor 0.8 and position C1-F4
        self.place_in_area(viz_group, 'C1', 'F4', scale_factor=0.8)
        
        self.play(Create(curve))
        self.play(Create(highlight_segment))
        # Pulsing effect
        self.play(highlight_segment.animate.set_stroke(width=12), run_time=0.4)
        self.play(highlight_segment.animate.set_stroke(width=6), run_time=0.4)
        self.wait(1.5)
