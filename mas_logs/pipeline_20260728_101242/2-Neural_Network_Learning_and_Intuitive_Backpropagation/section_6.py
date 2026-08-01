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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Gradient Descent: Taking the Step",
            [
                "Gradient descent moves us down the error hill.",
                "We update weights in the downhill direction.",
                "The learning rate controls our step size.",
                "Careful steps prevent overshooting the valley floor.",
                "Gradually, we converge on the lowest possible error."
            ]
        )

        # Colors
        HIKER_COLOR = "#FFD700"
        SLOPE_COLOR = "#8B4513"
        GRADIENT_COLOR = "#00FF00"
        SLIDER_COLOR = "#FFFF00"
        
        # Assets
        HIKER_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg"

        # === Animation for Lecture Line 1 ===
        # Create the slope (parabola)
        slope_graph = FunctionGraph(
            lambda x: 0.5 * x**2,
            x_range=[-2.5, 2.5],
            color=SLOPE_COLOR,
            stroke_width=6
        )
        error_hill_group = VGroup(slope_graph)
        # Anchor to grid area (Issue 42)
        self.place_in_area(error_hill_group, 'A1', 'D6', scale_factor=0.9)
        
        # Helper to get world position on the graph
        # Bounding box center of slope_graph is at (0, 1.5625) in its local coords
        def get_pos(x_val):
            # x=0, y=0.5*x**2 maps to center + (0, -1.5625)
            return slope_graph.get_center() + np.array([x_val, 0.5 * x_val**2 - 1.5625, 0])

        # Hiker Asset (Issue 31)
        hiker = SVGMobject(HIKER_ASSET).set_color(HIKER_COLOR)
        hiker.scale(0.3)
        
        # Using ValueTracker for persistent movement (Constraint 10)
        x_tracker = ValueTracker(-2.0)
        hiker.add_updater(lambda m: m.move_to(get_pos(x_tracker.get_value()) + UP * 0.35))

        self.lecture[0].set_color(HIKER_COLOR)
        self.play(Create(slope_graph), FadeIn(hiker))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Downhill arrow
        downhill_arrow = Arrow(color=GRADIENT_COLOR, buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.3)
        downhill_arrow.set_opacity(0)
        
        def update_arrow(arrow):
            x = x_tracker.get_value()
            p = get_pos(x)
            # Tangent is (1, x)
            # If x < 0, downhill is to the right: (1, x)
            # If x > 0, downhill is to the left: (-1, -x)
            if x < 0:
                direction = np.array([1, x, 0])
            else:
                direction = np.array([-1, -x, 0])
            
            mag = np.linalg.norm(direction)
            if mag != 0:
                direction = direction / mag
            
            arrow.put_start_and_end_on(p + UP * 0.1, p + UP * 0.1 + direction * 0.7)

        downhill_arrow.add_updater(update_arrow)
        self.add(downhill_arrow)

        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GRADIENT_COLOR)
        self.play(downhill_arrow.animate.set_opacity(1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Learning Rate Slider
        slider_line = Line(LEFT*1.2, RIGHT*1.2, color=SLIDER_COLOR)
        slider_knob = Dot(color=SLIDER_COLOR, radius=0.12)
        slider_label = Text("Learning Rate", font_size=20, color=SLIDER_COLOR).next_to(slider_line, UP, buff=0.2)
        learning_rate_group = VGroup(slider_line, slider_knob, slider_label)
        
        slider_knob.move_to(slider_line.get_center())
        
        # Anchor to grid area (Issue 43)
        self.place_in_area(learning_rate_group, 'E2', 'F5', scale_factor=0.8)

        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(SLIDER_COLOR)
        self.play(FadeIn(learning_rate_group))
        
        # Show knob adjustment
        self.play(slider_knob.animate.shift(LEFT * 0.4), run_time=0.5)
        self.play(slider_knob.animate.shift(RIGHT * 0.8), run_time=0.5)
        self.play(slider_knob.animate.move_to(slider_line.get_center()), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Careful steps downwards
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GRADIENT_COLOR)
        
        # Step 1
        self.play(x_tracker.animate.set_value(-1.2), run_time=1)
        self.wait(0.5)
        # Step 2
        self.play(x_tracker.animate.set_value(-0.5), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Converge at the bottom (x=0)
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GRADIENT_COLOR)
        
        self.play(
            x_tracker.animate.set_value(0.0),
            downhill_arrow.animate.set_opacity(0),
            run_time=2
        )
        
        # Success indicator (Optimization point)
        check_mark = Text("Optimal", color=GRADIENT_COLOR, font_size=24).next_to(hiker, UP, buff=0.3)
        self.play(Write(check_mark), hiker.animate.scale(1.2))
        self.play(Indicate(hiker), Flash(hiker, color=WHITE))
        
        self.wait(2)

# Marking issues as resolved
# update_issue(31, under_review=True, resolution_note="Integrated hiker SVG asset from provided path.")
# update_issue(42, under_review=True, resolution_note="Anchored error_hill_group to grid area A1-D6.")
# update_issue(43, under_review=True, resolution_note="Anchored learning_rate_group to grid area E2-F5.")

import json
# The following block is just for the internal MAS runner to track issue resolution.
# It is not part of the required output but provided for completeness in thought.
"""
[
    {"call": "update_issue", "args": {"issue_id": 31, "under_review": true, "resolution_note": "Integrated hiker SVG asset from provided path."}},
    {"call": "update_issue", "args": {"issue_id": 42, "under_review": true, "resolution_note": "Anchored error_hill_group to grid area A1-D6."}},
    {"call": "update_issue", "args": {"issue_id": 43, "under_review": true, "resolution_note": "Anchored learning_rate_group to grid area E2-F5."}}
]
"""
