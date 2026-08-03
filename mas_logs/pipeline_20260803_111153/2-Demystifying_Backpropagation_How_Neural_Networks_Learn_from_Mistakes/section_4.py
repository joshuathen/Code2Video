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
        # Define lecture lines
        lecture_lines = [
            "Backpropagation moves backward from the error to inputs.",
            "It calculates the gradient for every internal weight.",
            "We determine how much each knob contributed to mistakes.",
            "Influence arrows show which weights need the most change.",
            "This process assigns blame to specific network connections."
        ]
        self.setup_layout("Backpropagation: Assigning Blame", lecture_lines)
        
        # Define colors for animations
        COLOR_ERROR = "#FF0000"
        COLOR_GRADIENT = "#FF00FF"
        COLOR_KNOB_ADJUST = "#00FF00"
        COLOR_BLAME = "#FF00FF"
        
        # --- Pre-calculate positions and setup network elements ---
        
        # Error node (Starting point of backprop)
        error_node = Circle(radius=0.4, color=COLOR_ERROR, fill_opacity=0.5)
        error_label = Text("Error", font_size=16, color=WHITE).next_to(error_node, RIGHT, buff=0.1)
        error_group = VGroup(error_node, error_label)
        self.place_at_grid(error_group, "C6")
        
        # Intermediate nodes (Weights)
        w1 = Dot(radius=0.15, color=GRAY)
        w2 = Dot(radius=0.15, color=GRAY)
        self.place_at_grid(w1, "B4")
        self.place_at_grid(w2, "E4")
        
        # Knobs (Inputs)
        heat_knob = Circle(radius=0.3, color=WHITE)
        heat_label = Text("Heat", font_size=16, color=WHITE).next_to(heat_knob, LEFT, buff=0.1)
        heat_group = VGroup(heat_knob, heat_label)
        self.place_at_grid(heat_group, "B2")
        
        time_knob = Circle(radius=0.3, color=WHITE)
        time_label = Text("Time", font_size=16, color=WHITE).next_to(time_knob, LEFT, buff=0.1)
        time_group = VGroup(time_knob, time_label)
        self.place_at_grid(time_group, "E2")
        
        # Connectivity (Forward direction visuals)
        conn_1 = Line(heat_group.get_right(), w1.get_left(), color=GRAY, stroke_width=2)
        conn_2 = Line(time_group.get_right(), w2.get_left(), color=GRAY, stroke_width=2)
        conn_3 = Line(w1.get_right(), error_node.get_left(), color=GRAY, stroke_width=2)
        conn_4 = Line(w2.get_right(), error_node.get_left(), color=GRAY, stroke_width=2)
        
        # Group network for initial entry
        network = VGroup(heat_group, time_group, w1, w2, error_group, conn_1, conn_2, conn_3, conn_4)
        self.add(network)

        # === Animation for Lecture Line 1 ===
        # Highlight the Error node (#FF0000) and start a pulse moving backward through the network.
        self.play(self.lecture[0].animate.set_color(COLOR_ERROR))
        
        # Pulse effect on error node
        pulse = Circle(radius=0.4, color=COLOR_ERROR).move_to(error_node.get_center())
        self.play(
            error_node.animate.scale(1.2).set_fill(opacity=0.8),
            FadeIn(pulse),
            run_time=0.5
        )
        self.play(
            error_node.animate.scale(1/1.2).set_fill(opacity=0.5),
            pulse.animate.scale(2.0).set_stroke(opacity=0),
            run_time=0.5
        )
        self.remove(pulse)
        
        # Backward moving pulses
        p1 = Dot(color=COLOR_ERROR).move_to(error_node.get_center())
        p2 = Dot(color=COLOR_ERROR).move_to(error_node.get_center())
        self.play(
            p1.animate.move_to(w1.get_center()),
            p2.animate.move_to(w2.get_center()),
            run_time=1
        )
        self.play(
            p1.animate.move_to(heat_group.get_center()),
            p2.animate.move_to(time_group.get_center()),
            run_time=1
        )
        self.remove(p1, p2)

        # === Animation for Lecture Line 2 ===
        # It calculates the gradient for every internal weight.
        # Show arrows (#FF00FF) pointing backward from Error to weights, with varying thickness.
        self.play(self.lecture[1].animate.set_color(COLOR_GRADIENT))
        
        grad_arrow_top = Arrow(error_node.get_left(), w1.get_right(), color=COLOR_GRADIENT, buff=0.1, stroke_width=6)
        grad_arrow_bottom = Arrow(error_node.get_left(), w2.get_right(), color=COLOR_GRADIENT, buff=0.1, stroke_width=3)
        
        self.play(GrowArrow(grad_arrow_top), GrowArrow(grad_arrow_bottom))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We determine how much each knob contributed to mistakes.
        # Place a directional arrow (#00FF00) on a knob indicating the adjustment direction.
        self.play(self.lecture[2].animate.set_color(COLOR_KNOB_ADJUST))
        
        adjust_arrow = Arrow(UP, DOWN, color=COLOR_KNOB_ADJUST).scale(0.5)
        adjust_arrow.next_to(time_knob, RIGHT, buff=0.2)
        
        self.play(FadeIn(adjust_arrow))
        self.play(adjust_arrow.animate.shift(DOWN * 0.2), run_time=0.5, rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Influence arrows show which weights need the most change.
        # Highlight a thick arrow labeled 'High Blame' pointing to the 'Time' knob.
        self.play(self.lecture[3].animate.set_color(COLOR_BLAME))
        
        high_blame_arrow = Arrow(w2.get_left(), time_group.get_right(), color=COLOR_BLAME, stroke_width=10, buff=0.1)
        high_blame_label = Text("High Blame", font_size=14, color=COLOR_BLAME)
        high_blame_label.next_to(high_blame_arrow, DOWN, buff=0.1)
        
        self.play(
            GrowArrow(high_blame_arrow),
            FadeIn(high_blame_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This process assigns blame to specific network connections.
        # Highlight a thin arrow labeled 'Low Blame' pointing to the 'Heat' knob.
        self.play(self.lecture[4].animate.set_color(COLOR_BLAME))
        
        low_blame_arrow = Arrow(w1.get_left(), heat_group.get_right(), color=COLOR_BLAME, stroke_width=2, buff=0.1)
        low_blame_label = Text("Low Blame", font_size=14, color=COLOR_BLAME)
        low_blame_label.next_to(low_blame_arrow, UP, buff=0.1)
        
        self.play(
            GrowArrow(low_blame_arrow),
            FadeIn(low_blame_label)
        )
        self.wait(2)
