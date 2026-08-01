from manim import *
import numpy as np

# Base class provided in the prompt
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
        # Setup data
        title_text = "Visualizing Convolution: Flip, Shift, and Area"
        lecture_lines = [
            "First, we flip one distribution horizontally.",
            "Then, we shift it by the total sum.",
            "We calculate the area where they overlap.",
            "Two uniform blocks create a triangular sum.",
            "The resulting shape is the new distribution."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_PULSE1 = WHITE
        COLOR_FLIP = "#FF4500"
        COLOR_SHIFT = "#1E90FF"
        COLOR_OVERLAP = "#32CD32"
        COLOR_TRIANGLE = "#00FFFF"
        COLOR_LABEL = "#FFFFFF"

        # 1. Setup Axes
        # Use a size that fits well within the grid area B1-E5
        axes = Axes(
            x_range=[-2, 3, 1],
            y_range=[0, 1.5, 1],
            x_length=4.5,
            y_length=2.5,
            axis_config={"include_tip": False, "font_size": 20}
        )
        self.place_in_area(axes, "B1", "E5", scale_factor=0.9)
        
        # PULSE 1: f(t) = 1 for t in [0, 1]
        p1_top = Line(axes.c2p(0, 1), axes.c2p(1, 1), color=COLOR_PULSE1)
        p1_l = Line(axes.c2p(0, 0), axes.c2p(0, 1), color=COLOR_PULSE1)
        p1_r = Line(axes.c2p(1, 0), axes.c2p(1, 1), color=COLOR_PULSE1)
        pulse1_group = VGroup(p1_top, p1_l, p1_r)

        # PULSE 2: g(t) = 1 for t in [0, 1]
        p2_top = Line(axes.c2p(0, 1), axes.c2p(1, 1), color=COLOR_PULSE1)
        p2_l = Line(axes.c2p(0, 0), axes.c2p(0, 1), color=COLOR_PULSE1)
        p2_r = Line(axes.c2p(1, 0), axes.c2p(1, 1), color=COLOR_PULSE1)
        pulse2_group = VGroup(p2_top, p2_l, p2_r)

        label_f = MathTex("f(t)", color=COLOR_PULSE1, font_size=24)
        # Fix Issue 38: Move f(t) label to Row A to avoid collision with shifting elements
        self.place_at_grid(label_f, "A3", scale_factor=0.8)
        
        label_g = MathTex("g(t)", color=COLOR_PULSE1, font_size=24)
        # Position label_g in Row A for consistency
        self.place_at_grid(label_g, "A4", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # First, we flip one distribution horizontally.
        self.play(self.lecture[0].animate.set_color(COLOR_FLIP))
        self.play(Create(axes), FadeIn(pulse1_group), FadeIn(pulse2_group), Write(label_f), Write(label_g))
        self.wait(0.5)

        # Flip pulse 2 to g(-t) on [-1, 0]
        p2_flipped_top = Line(axes.c2p(-1, 1), axes.c2p(0, 1), color=COLOR_FLIP)
        p2_flipped_l = Line(axes.c2p(-1, 0), axes.c2p(-1, 1), color=COLOR_FLIP)
        p2_flipped_r = Line(axes.c2p(0, 0), axes.c2p(0, 1), color=COLOR_FLIP)
        pulse2_flipped_group = VGroup(p2_flipped_top, p2_flipped_l, p2_flipped_r)
        
        label_g_flipped = MathTex("g(-t)", color=COLOR_FLIP, font_size=24)
        # Fix Issue 39: Move g(-t) label to Row A to prevent visual crowding
        self.place_at_grid(label_g_flipped, "A2", scale_factor=0.8)

        self.play(
            Transform(pulse2_group, pulse2_flipped_group),
            Transform(label_g, label_g_flipped)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Then, we shift it by the total sum.
        self.play(self.lecture[1].animate.set_color(COLOR_SHIFT))
        
        z_tracker = ValueTracker(-1)
        
        # Shifted pulse g(z-t)
        sp_top = Line(axes.c2p(-1, 1), axes.c2p(0, 1), color=COLOR_SHIFT)
        sp_l = Line(axes.c2p(-1, 0), axes.c2p(-1, 1), color=COLOR_SHIFT)
        sp_r = Line(axes.c2p(0, 0), axes.c2p(0, 1), color=COLOR_SHIFT)
        shifted_pulse = VGroup(sp_top, sp_l, sp_r)
        
        label_g_shifted = MathTex("g(z-t)", color=COLOR_SHIFT, font_size=24)
        
        def update_shifted_pulse(mob):
            z = z_tracker.get_value()
            mob.move_to(axes.c2p(z - 0.5, 0.5))
            
        def update_shifted_label(mob):
            z = z_tracker.get_value()
            mob.move_to(axes.c2p(z - 0.5, 1.25))

        shifted_pulse.add_updater(update_shifted_pulse)
        label_g_shifted.add_updater(update_shifted_label)
        
        self.play(
            FadeOut(pulse2_group), 
            FadeOut(label_g),
            FadeIn(shifted_pulse),
            FadeIn(label_g_shifted)
        )
        self.play(z_tracker.animate.set_value(0), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # We calculate the area where they overlap.
        self.play(self.lecture[2].animate.set_color(COLOR_OVERLAP))
        
        overlap_rect = Rectangle(color=COLOR_OVERLAP, fill_opacity=0.6, stroke_width=0)
        
        def update_overlap(mob):
            z = z_tracker.get_value()
            start = max(0, z-1)
            end = min(1, z)
            if start < end:
                w = axes.c2p(end, 0)[0] - axes.c2p(start, 0)[0]
                h = axes.c2p(0, 1)[1] - axes.c2p(0, 0)[1]
                mob.stretch_to_fit_width(w)
                mob.stretch_to_fit_height(h)
                mob.move_to(axes.c2p((start+end)/2, 0.5))
                mob.set_fill(opacity=0.6)
            else:
                mob.set_fill(opacity=0)

        overlap_rect.add_updater(update_overlap)
        self.add(overlap_rect)

        triangle_trace = VMobject(color=COLOR_TRIANGLE, stroke_width=4)
        triangle_trace.set_points_as_corners([axes.c2p(0,0), axes.c2p(0,0)])
        
        def update_trace(mob):
            curr_z = z_tracker.get_value()
            if curr_z < 0:
                mob.set_points_as_corners([axes.c2p(0,0), axes.c2p(0,0)])
                return
            zs = np.linspace(0, min(2.0, curr_z), 40)
            pts = [axes.c2p(z, max(0, 1 - abs(z-1))) for z in zs]
            mob.set_points_as_corners(pts)

        triangle_trace.add_updater(update_trace)
        self.add(triangle_trace)

        self.play(z_tracker.animate.set_value(1.0), run_time=3)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Two uniform blocks create a triangular sum.
        self.play(self.lecture[3].animate.set_color(COLOR_TRIANGLE))
        
        self.play(z_tracker.animate.set_value(2.0), run_time=3)
        self.wait(0.5)
        
        triangle_trace.remove_updater(update_trace)
        overlap_rect.remove_updater(update_overlap)
        shifted_pulse.remove_updater(update_shifted_pulse)
        label_g_shifted.remove_updater(update_shifted_label)
        
        self.play(Indicate(triangle_trace, color=COLOR_TRIANGLE))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # The resulting shape is the new distribution.
        self.play(self.lecture[4].animate.set_color(COLOR_LABEL))
        
        label_final = Text("Sum Distribution Z", font_size=20, color=COLOR_LABEL)
        # Fix Issue 37: Move final label to top-right area to avoid obstructing the plot
        self.place_in_area(label_final, "A5", "A6", scale_factor=0.8)
        
        self.play(Write(label_final))
        self.wait(2)
