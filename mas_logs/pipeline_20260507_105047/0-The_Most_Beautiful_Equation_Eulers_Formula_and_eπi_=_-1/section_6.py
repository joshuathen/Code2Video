from manim import *
import numpy as np

# Use Text as a fallback for MathTex to avoid LaTeX dependency issues
MathTex = Text

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
        # Initialize Layout
        title_text = "Conclusion: Synthesis and Impact"
        lecture_lines = [
            "This identity links exponential growth to circular rotation.",
            "It underpins modern physics, signal processing, and waves.",
            "Complexity simplifies into this one elegant truth."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Colors
        line1_color = YELLOW
        self.play(self.lecture[0].animate.set_color(line1_color))

        # Setup Complex Plane (Unit Circle and Axes)
        plane_axes = Axes(
            x_range=[-1.5, 1.5, 1], y_range=[-1.5, 1.5, 1],
            x_length=3, y_length=3,
            axis_config={"color": BLUE_E, "include_tip": False}
        )
        unit_circle = Circle(radius=1.0, color=WHITE, stroke_opacity=0.5)
        plane_group = VGroup(plane_axes, unit_circle)
        self.place_at_grid(plane_group, "C2", scale_factor=0.8)

        # Vector rotation
        theta_tracker = ValueTracker(0)
        vector = Arrow(
            start=plane_axes.c2p(0, 0),
            end=plane_axes.c2p(1, 0),
            buff=0,
            color=YELLOW
        )
        vector.add_updater(
            lambda v: v.become(
                Arrow(
                    start=plane_axes.c2p(0, 0),
                    end=plane_axes.c2p(np.cos(theta_tracker.get_value()), np.sin(theta_tracker.get_value())),
                    buff=0,
                    color=YELLOW
                )
            )
        )

        # Initial term "e^i\u03c0" anchored to grid (Issue 46)
        e_i_pi_label = Text("e^i\u03c0", color=YELLOW)
        self.place_at_grid(e_i_pi_label, "C3", scale_factor=1.5)

        self.play(Create(plane_group), FadeIn(e_i_pi_label))
        self.play(Create(vector))
        self.play(theta_tracker.animate.set_value(PI), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        line2_color = CYAN
        self.play(self.lecture[1].animate.set_color(line2_color))

        # Traveling Sine Wave
        # We'll create a wave mobject that updates based on the tracker
        wave_config = {"color": line2_color, "stroke_width": 2}
        
        def get_sine_wave():
            current_theta = theta_tracker.get_value()
            points = []
            # Wave moves from right of circle
            start_x = self.grid["C3"][0] - 0.5
            for x_offset in np.arange(0, 3.5, 0.1):
                # The y value is linked to the sin of the current rotation minus distance
                y_val = self.grid["C2"][1] + 0.8 * np.sin(current_theta - x_offset * 2)
                points.append([start_x + x_offset, y_val, 0])
            return VMobject(**wave_config).set_points_as_corners(points)

        wave = always_redraw(get_sine_wave)
        
        # Labels for signal processing (Cyan/Aqua colors)
        label_sp = Text("Signal Processing", color="#00FFFF")
        label_qw = Text("Quantum Waves", color="#00FFFF")
        self.place_at_grid(label_sp, "D5", scale_factor=0.6)
        self.place_at_grid(label_qw, "E5", scale_factor=0.6)

        self.play(FadeIn(wave), FadeIn(label_sp), FadeIn(label_qw))
        
        # Keep rotating to show wave travel
        self.play(theta_tracker.animate.set_value(4 * PI), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        line3_color = GOLD
        self.play(self.lecture[2].animate.set_color(line3_color))

        # Fade out elements to clear for the final truth
        self.play(
            FadeOut(plane_group),
            FadeOut(vector),
            FadeOut(wave),
            FadeOut(label_sp),
            FadeOut(label_qw),
            FadeOut(e_i_pi_label)
        )

        # The Most Beautiful Equation (Euler's Identity)
        # Applying Issue 47: Area B2 to E5, scale 1.4
        euler_identity = Text("e^i\u03c0 + 1 = 0", color=GOLD)
        self.place_in_area(euler_identity, "B2", "E5", scale_factor=1.4)

        self.play(Write(euler_identity))
        self.play(Indicate(euler_identity, color=GOLD))
        self.wait(3)
