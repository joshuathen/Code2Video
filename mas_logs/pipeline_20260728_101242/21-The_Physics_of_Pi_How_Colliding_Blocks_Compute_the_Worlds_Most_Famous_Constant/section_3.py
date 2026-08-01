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
        self.setup_layout(
            "Prerequisite: The Laws of Conservation",
            [
                "Conservation of energy and momentum govern every collision.",
                "We can track the blocks using their velocities.",
                "The energy equation forms an ellipse in velocity space."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        momentum_eq = MathTex("m v_1 + M v_2 = C", color="#00FFFF")
        energy_eq = MathTex(r"\frac{1}{2} m v_1^2 + \frac{1}{2} M v_2^2 = E", color="#FF4500")
        
        # Positioning fixes based on Issues 26 & 27
        self.place_in_area(momentum_eq, "A3", "A5", scale_factor=0.8)
        self.place_in_area(energy_eq, "B3", "B5", scale_factor=0.8)
        
        self.play(Write(momentum_eq), run_time=1)
        self.play(Write(energy_eq), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=3,
            axis_config={"color": WHITE, "include_tip": True},
            tips=False
        )
        labels = axes.get_axis_labels(x_label="v_1", y_label="v_2")
        axes_group = VGroup(axes, labels)
        
        # Positioning fix based on Issue 28
        self.place_in_area(axes_group, "C3", "F6", scale_factor=0.7)
        
        self.play(Create(axes), Write(labels), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Ellipse parameters for visualization
        a_semi, b_semi = 2.0, 1.2
        ellipse = axes.plot_parametric_curve(
            lambda t: np.array([a_semi * np.cos(t), b_semi * np.sin(t), 0]),
            t_range=[0, 2 * PI],
            color="#FFFFFF"
        )
        
        self.play(Create(ellipse), run_time=1.5)
        
        # Pulsing point moving along the ellipse
        dot = Dot(color="#FFFF00")
        t_tracker = ValueTracker(0)
        
        # Updater for position and pulsing effect
        def update_dot(mob):
            t = t_tracker.get_value()
            mob.move_to(axes.c2p(a_semi * np.cos(t), b_semi * np.sin(t)))
            pulse = 1 + 0.3 * np.sin(t * 8)
            mob.set_width(0.15 * pulse)

        dot.add_updater(update_dot)
        self.add(dot)
        
        # Move the point once around the ellipse
        self.play(t_tracker.animate(run_time=6, rate_func=linear).set_value(2 * PI))
        self.wait(2)
