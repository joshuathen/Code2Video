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

class Section5Scene(TeachingScene):
    def construct(self):
        # Updated lines to match Teaching Content script
        lines = [
            '- Every hit moves the point by a fixed arc angle.',
            '- This angle depends on the ratio of the masses.',
            '- The point travels along the perimeter of the circle.',
            '- We count how many steps fit into half the circle.',
            '- The final count reveals the digits of Pi.'
        ]
        self.setup_layout("The Angle of Pi: Why the Counting Works", lines)
        
        # Define constants and colors
        COLOR_THETA = "#FFFF00"
        COLOR_PI = "#FFFFFF"
        initial_theta = 0.6  # Initial visual theta
        small_theta = 0.2    # Representing larger mass
        circle_radius = 1.8

        # Persistent Circle - Fixed position per Issue 40
        circle = Circle(radius=circle_radius, color=GREY_E)
        self.place_in_area(circle, "B3", "E6")
        dot = Dot(circle.point_at_angle(0), color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_THETA))
        self.play(Create(circle))
        
        # Arc and Theta Label
        arc1 = Arc(radius=circle_radius, start_angle=0, angle=initial_theta, color=COLOR_THETA)
        arc1.move_to(circle.get_center())
        
        # Unicode Theta label
        theta_label = Text("θ", color=COLOR_THETA, font_size=24)
        label_pos = circle.get_center() + 2.2 * np.array([np.cos(initial_theta/2), np.sin(initial_theta/2), 0])
        theta_label.move_to(label_pos)
        
        self.play(Create(arc1), FadeIn(theta_label))
        self.play(dot.animate.move_to(circle.point_at_angle(initial_theta)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_THETA)
        )
        
        # Formula - Fixed position per Issue 41
        theta_eq = Text("θ = arctan(√(m/M))", color=COLOR_THETA, font_size=24)
        self.place_in_area(theta_eq, "A3", "A6", scale_factor=0.9)
        
        semi_circle_bg = Arc(radius=circle_radius, start_angle=0, angle=PI, color=WHITE, stroke_width=2)
        semi_circle_bg.move_to(circle.get_center())
        
        self.play(Write(theta_eq))
        self.play(Create(semi_circle_bg))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_THETA)
        )
        
        # Update angle to show smaller steps for higher mass
        new_arc = Arc(radius=circle_radius, start_angle=0, angle=small_theta, color=COLOR_THETA)
        new_arc.move_to(circle.get_center())
        
        new_label_pos = circle.get_center() + 2.1 * np.array([np.cos(small_theta/2), np.sin(small_theta/2), 0])
        
        self.play(
            ReplacementTransform(arc1, new_arc),
            theta_label.animate.move_to(new_label_pos).scale(0.8),
            dot.animate.move_to(circle.point_at_angle(small_theta))
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_THETA)
        )
        
        # Fill semi-circle with steps
        num_arcs = int(PI / small_theta)
        arcs = VGroup()
        for i in range(1, num_arcs):
            a = Arc(radius=circle_radius, start_angle=i*small_theta, angle=small_theta, color=COLOR_THETA)
            a.move_to(circle.get_center())
            arcs.add(a)
        
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.1, run_time=2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_PI)
        )
        
        # Final Formula - Fixed position per Issue 41
        final_eq = Text("Collisions = ⌊π/θ⌋", color=COLOR_PI, font_size=24)
        self.place_in_area(final_eq, "F3", "F6", scale_factor=1.0)
        
        # Pi Label - Fixed position per Issue 39
        pi_label = Text("π radians", color=WHITE, font_size=24)
        self.place_in_area(pi_label, "C2", "D2", scale_factor=0.7)
        
        self.play(Write(final_eq), FadeIn(pi_label))
        self.wait(2)
