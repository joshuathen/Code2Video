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
        # Data from storyboard
        lecture_lines = [
            "The mass ratio determines the angle of each jump.",
            "Larger masses create smaller angles and more collisions.",
            "Each collision consumes a specific arc of the circle.",
            "The total collisions are limited by the circle's circumference.",
            "This geometric limit is where Pi enters the equation."
        ]
        
        self.setup_layout("The Arc Length and Pi", lecture_lines)
        
        # Colors
        color_1 = "#87CEEB" # Sky Blue
        color_2 = "#98FB98" # Pale Green
        color_3 = "#FFA500" # Orange
        color_4 = "#FFB6C1" # Light Pink
        color_5 = "#FFFFE0" # Light Yellow

        # === Animation for Lecture Line 1 ===
        # "The mass ratio determines the angle of each jump."
        self.lecture[0].set_color(color_1)
        
        circle = Circle(radius=1.8, color=WHITE)
        # Fix: Positioning the circle in area C2-F5 to utilize the bottom row (Issue 38/33)
        self.place_in_area(circle, "C2", "F5", scale_factor=1.0)
        center = circle.get_center()
        
        theta_val = 0.8
        line1 = Line(center, center + 1.8 * RIGHT, color=color_1)
        # Using rotate_vector from manim
        vec2 = rotate_vector(1.8 * RIGHT, theta_val)
        line2 = Line(center, center + vec2, color=color_1)
        angle_arc = Arc(radius=0.6, start_angle=0, angle=theta_val, color=color_1).move_arc_center_to(center)
        theta_label = MathTex(r"\theta", color=color_1, font_size=36)
        label_pos = rotate_vector(0.9 * RIGHT, theta_val/2)
        theta_label.move_to(center + label_pos)
        
        self.play(Create(circle))
        self.play(Create(line1), Create(line2), Create(angle_arc), Write(theta_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Larger masses create smaller angles and more collisions."
        self.lecture[1].set_color(color_2)
        
        new_theta_val = 0.4
        new_vec2 = rotate_vector(1.8 * RIGHT, new_theta_val)
        new_line2 = Line(center, center + new_vec2, color=color_2)
        new_angle_arc = Arc(radius=0.6, start_angle=0, angle=new_theta_val, color=color_2).move_arc_center_to(center)
        new_label_pos = rotate_vector(0.9 * RIGHT, new_theta_val/2)
        
        self.play(
            line2.animate.become(new_line2),
            angle_arc.animate.become(new_angle_arc),
            theta_label.animate.move_to(center + new_label_pos).set_color(color_2),
            line1.animate.set_color(color_2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Each collision consumes a specific arc of the circle."
        # Highlight a small arc (#FFA500)
        self.lecture[2].set_color(color_3)
        
        arc_highlight = Arc(radius=1.8, start_angle=0, angle=new_theta_val, color=color_3, stroke_width=8).move_arc_center_to(center)
        self.play(Create(arc_highlight))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The total collisions are limited by the circle's circumference."
        # Clock-like hand sweeping
        self.lecture[3].set_color(color_4)
        
        # Cleanup some elements for clarity, keep circle
        self.play(FadeOut(line1), FadeOut(line2), FadeOut(angle_arc), FadeOut(theta_label), FadeOut(arc_highlight))
        
        # Integrate clock.svg asset (Issue 38/20)
        clock_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")
        clock_icon.scale(0.3).move_to(center)
        self.play(FadeIn(clock_icon))
        
        hand = Line(center, center + 1.8 * RIGHT, color=color_4, stroke_width=4)
        self.add(hand)
        
        # Total steps for a full circle
        num_steps = int(TAU / new_theta_val)
        
        # Discrete steps for the sweep
        for i in range(1, num_steps + 1):
            start_a = (i-1) * new_theta_val
            # Use 0.15s per step to keep it snappy and within budget
            self.play(Rotate(hand, angle=new_theta_val, about_point=center), run_time=0.15, rate_func=linear)
            s_arc = Arc(radius=1.8, start_angle=start_a, angle=new_theta_val, color=color_4).move_arc_center_to(center)
            self.add(s_arc)
            
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This geometric limit is where Pi enters the equation."
        # Display '2pi / theta'
        self.lecture[4].set_color(color_5)
        
        formula = MathTex(r"\text{Collisions} \approx \frac{2\pi}{\theta}", color=color_5, font_size=40)
        # Fix: Positioning formula in area A3-B5 with scale 0.8 (Issue 38/32)
        self.place_in_area(formula, "A3", "B5", scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(2)
