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
        self.setup_layout("The Secret Angle: Why Pi?", [
            "Each collision rotates the state by a constant angle.",
            "This angle depends on the ratio of block masses.",
            "We count how many jumps fit within the circle.",
            "Total collisions equal the total arc divided by angle.",
            "Half the circle represents Pi radians of total rotation."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Circle setup
        circle = Circle(radius=2, color="#00FFFF") # Cyan
        self.place_in_area(circle, "A1", "F6", scale_factor=0.8)
        center = circle.get_center()
        self.add(circle)

        # State point
        state_point = Dot(color="#FFFF00") # Yellow
        state_point.move_to(circle.point_at_angle(0))
        self.add(state_point)

        # Constant angle jumps
        theta_val = 30 * DEGREES
        num_jumps = 4
        
        for i in range(1, num_jumps + 1):
            target_angle = i * theta_val
            self.play(
                state_point.animate.move_to(circle.point_at_angle(target_angle)),
                run_time=0.5
            )
        
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF00FF") # Magenta

        # Show angle theta
        p1 = circle.point_at_angle(0)
        p2 = circle.point_at_angle(theta_val)
        
        radial_line1 = Line(center, p1, color=WHITE, stroke_width=2)
        radial_line2 = Line(center, p2, color=WHITE, stroke_width=2)
        
        angle_arc = Arc(radius=0.5, start_angle=0, angle=theta_val, arc_center=center, color="#FF00FF")
        theta_label = MathTex(r"\theta", color="#FF00FF", font_size=24)
        # Position label near arc
        theta_label.move_to(center + 0.8 * np.array([np.cos(theta_val/2), np.sin(theta_val/2), 0]))

        # Asset Integration: Block M
        block_m = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE_B)
        self.place_at_grid(block_m, "A6", scale_factor=0.5)
        label_m = MathTex("M", color=WHITE, font_size=24).next_to(block_m, UP, buff=0.1)

        self.play(
            Create(radial_line1), 
            Create(radial_line2), 
            Create(angle_arc), 
            Write(theta_label),
            FadeIn(block_m),
            Write(label_m)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Show mass M increasing -> theta decreasing
        new_theta = 15 * DEGREES
        
        # Update components
        new_p2 = circle.point_at_angle(new_theta)
        new_radial_line2 = Line(center, new_p2, color=WHITE, stroke_width=2)
        new_angle_arc = Arc(radius=0.5, start_angle=0, angle=new_theta, arc_center=center, color="#FF00FF")
        new_theta_label_pos = center + 0.8 * np.array([np.cos(new_theta/2), np.sin(new_theta/2), 0])

        self.play(
            Transform(radial_line2, new_radial_line2),
            Transform(angle_arc, new_angle_arc),
            theta_label.animate.move_to(new_theta_label_pos),
            state_point.animate.move_to(circle.point_at_angle(new_theta)),
            block_m.animate.scale(1.5),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(WHITE) # Using White as requested in prompt "in white (#FFFFFF)"

        # Calculation display
        calc = MathTex(r"\text{Hits} = \frac{\pi}{\theta}", color=WHITE, font_size=32)
        # Apply Issue 41 fix: place_in_area('F3', 'F4')
        self.place_in_area(calc, 'F3', 'F4', scale_factor=0.8)
        
        self.play(Write(calc))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFFFF") # White

        # Highlight top half arc
        pi_arc = Arc(radius=2, start_angle=0, angle=PI, arc_center=center, color=WHITE, stroke_width=6)
        
        # Label for Pi radians
        pi_label = MathTex(r"\pi \text{ radians}", color=WHITE, font_size=28)
        # Apply Issue 42 fix: place_in_area('A3', 'B4')
        self.place_in_area(pi_label, 'A3', 'B4', scale_factor=0.8)

        self.play(Create(pi_arc), Write(pi_label))
        
        # Final point traversal to show it fits within Pi
        for i in range(2, 13): # 12 * 15 degrees = 180 degrees
            self.play(
                state_point.animate.move_to(circle.point_at_angle(i * new_theta)),
                run_time=0.2
            )

        self.wait(2)
