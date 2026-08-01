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
        # Layout setup
        title = "Guided Practice: The Circular Track"
        lecture_lines = [
            "This circular track is defined by x squared plus y squared.",
            "Dash the Cheetah runs along the circle's edge. [Asset: Dash the Cheetah]",
            "Differentiating both sides gives 2x plus 2y dy/dx.",
            "Solve for dy/dx to find our general slope formula.",
            "At point (3,4), the cheetah's slope is negative three-fourths."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Draw a circle representing x^2 + y^2 = 25 and label it in #FFFFFF.
        self.lecture[0].set_color(WHITE)
        circle = Circle(radius=1.5, color=WHITE)
        self.place_in_area(circle, "C1", "E3")
        
        circle_eq = Text("x^2 + y^2 = 25", font_size=24, color=WHITE)
        self.place_at_grid(circle_eq, "B2", scale_factor=0.8)
        
        self.play(Create(circle))
        self.play(Write(circle_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place a point at (3, 4) and bring in [Asset: Dash the Cheetah] at that location.
        # Dash the Cheetah runs along the circle's edge.
        self.lecture[1].set_color(WHITE)
        
        # Point (3,4) corresponds to angle arctan2(4, 3)
        angle_34 = np.arctan2(4, 3)
        dash_pos = circle.point_at_angle(angle_34)
        
        # Cheetah Asset
        try:
            dash_cheetah = ImageMobject("Dash_the_Cheetah.png").scale(0.3).move_to(dash_pos)
        except:
            dash_dot = Dot(dash_pos, color=ORANGE)
            cheetah_label = Text("Dash", font_size=16).next_to(dash_dot, UR, buff=0.1)
            dash_cheetah = VGroup(dash_dot, cheetah_label)
        
        self.play(FadeIn(dash_cheetah))
        
        # Orbit animation (Running along the edge)
        def orbiting_update(mob, alpha):
            current_angle = angle_34 + alpha * 2 * PI
            new_pos = circle.point_at_angle(current_angle)
            mob.move_to(new_pos)

        self.play(
            UpdateFromAlphaFunc(dash_cheetah, orbiting_update),
            run_time=3,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show the differentiation steps: 2x + 2y(dy/dx) = 0, highlighting the Chain Rule result in #FFFF00.
        self.lecture[2].set_color("#FFFF00")
        
        # Issue 40: scale factor and area adjustment to prevent clipping
        diff_eq_start = Text("d/dx(x^2 + y^2) = d/dx(25)", font_size=24, color=WHITE)
        self.place_in_area(diff_eq_start, "B4", "B6", scale_factor=0.6)
        
        # Issue 41: move to C4 for better layout
        diff_eq_result = VGroup(
            Text("2x + ", font_size=24, color=WHITE),
            Text("2y dy/dx", font_size=24, color="#FFFF00"), 
            Text(" = 0", font_size=24, color=WHITE)
        ).arrange(RIGHT, buff=0.1)
        self.place_at_grid(diff_eq_result, "C4", scale_factor=0.8)
        
        self.play(Write(diff_eq_start))
        self.wait(0.5)
        self.play(FadeIn(diff_eq_result, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Isolate dy/dx to show the formula dy/dx = -x/y in #00FFFF.
        self.lecture[3].set_color("#00FFFF")
        
        isolation_eq = Text("dy/dx = -x/y", font_size=24, color="#00FFFF")
        self.place_at_grid(isolation_eq, "D4", scale_factor=0.8)
        
        self.play(Write(isolation_eq))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Calculate the slope at (3, 4) as -3/4 and draw a tangent line at the cheetah's position.
        self.lecture[4].set_color(WHITE)
        
        # Ensure dash is at (3,4) for final calculation
        dash_cheetah.move_to(dash_pos)
        
        # Issue 42: move slope_calc to E4
        slope_calc = Text("m = -3/4", font_size=24, color="#00FFFF")
        self.place_at_grid(slope_calc, "E4", scale_factor=0.7)
        
        # Tangent line vector
        tangent_vec = np.array([-4, 3, 0]) / 5
        tangent_line = Line(
            start=dash_pos + 1.2 * tangent_vec,
            end=dash_pos - 1.2 * tangent_vec,
            color="#00FFFF"
        )

        self.play(Write(slope_calc))
        self.play(Create(tangent_line))
        self.wait(2)
