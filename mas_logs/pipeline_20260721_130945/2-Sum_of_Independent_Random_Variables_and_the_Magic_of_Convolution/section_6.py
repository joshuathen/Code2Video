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
        # Teaching Content from shared state
        title_text = "Summary and Real-World Application"
        lecture_lines = [
            "Convolution combines independent uncertainties into one.",
            "This 'Flip-Shift-Integrate' workflow works for many fields.",
            "Adding variables often creates smoother, bell-shaped curves."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors aligned with lecture line requirements
        COLOR_1 = WHITE
        COLOR_2 = "#00FF00"  # Green
        COLOR_3 = "#FFD700"  # Gold/Yellow
        
        # === Animation for Lecture Line 1 ===
        # Step: List steps 'Flip, Shift, Integrate' in a vertical column (#FFFFFF).
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        flip_text = Text("1. Flip", font_size=24, color=COLOR_1)
        shift_text = Text("2. Shift", font_size=24, color=COLOR_1)
        integrate_text = Text("3. Integrate", font_size=24, color=COLOR_1)
        
        workflow = VGroup(flip_text, shift_text, integrate_text).arrange(DOWN, buff=0.5)
        # Fix for Issue 40: Avoiding overlap and using provided grid coordinates
        self.place_in_area(workflow, 'A2', 'C5', scale_factor=0.9)
        
        self.play(Write(workflow))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Step: Animate Cargo robot moving past the triangle peak (#00FF00).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_2),
            FadeOut(workflow)
        )
        
        # Triangle distribution (sum of uniform variables)
        triangle = Polygon(
            [-1.5, -1, 0], [0, 1, 0], [1.5, -1, 0], 
            color=COLOR_2, stroke_width=4
        )
        triangle.set_fill(COLOR_2, opacity=0.3)
        # Fix for Issue 41: Positioning and scaling for better visibility
        self.place_in_area(triangle, 'D2', 'F6', scale_factor=1.1)
        
        # Cargo robot (procedural visual group as no SVG asset was specified in storyboard)
        robot_body = Square(side_length=0.4, color=COLOR_2, fill_opacity=1)
        # Eye positioned on the top right
        robot_eye = Dot(radius=0.05, color=BLACK).move_to(robot_body.get_critical_point(UP) + DOWN*0.1 + RIGHT*0.1)
        cargo_robot = VGroup(robot_body, robot_eye)
        
        # Define movement path based on triangle boundaries (L008 compliant)
        # We'll make the robot move across the x-axis near the triangle's base level
        start_pt = triangle.get_critical_point(LEFT) + LEFT * 0.5 + UP * 0.2
        end_pt = triangle.get_critical_point(RIGHT) + RIGHT * 0.5 + UP * 0.2
        cargo_robot.move_to(start_pt)
        
        self.play(Create(triangle))
        
        # Use ValueTracker for movement (L010 instruction)
        robot_x = ValueTracker(start_pt[0])
        cargo_robot.add_updater(lambda m: m.set_x(robot_x.get_value()))
        self.add(cargo_robot)
        
        # Animate robot moving past the peak
        self.play(robot_x.animate.set_value(end_pt[0]), run_time=3, rate_func=linear)
        self.wait(1)
        cargo_robot.clear_updaters()
        
        # === Animation for Lecture Line 3 ===
        # Step: Morph triangle distribution into a smooth bell curve (#FFD700).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_3),
            FadeOut(cargo_robot)
        )
        
        # Gaussian curve approximation as a Polygon for smooth Transform animation
        # We build the bell curve to have similar dimensions to the triangle for a clean morph
        bell_points = []
        # Bell curve defined on a similar range to the triangle base
        for x_coord in np.linspace(-1.5, 1.5, 50):
            # y = 2 * exp(-(x*2)^2) - 1 to approximate the triangle's height (2 units from -1 to 1)
            y_val = 2 * np.exp(-(x_coord * 1.5)**2) - 1
            bell_points.append([x_coord, y_val, 0])
        # Close the polygon base at y = -1
        bell_points.append([1.5, -1, 0])
        bell_points.append([-1.5, -1, 0])
        
        bell_poly = Polygon(*bell_points, color=COLOR_3, stroke_width=4)
        bell_poly.set_fill(COLOR_3, opacity=0.3)
        
        # Align with existing triangle and apply same scaling
        bell_poly.move_to(triangle.get_center())
        # The bell points were already scaled relative to triangle points, so we apply the 1.1 grid scale factor
        bell_poly.scale(1.1)
        
        self.play(
            Transform(triangle, bell_poly),
            run_time=2
        )
        self.wait(3)
