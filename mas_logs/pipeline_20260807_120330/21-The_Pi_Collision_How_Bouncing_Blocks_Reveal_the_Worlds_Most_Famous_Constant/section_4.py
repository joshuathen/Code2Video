from manim import *

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
        self.setup_layout("The Geometric Transformation", [
            "We scale the velocities to simplify the math.",
            "This transformation turns our energy ellipse into a circle.",
            "Now, every collision becomes a simple reflection."
        ])
        
        # Colors for stage matching
        color_line1 = BLUE_A
        color_line2 = GREEN_A
        color_line3 = YELLOW_A
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_line1))
        
        # Transformation equations
        eq_x = MathTex("x = v_1 \\sqrt{m_1}", color=color_line1)
        eq_y = MathTex("y = v_2 \\sqrt{m_2}", color=color_line1)
        eq_vgroup = VGroup(eq_x, eq_y).arrange(RIGHT, buff=0.8)
        
        # Issue 30: The transformation equations (eq_vgroup) are undersized, making poor use of the allocated horizontal space in row A.
        # Fix: Line 66: self.place_in_area(self.eq_vgroup, 'A2', 'A5', scale_factor=1.0)
        self.place_in_area(eq_vgroup, 'A2', 'A5', scale_factor=1.0)
        self.play(Write(eq_vgroup))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_line2))
        
        # Initial conservation equation
        energy_ellipse = MathTex("m_1 v_1^2 + m_2 v_2^2 = R^2", color=color_line2)
        # Issue 31: The energy equations have suboptimal scale factors, leading to inconsistent text sizing compared to other elements.
        # Fix: Line 75: self.place_in_area(self.energy_ellipse, 'B2', 'B5', scale_factor=1.0)
        self.place_in_area(energy_ellipse, 'B2', 'B5', scale_factor=1.0)
        self.play(Write(energy_ellipse))
        self.wait(1)
        
        # Transformed circular equation
        energy_circle = MathTex("x^2 + y^2 = R^2", color=color_line2)
        # Issue 31: Fix: Line 81: self.place_in_area(self.energy_circle, 'B2', 'B5', scale_factor=1.0)
        self.place_in_area(energy_circle, 'B2', 'B5', scale_factor=1.0)
        
        # Transform the formula
        self.play(ReplacementTransform(energy_ellipse, energy_circle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_line3))
        
        # Setup Axes
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3,
            y_length=3,
            axis_config={"include_tip": True, "color": GRAY},
            tips=False
        )
        x_lab = axes.get_x_axis_label("x").scale(0.6)
        y_lab = axes.get_y_axis_label("y").scale(0.6)
        axes_group = VGroup(axes, x_lab, y_lab)
        
        # Issue 29: An empty row at C creates a vertical gap that disconnects the transformation equations from the visual diagram.
        # Fix: Line 102: self.place_in_area(self.axes_group, 'C2', 'F5', scale_factor=1.0)
        self.place_in_area(axes_group, 'C2', 'F5', scale_factor=1.0)
        
        # The Green Circle
        # Calculate radius in scene units relative to axes
        radius_val = 1.2
        origin = axes.coords_to_point(0, 0)
        point_on_circle = axes.coords_to_point(radius_val, 0)
        circle_radius = np.linalg.norm(point_on_circle - origin)
        
        circle = Circle(radius=circle_radius, color="#00FF00")
        circle.move_to(axes.get_center())
        
        self.play(Create(axes_group), Create(circle))
        
        # Initial point on circle
        angle_tracker = ValueTracker(PI/6) # 30 degrees
        
        dot = Dot(color=color_line3)
        # Persistent updater for movement along the arc
        dot.add_updater(lambda d: d.move_to(axes.coords_to_point(
            radius_val * np.cos(angle_tracker.get_value()),
            radius_val * np.sin(angle_tracker.get_value())
        )))
        
        self.play(FadeIn(dot))
        
        # "Hopping" along the arc (simulating reflections)
        target_angles = [5*PI/6, 9*PI/6, 13*PI/6]
        
        for target in target_angles:
            self.play(
                angle_tracker.animate.set_value(target),
                run_time=1.2,
                rate_func=bezier([0, 0, 1, 1]) # Slightly snappy movement
            )
            # Collision indicator
            self.play(Flash(dot, color=color_line3, flash_radius=0.15, num_lines=8))
            self.wait(0.2)

        self.wait(2)
