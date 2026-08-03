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
        self.setup_layout("Visualizing Phase Space: From Ellipse to Circle", [
            "- We can graph the velocities in phase space.",
            "- Rescaling the axes turns an ellipse into a circle.",
            "- Now, the system state is a single moving point."
        ])
        
        # Initial lecture colors
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Create axes for velocity space
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            axis_config={"include_tip": True, "color": GREY},
            x_length=5,
            y_length=3
        )
        x_label = MathTex("v_1", color=WHITE, font_size=24).next_to(axes.x_axis.get_end(), DOWN)
        y_label = MathTex("v_2", color=WHITE, font_size=24).next_to(axes.y_axis.get_end(), LEFT)
        
        # Plot ellipse representing the energy equation
        ellipse = Ellipse(width=4.5, height=1.8, color=WHITE)
        
        # Position axes and elements in the right area
        # Fix for Issue 31: adjusted grid area and scale factor to avoid crowding
        axes_group = VGroup(axes, x_label, y_label, ellipse)
        self.place_in_area(axes_group, "A2", "F6", scale_factor=0.8)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(ellipse))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(BLUE_A)
        
        # Rescaling labels for mass-scaled coordinates
        new_x_label = MathTex(r"\sqrt{M}v_1", color=BLUE_A, font_size=24).move_to(x_label)
        new_y_label = MathTex(r"\sqrt{m}v_2", color=BLUE_A, font_size=24).move_to(y_label)
        
        # Morph ellipse into a unit circle in rescaled phase space
        # Circle radius chosen for visual balance within axes
        circle = Circle(radius=1.5, color=BLUE_A).move_to(ellipse.get_center())
        
        self.play(
            Transform(ellipse, circle),
            Transform(x_label, new_x_label),
            Transform(y_label, new_y_label),
            run_time=2.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(YELLOW)
        
        # Mark system state as a point on the circle
        # We use the 'circle' mobject for geometric position reference.
        dot = Dot(color=YELLOW)
        dot.move_to(circle.point_at_angle(PI/4))
        
        # Continuous rotation using ValueTracker and updater for efficiency
        angle_tracker = ValueTracker(PI/4)
        dot.add_updater(lambda d: d.move_to(circle.point_at_angle(angle_tracker.get_value())))
        
        self.play(FadeIn(dot))
        # Animate the point moving along the circle boundary
        self.play(angle_tracker.animate.set_value(PI/4 + 2*PI), run_time=4, rate_func=linear)
        
        # Flash the circle boundary in yellow to emphasize the symmetry/path
        flash_circle = circle.copy().set_color(YELLOW).set_stroke(width=6)
        self.play(FadeIn(flash_circle, scale=1.05), run_time=0.4)
        self.play(FadeOut(flash_circle), run_time=0.4)
        
        self.wait(3)
