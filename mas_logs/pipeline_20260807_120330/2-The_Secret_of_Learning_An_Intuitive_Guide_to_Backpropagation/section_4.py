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
        # Title and Lecture Lines from Storyboard
        title = "Visualizing the Gradient (The Slope)"
        lines = [
            "The Gradient shows the direction of the Error Hill.",
            "Positive slopes mean increasing the weight increases the error.",
            "Negative slopes mean increasing the weight decreases the error."
        ]
        self.setup_layout(title, lines)
        
        # Assets Paths
        HILL_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/hill.svg"
        BALL_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg"
        
        # Colors
        HILL_COLOR = "#00FF00"
        TANGENT_COLOR = "#FFFFFF"
        BALL_COLOR = "#FFFF00"
        
        # Helper: Parabola to represent the loss landscape (Valley)
        # Center of Area B2-E5 is (3.0, -0.3)
        # Path: y = (x-3)^2 - 1.8
        def parabola_func(x):
            return (x - 3.0)**2 - 1.8
        
        def get_tangent_points(x_val):
            length = 0.6
            slope = 2 * (x_val - 3.0)
            # direction vector
            dx = length / np.sqrt(1 + slope**2)
            dy = slope * dx
            p = np.array([x_val, parabola_func(x_val), 0])
            return [p - np.array([dx, dy, 0]), p + np.array([dx, dy, 0])]

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.lecture[0].set_color(HILL_COLOR)
        
        # Create hill from Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/hill.svg]
        hill = SVGMobject(HILL_ASSET).set_color(HILL_COLOR)
        # Ensure the SVG is sized to fit the area B2 to E5
        self.place_in_area(hill, "B2", "E5", scale_factor=1.5)
        
        # Labels for the axis
        # Resolve Issue 34: weight_label at F4, scale 0.8
        weight_label = Text("Weight", font_size=20, color=WHITE)
        self.place_at_grid(weight_label, "F4", scale_factor=0.8)
        
        # Resolve Issue 33: error_label at C2, scale 0.8
        error_label = Text("Error", font_size=20, color=WHITE).rotate(90*DEGREES)
        self.place_at_grid(error_label, "C2", scale_factor=0.8)
        
        self.play(DrawBorderThenFill(hill), Write(weight_label), Write(error_label), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.lecture[1].set_color(TANGENT_COLOR)
        
        # Ball from Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg]
        ball = SVGMobject(BALL_ASSET).set_color(BALL_COLOR)
        ball_pos = ValueTracker(4.2) # Start on the right side (positive slope)
        
        # Setup Ball with updater to follow parabola
        ball.add_updater(lambda m: m.move_to(np.array([ball_pos.get_value(), parabola_func(ball_pos.get_value()), 0])))
        ball.scale(0.3) # Initial scaling for the ball
        
        # Tangent line at starting position
        pts = get_tangent_points(4.2)
        tangent = Line(pts[0], pts[1], color=TANGENT_COLOR)
        tangent.add_updater(lambda m: m.set_points_as_corners(get_tangent_points(ball_pos.get_value())))
        
        self.play(FadeIn(ball), Create(tangent))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.lecture[2].set_color(TANGENT_COLOR)
        
        # Animate the ball moving down the slope toward the local minimum (x=3.0)
        # First show negative slope side (x=1.8)
        self.play(ball_pos.animate.set_value(1.8), run_time=2, rate_func=linear)
        self.wait(0.5)
        
        # Finally settle at minimum
        self.play(ball_pos.animate.set_value(3.0), run_time=1.5, rate_func=slow_into)
        self.wait(2)
        
        # Cleanup updaters
        ball.clear_updaters()
        tangent.clear_updaters()
