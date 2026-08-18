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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title = "Prerequisite 1: Conservation of Energy"
        lecture_lines = [
            "Energy conservation dictates speed depends only on vertical drop.",
            "The formula v equals square root of 2 g y.",
            "Falling further down translates directly into moving faster."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        BEAD_COLOR = "#FFFFFF"
        VELOCITY_COLOR = "#00FFFF"
        FORMULA_COLOR = "#FFFF00"
        PATH_COLOR = "#888888"

        # === Animation for Lecture Line 1 ===
        # Energy conservation dictates speed depends only on vertical drop.
        self.lecture[0].set_color(BEAD_COLOR)
        
        # Define the wire path (a simple curve from A2 to F5)
        # Using CubicBezier to simulate a quadratic path via degree elevation
        start_point = self.grid["A2"]
        end_point = self.grid["F5"]
        control_point = self.grid["F2"]
        
        # Quadratic Bezier to Cubic Bezier conversion points
        cp1 = (start_point + 2 * control_point) / 3
        cp2 = (end_point + 2 * control_point) / 3
        
        path = CubicBezier(start_point, cp1, cp2, end_point, color=PATH_COLOR)
        
        # Label point A
        label_a = Text("A", font_size=20, color=WHITE).next_to(start_point, UP, buff=0.1)
        
        # Bead at the start
        bead = Dot(point=start_point, color=BEAD_COLOR, radius=0.1)
        
        self.play(Create(path), Write(label_a), FadeIn(bead))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The formula v equals square root of 2 g y.
        self.lecture[1].set_color(FORMULA_COLOR)
        
        formula = MathTex(r"v = \sqrt{2gy}", color=FORMULA_COLOR)
        # Resolved Issue 27: Position changed from B4 to B5 to reduce clutter near Point A
        self.place_at_grid(formula, "B5", scale_factor=1.2)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Falling further down translates directly into moving faster.
        self.lecture[2].set_color(VELOCITY_COLOR)
        
        # Track progress along the path
        t_tracker = ValueTracker(0)
        
        # Velocity vector (arrow) attached to the bead
        y_start = start_point[1]
        
        def get_bead_pos():
            return path.point_from_proportion(t_tracker.get_value())
        
        # Persistent Arrow mobject
        velocity_vector = Arrow(
            start=start_point, 
            end=start_point + RIGHT * 0.1, 
            color=VELOCITY_COLOR, 
            buff=0,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.3
        )
        
        def update_velocity_vector(obj):
            pos = get_bead_pos()
            y_current = pos[1]
            y_drop = max(0, y_start - y_current)
            # v = sqrt(2gy). Assuming g=9.8 for realistic-looking scaling
            speed = np.sqrt(2 * 9.8 * y_drop)
            # Scaling speed for visualization
            vector_len = speed * 0.15
            
            # Direction is tangent to path
            epsilon = 0.001
            t = t_tracker.get_value()
            if t < 1 - epsilon:
                tangent = path.point_from_proportion(t + epsilon) - pos
            else:
                tangent = pos - path.point_from_proportion(t - epsilon)
            
            norm = np.linalg.norm(tangent)
            if norm > 0:
                direction = tangent / norm
            else:
                direction = RIGHT
                
            obj.put_start_and_end_on(pos, pos + direction * vector_len)

        velocity_vector.add_updater(update_velocity_vector)
        bead.add_updater(lambda m: m.move_to(get_bead_pos()))
        
        # Speedometer (Numeric display)
        speed_label = Text("Speed:", font_size=18, color=WHITE)
        speed_value = DecimalNumber(0, num_decimal_places=2, color=VELOCITY_COLOR, font_size=24)
        speedometer = VGroup(speed_label, speed_value).arrange(RIGHT, buff=0.2)
        # Resolved Issue 28: Position changed from E2 to E5 to better connect with animation
        self.place_at_grid(speedometer, "E5", scale_factor=1.0)
        
        def update_speed_value(obj):
            pos = get_bead_pos()
            y_drop = max(0, y_start - pos[1])
            speed = np.sqrt(2 * 9.8 * y_drop)
            obj.set_value(speed)

        speed_value.add_updater(update_speed_value)

        self.add(velocity_vector, speedometer)
        # rush_into makes the acceleration visible as the bead drops
        self.play(t_tracker.animate.set_value(1), run_time=5, rate_func=rush_into)
        self.wait(2)

        # Cleanup updaters
        velocity_vector.clear_updaters()
        bead.clear_updaters()
        speed_value.clear_updaters()
