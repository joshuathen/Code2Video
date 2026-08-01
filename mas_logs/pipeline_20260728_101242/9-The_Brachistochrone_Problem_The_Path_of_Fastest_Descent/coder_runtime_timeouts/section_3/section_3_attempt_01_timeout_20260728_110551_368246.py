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
        self.setup_layout("The Physics: Energy to Velocity", [
            "In gravity, potential energy converts to kinetic energy.",
            "Velocity depends on the vertical distance fallen.",
            "This links depth directly to the speed of travel."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Wire: a downward curve representing the path
        # A parabolic curve is used to demonstrate the principle
        wire_path = ParametricFunction(
            lambda t: np.array([t, -0.25 * t**2, 0]),
            t_range=[0, 4],
            color="#888888"
        )
        # Position the wire in the grid area
        self.place_in_area(wire_path, "B2", "E5", scale_factor=0.8)
        
        # The bead that will slide down
        bead = Dot(color="#FFFFFF", radius=0.1)
        bead.move_to(wire_path.get_start())
        
        # Energy labels
        pe_label = Text("PE", font_size=20, color=WHITE)
        self.place_at_grid(pe_label, "B1", scale_factor=1.0)
        pe_label.shift(DOWN * 0.3)
        
        ke_label = Text("KE", font_size=20, color=WHITE)
        # KE label will be added and updated later
        
        # Highlight lecture line 1
        self.lecture[0].set_color(WHITE)
        
        self.play(
            Create(wire_path),
            FadeIn(bead),
            Write(pe_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.lecture[0].set_color(GREY)
        self.lecture[1].set_color("#FF00FF") # Magenta matches the vertical bar
        
        t_tracker = ValueTracker(0)
        start_y = wire_path.get_start()[1]
        
        # Vertical bar representing height 'y'
        y_bar = Line(color="#FF00FF", stroke_width=6)
        def update_y_bar(m):
            curr_pos = bead.get_center()
            # Position the bar to the left of the bead
            m.put_start_and_end_on(
                np.array([curr_pos[0] - 0.4, start_y, 0]),
                np.array([curr_pos[0] - 0.4, curr_pos[1], 0])
            )
        y_bar.add_updater(update_y_bar)
        
        y_text = MathTex("y", color="#FF00FF").scale(0.8)
        y_text.add_updater(lambda m: m.next_to(y_bar, LEFT, buff=0.1))
        
        # Updater to bind bead to the curve via the tracker
        bead.add_updater(lambda m: m.move_to(wire_path.point_from_proportion(t_tracker.get_value()/4)))

        self.play(
            Create(y_bar),
            FadeIn(y_text),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.lecture[1].set_color(GREY)
        self.lecture[2].set_color("#00FFFF") # Cyan matches the velocity formula
        
        # Physics formula v = sqrt(2gy)
        formula = MathTex("v = \\sqrt{2gy}", color="#00FFFF")
        self.place_at_grid(formula, "A3", scale_factor=1.2)
        formula.shift(LEFT * 1.0) # Adjust to avoid crowding
        
        # Velocity arrow representing the vector
        v_arrow = Arrow(color="#00FFFF", buff=0, stroke_width=4)
        def update_v_arrow(m):
            t = t_tracker.get_value()
            if t < 0.05:
                # Keep it invisible at start
                m.put_start_and_end_on(bead.get_center(), bead.get_center())
                return
            
            p1 = bead.get_center()
            # Tangent approximation using a small step forward
            target_t = min(3.99, t + 0.1)
            p2 = wire_path.point_from_proportion(target_t/4)
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            
            if norm > 0:
                # Physics magnitude: v is proportional to sqrt(depth)
                dy = max(0, start_y - p1[1])
                velocity_mag = np.sqrt(2 * 9.8 * dy) * 0.15 # 0.15 is a visual scaling factor
                m.put_start_and_end_on(p1, p1 + (direction / norm) * velocity_mag)
        
        v_arrow.add_updater(update_v_arrow)
        
        # KE label follows the bead
        ke_label.add_updater(lambda m: m.next_to(bead, DOWN, buff=0.2))

        self.play(
            Write(formula),
            FadeIn(v_arrow),
            FadeIn(ke_label),
            run_time=1.5
        )
        
        # Perform the descent animation
        # ease_in_sine mimics the acceleration of gravity
        self.play(
            t_tracker.animate.set_value(4),
            run_time=5,
            rate_func=rate_functions.ease_in_sine
        )
        
        self.wait(2)
