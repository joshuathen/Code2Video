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

class Section3Scene(TeachingScene):
    def construct(self):
        # === Fetching data ===
        title_text = "The Mechanism of the Pivot"
        lecture_lines = [
            "Watch the line sweep across the plane like radar.",
            "Upon impact, the center of rotation instantly changes.",
            "The clockwise motion remains constant throughout the dance."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # === Visual Assets ===
        # Points at grid locations requested in issues 26, 27, 28 (L002: Avoid Col 1)
        p1 = Dot(color="#FFFFFF")
        p2 = Dot(color="#FFFFFF")
        p3 = Dot(color="#FFFFFF")
        
        self.place_at_grid(p1, "C3") # Issue 26: Moved from C2 to C3 to avoid lecture obstruction
        self.place_at_grid(p2, "E5") # Issue 27: Moved from D4 to E5 for better spacing
        self.place_at_grid(p3, "B6") # Issue 28: Moved from B5 to B6 for better coverage
        
        # Labels scaled and positioned (L002)
        p1_label = Text("P1", font_size=20, color="#FFFFFF").scale(0.8).next_to(p1, UP, buff=0.1)
        p2_label = Text("P2", font_size=20, color="#FFFFFF").scale(0.8).next_to(p2, DOWN, buff=0.1)
        p3_label = Text("P3", font_size=20, color="#FFFFFF").scale(0.8).next_to(p3, RIGHT, buff=0.1)
        
        points_group = VGroup(p1, p2, p3, p1_label, p2_label, p3_label)
        self.add(points_group)
        
        # Animation state tracking
        state = {"pivot": p1}
        angle_tracker = ValueTracker(90 * DEGREES) # Start pointing UP
        
        # Primary rotating line (L027: Persistent mobject + updater)
        line = Line(start=LEFT*5, end=RIGHT*5, color="#00FFFF", stroke_width=3)
        
        def line_updater(obj):
            angle = angle_tracker.get_value()
            center = state["pivot"].get_center()
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            obj.set_points_as_corners([
                center - 5.0 * direction,
                center + 5.0 * direction
            ])
            
        line.add_updater(line_updater)
        
        # === Animation for Lecture Line 1 ===
        # Radar sweep like a beam
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.add(line)
        
        # Trail 1
        # L031: Use fill_opacity in constructor
        trail1 = Sector(
            radius=6,
            start_angle=90 * DEGREES,
            angle=0,
            color="#FFD700",
            fill_opacity=0.2,
            stroke_width=0
        ).move_to(p1.get_center())
        self.add(trail1)
        
        # Sector angle update for clockwise sweep
        trail1.add_updater(lambda m: m.set_angle(angle_tracker.get_value() - 90 * DEGREES))
        
        # Target angle to hit P2 from P1 (dx=2, dy=-2)
        target_angle_1 = np.arctan2(-2.0, 2.0) # -45 degrees
        
        self.play(angle_tracker.animate.set_value(target_angle_1), run_time=2.5, rate_func=linear)
        trail1.clear_updaters()
        self.wait(1.5) # Absorption time

        # === Animation for Lecture Line 2 ===
        # Instant shift and impact flash
        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            self.lecture[1].animate.set_color("#00FF00")
        )
        
        # Update state variable for the line updater
        state["pivot"] = p2
        self.play(Indicate(p2, color="#00FF00", scale_factor=1.8)) # L004: Indicate
        self.wait(2.0) # Absorption time

        # === Animation for Lecture Line 3 ===
        # Continued sweep
        self.play(
            self.lecture[1].animate.set_color("#FFFFFF"),
            self.lecture[2].animate.set_color("#FFD700")
        )
        
        # Trail 2 from P2
        trail2 = Sector(
            radius=6,
            start_angle=target_angle_1,
            angle=0,
            color="#FFD700",
            fill_opacity=0.2,
            stroke_width=0
        ).move_to(p2.get_center())
        self.add(trail2)
        
        trail2.add_updater(lambda m: m.set_angle(angle_tracker.get_value() - target_angle_1))
        
        # Continue sweep clockwise (e.g., 120 degrees more)
        target_angle_2 = target_angle_1 - 120 * DEGREES
        
        self.play(angle_tracker.animate.set_value(target_angle_2), run_time=2.5, rate_func=linear)
        trail2.clear_updaters()
        self.wait(1.5) # Absorption time

        # Final state cleanup
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(2)
