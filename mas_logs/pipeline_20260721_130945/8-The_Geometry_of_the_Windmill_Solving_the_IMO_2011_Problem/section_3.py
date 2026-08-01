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
        title = "The Core Rule: The Hand-off"
        lecture_lines = [
            "- The laser rotates around its current pivot point.",
            "- It continues until it hits another star.",
            "- Instantly, that new star becomes the pivot.",
            "- The rotation continues from this new center.",
            "- This \"hand-off\" creates the windmill motion."
        ]
        
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_CYAN = "#00FFFF"
        COLOR_YELLOW = "#FFFF00"
        COLOR_WHITE = "#FFFFFF"

        # Star positions on grid
        # Avoid Row A (Title) and Column 6 (Clipping)
        # Resolved Issue 29: star_b to B5
        pos_a = self.grid["C3"]
        pos_b = self.grid["B5"]
        pos_c = self.grid["E5"]

        # Stars
        star_a = Star(n=5, outer_radius=0.2, inner_radius=0.1, color=COLOR_YELLOW, fill_opacity=1)
        star_b = Star(n=5, outer_radius=0.2, inner_radius=0.1, color=COLOR_WHITE, fill_opacity=1)
        star_c = Star(n=5, outer_radius=0.2, inner_radius=0.1, color=COLOR_WHITE, fill_opacity=1)

        self.place_at_grid(star_a, "C3")
        self.place_at_grid(star_b, "B5")
        self.place_at_grid(star_c, "E5")

        # Labels
        label_a = Text("A", font_size=20, color=WHITE)
        label_b = Text("B", font_size=20, color=WHITE)
        label_c = Text("C", font_size=20, color=WHITE)
        
        # Place labels near stars (within 1 grid unit)
        # Resolved Issue 28: label_a to D3
        # Resolved Issue 29: label_b to A5
        # Resolved Issue 30: label_c to F5
        self.place_at_grid(label_a, "D3", scale_factor=0.8)
        self.place_at_grid(label_b, "A5", scale_factor=0.8)
        self.place_at_grid(label_c, "F5", scale_factor=0.8)

        # Line setup
        # The line is long enough to cover the grid area
        angle_tracker = ValueTracker(60 * DEGREES)
        pivot_tracker = ValueTracker(0) # 0: A, 1: B, 2: C
        
        # Line logic: must pass through the current pivot
        # We use a long line (length 8)
        line = Line(start=LEFT*5, end=RIGHT*5, color=COLOR_CYAN, stroke_width=4)
        
        def line_updater(obj):
            p_idx = int(pivot_tracker.get_value())
            if p_idx == 0:
                center = pos_a
            elif p_idx == 1:
                center = pos_b
            else:
                center = pos_c
            
            angle = angle_tracker.get_value()
            obj.set_angle(angle)
            obj.move_to(center)

        line.add_updater(line_updater)

        # === Animation for Lecture Line 1 ===
        # "- The laser rotates around its current pivot point."
        self.lecture[0].set_color(COLOR_CYAN)
        self.add(star_a, star_b, star_c, label_a, label_b, label_c)
        self.play(Create(line), run_time=1)
        
        # Start rotating CW
        self.play(angle_tracker.animate.set_value(45 * DEGREES), run_time=1.5, rate_func=linear)

        # === Animation for Lecture Line 2 ===
        # "- It continues until it hits another star."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_CYAN)
        
        # Calculate angle when line hits Star B (pos_b) from Star A (pos_a)
        # Vector AB = pos_b - pos_a = [2, 1, 0]
        vec_ab = pos_b - pos_a
        target_angle_ab = np.arctan2(vec_ab[1], vec_ab[0]) 
        
        self.play(angle_tracker.animate.set_value(target_angle_ab), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # "- Instantly, that new star becomes the pivot."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_YELLOW)
        
        # Swap pivot to B
        self.play(
            star_a.animate.set_color(COLOR_WHITE),
            star_b.animate.set_color(COLOR_YELLOW),
            Indicate(star_b, color=COLOR_YELLOW),
            pivot_tracker.animate.set_value(1),
            run_time=1
        )

        # === Animation for Lecture Line 4 ===
        # "- The rotation continues from this new center."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_CYAN)
        
        # Calculate angle when line hits Star C (pos_c) from Star B (pos_b)
        # Vector BC = pos_c - pos_b = [0, -3, 0]
        vec_bc = pos_c - pos_b
        target_angle_bc = np.arctan2(vec_bc[1], vec_bc[0]) # -90 deg
        
        self.play(angle_tracker.animate.set_value(target_angle_bc), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 5 ===
        # "- This \"hand-off\" creates the windmill motion."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_YELLOW)
        
        # Swap pivot to C
        self.play(
            star_b.animate.set_color(COLOR_WHITE),
            star_c.animate.set_color(COLOR_YELLOW),
            Indicate(star_c, color=COLOR_YELLOW),
            pivot_tracker.animate.set_value(2),
            run_time=1
        )
        
        # Rotate a bit more to show the hand-off is complete
        self.play(angle_tracker.animate.set_value(target_angle_bc - 30 * DEGREES), run_time=1.5, rate_func=linear)
        self.wait(1)
        self.lecture[4].set_color(WHITE)
