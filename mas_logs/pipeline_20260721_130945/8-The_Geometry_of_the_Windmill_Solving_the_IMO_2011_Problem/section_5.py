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

class Section5Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title_text = "Step-by-Step Visualization (3-Point Case)"
        lecture_lines = [
            "Start with the laser pivoting at Point A.",
            "It rotates and hits Point B, its new pivot.",
            "Next, it hits Point C and continues spinning.",
            "After 180 degrees, the line returns to Point A.",
            "Notice how the sides have now completely swapped."
        ]
        
        # Colors
        color_a = "#FFD700"  # Gold
        color_b = "#00BFFF"  # DeepSkyBlue
        color_c = "#32CD32"  # LimeGreen
        color_line_left = BLUE
        color_line_right = RED
        color_path = "#555555"
        color_swap = "#FF69B4"

        self.setup_layout(title_text, lecture_lines)

        # Positions (Updated per Issue 34 and 36)
        pos_a = self.grid["D6"]
        pos_b = self.grid["E5"]
        pos_c = self.grid["D4"]

        # Stars and Labels
        star_a = Star(n=5, outer_radius=0.15, inner_radius=0.07, color=color_a, fill_opacity=1)
        star_b = Star(n=5, outer_radius=0.15, inner_radius=0.07, color=color_b, fill_opacity=1)
        star_c = Star(n=5, outer_radius=0.15, inner_radius=0.07, color=color_c, fill_opacity=1)
        
        # Grid positioning per Issue 34
        self.place_at_grid(star_a, "D6")
        self.place_at_grid(star_b, "E5")
        self.place_at_grid(star_c, "D4")

        label_a = Text("A", font_size=20, color=color_a).next_to(star_a, UR, buff=0.1).scale(0.8)
        label_b = Text("B", font_size=20, color=color_b).next_to(star_b, DOWN, buff=0.1).scale(0.8)
        label_c = Text("C", font_size=20, color=color_c).next_to(star_c, UL, buff=0.1).scale(0.8)

        # Laser Setup
        # angle_tracker starting at -180 (pointing Left/Right along the horizontal)
        angle_tracker = ValueTracker(-180 * DEGREES)
        pivot_dot = Dot(pos_a).set_opacity(0)
        
        # Two halves to show the swap
        laser_left = Line(LEFT * 3, ORIGIN, color=color_line_left, stroke_width=4)
        laser_right = Line(ORIGIN, RIGHT * 3, color=color_line_right, stroke_width=4)
        laser = VGroup(laser_left, laser_right)

        def update_laser(m):
            p = pivot_dot.get_center()
            ang = angle_tracker.get_value()
            vec_left = rotate_vector(LEFT * 3, ang + 180 * DEGREES)
            vec_right = rotate_vector(RIGHT * 3, ang + 180 * DEGREES)
            m[0].set_points_as_corners([p + vec_left, p])
            m[1].set_points_as_corners([p, p + vec_right])

        laser.add_updater(update_laser)

        # === Animation for Lecture Line 1 ===
        self.play(
            self.lecture[0].animate.set_color(color_a),
            FadeIn(star_a), FadeIn(star_b), FadeIn(star_c),
            Write(label_a), Write(label_b), Write(label_c),
            FadeIn(laser)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rotate CW from -180 to -225 to hit B (Vector AB angle is -135/225)
        self.play(
            self.lecture[1].animate.set_color(color_b),
            angle_tracker.animate.set_value(-225 * DEGREES),
            run_time=2,
            rate_func=linear
        )
        # Shift pivot to B
        pivot_dot.move_to(pos_b)
        self.play(Indicate(star_b, color=color_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Rotate CW from -225 to -315 to hit C (Vector BC angle is 135/ -225 / -315)
        # Slope of BC is -1, so angle -45 or 135. CW from -225 brings us to -315 (which is 45/225 line)
        self.play(
            self.lecture[2].animate.set_color(color_c),
            angle_tracker.animate.set_value(-315 * DEGREES),
            run_time=2,
            rate_func=linear
        )
        # Shift pivot to C
        pivot_dot.move_to(pos_c)
        self.play(Indicate(star_c, color=color_c))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Rotate CW from -315 to -360 (returning to horizontal orientation)
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            angle_tracker.animate.set_value(-360 * DEGREES),
            run_time=1.5,
            rate_func=linear
        )
        # Shift pivot back to A (conceptual return in cycle)
        pivot_dot.move_to(pos_a)
        
        # Show pivot path (triangle ABC)
        path_pts = [pos_a, pos_b, pos_c, pos_a]
        pivot_path = VMobject(color=color_path)
        pivot_path.set_points_as_corners(path_pts)
        dashed_path = DashedVMobject(pivot_path, num_dashes=30)
        
        self.play(Create(dashed_path))
        self.play(Indicate(star_a, color=color_a))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Sides have swapped. Position per Issue 35.
        swap_text = Text("Sides Swapped", font_size=24, color=color_swap)
        self.place_at_grid(swap_text, "B5", scale_factor=0.8)

        self.play(
            self.lecture[4].animate.set_color(color_swap),
            Write(swap_text),
            Indicate(laser_left, color=color_line_left),
            Indicate(laser_right, color=color_line_right)
        )
        self.wait(3)
