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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup the layout with specific title and lines
        # Section 2: Prerequisite: The Intermediate Value Theorem (1D Logic)
        title_text = "Prerequisite: The Intermediate Value Theorem (1D Logic)"
        lecture_lines = [
            "An ant crawls from a valley to a peak.",
            "It must pass every height in between the two.",
            "This continuous path ensures a middle point exists."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Draw a smooth hill curve (#8B4513) from bottom-left to top-right.
        # Colors: Hill #8B4513
        hill_color = "#8B4513"
        # Using grid points E1, D2, C3, B4, A6 to define a monotonic smooth hill
        curve_points = [
            self.grid["E1"],
            self.grid["D2"],
            self.grid["C3"],
            self.grid["B4"],
            self.grid["A6"]
        ]
        hill_curve = VMobject(color=hill_color)
        hill_curve.set_points_as_corners(curve_points).make_smooth()
        
        valley_label = Text("Valley", font_size=18, color=hill_color)
        self.place_at_grid(valley_label, "F1")
        
        peak_label = Text("Peak", font_size=18, color=hill_color)
        self.place_at_grid(peak_label, "A5")

        self.play(
            Create(hill_curve),
            Write(valley_label),
            Write(peak_label),
            self.lecture[0].animate.set_color(hill_color),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate a small blue circle (#0000FF) as the ant moving from the start to the peak.
        # Colors: Ant #0000FF
        ant_color = "#0000FF"
        ant = Circle(radius=0.12, color=ant_color, fill_opacity=1)
        self.place_at_grid(ant, "E1") # Position at start of the path
        
        self.play(
            MoveAlongPath(ant, hill_curve),
            self.lecture[1].animate.set_color(ant_color),
            run_time=3,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw a dashed horizontal line at the hill's midpoint (#FFD700) and highlight intersection.
        # Colors: Midpoint line #FFD700
        mid_color = "#FFD700"
        # Midpoint row is C (y=0.2), which is the arithmetic mean height of E (y=-1.8) and A (y=2.2)
        midpoint_line = DashedLine(
            start=self.grid["C1"],
            end=self.grid["C6"],
            color=mid_color
        )
        
        midpoint_label = Text("Midpoint", font_size=18, color=mid_color)
        self.place_at_grid(midpoint_label, "C1")
        
        # Based on curve points, the intersection occurs at C3
        intersection_dot = Dot(self.grid["C3"], color=mid_color, radius=0.1)
        flash = Flash(intersection_dot, color=mid_color, flash_radius=0.3)

        self.play(
            Create(midpoint_line),
            Write(midpoint_label),
            self.lecture[2].animate.set_color(mid_color),
            run_time=1.5
        )
        self.play(
            FadeIn(intersection_dot),
            flash,
            run_time=1
        )
        self.wait(2)
