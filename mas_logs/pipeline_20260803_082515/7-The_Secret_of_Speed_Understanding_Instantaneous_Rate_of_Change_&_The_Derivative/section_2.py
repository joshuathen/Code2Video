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
        # Data from storyboard
        title = "Prerequisite Check: The Slope of a Line"
        lecture_lines = [
            "Rate of change is the steepness of a line.",
            "We calculate slope using rise divided by run.",
            "For a hiker's steady path, this slope never changes."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        COLOR_LINE = "#FFFFFF"
        COLOR_DOTS = "#00FF00"
        COLOR_TRIANGLE = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        # "Rate of change is the steepness of a line."
        self.play(self.lecture[0].animate.set_color(COLOR_LINE))
        
        # Draw a straight line from E1 to B5
        main_line = Line(
            start=self.grid["E1"],
            end=self.grid["B5"],
            color=COLOR_LINE,
            stroke_width=4
        )
        self.play(Create(main_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We calculate slope using rise divided by run."
        self.play(self.lecture[1].animate.set_color(COLOR_DOTS))
        
        # Two green dots at D2 and C4 (on the line)
        dot1 = Dot(point=self.grid["D2"], color=COLOR_DOTS)
        dot2 = Dot(point=self.grid["C4"], color=COLOR_DOTS)
        
        self.play(FadeIn(dot1), FadeIn(dot2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "For a hiker's steady path, this slope never changes."
        self.play(self.lecture[2].animate.set_color(COLOR_TRIANGLE))
        
        # Right triangle: D2 -> D4 -> C4
        # Points: D2 (bottom-left), D4 (bottom-right corner), C4 (top-right)
        corner = self.grid["D4"]
        
        run_line = Line(self.grid["D2"], corner, color=COLOR_TRIANGLE)
        rise_line = Line(corner, self.grid["C4"], color=COLOR_TRIANGLE)
        
        # Labels - Issues 25 and 26 addressed here
        run_label = Text("Run", font_size=18, color=COLOR_TRIANGLE)
        self.place_at_grid(run_label, "E4", scale_factor=1.0) # Updated from E3 (Issue 26)
        
        rise_label = Text("Rise", font_size=18, color=COLOR_TRIANGLE)
        self.place_at_grid(rise_label, "C6", scale_factor=1.0) # Updated from C5 (Issue 25)
        
        self.play(Create(run_line), Create(rise_line))
        self.play(Write(run_label), Write(rise_label))
        self.wait(2)
