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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        title_text = "Prerequisite: The Unit Square of Probability"
        lecture_lines = [
            "Imagine a square representing all possible outcomes.",
            "The total area is exactly one.",
            "Specific events are sub-areas within this square."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Defined Colors
        COLOR_BLUE = "#3498DB"
        COLOR_WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Draw a 1x1 square (scaled to 3 units for visibility in the 6x6 grid)
        unit_square = Square(side_length=3.0, stroke_color=COLOR_WHITE, stroke_width=2)
        # Fix Issue 27: Adjusted positioning for better vertical balance
        self.place_in_area(unit_square, 'A2', 'D5')
        
        self.play(
            Create(unit_square),
            self.lecture[0].animate.set_color(COLOR_WHITE),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The total area is exactly one.
        area_label = Text("Total Area = 1", font_size=24, color=COLOR_WHITE)
        # Fix Issue 28: Position label closer to the square and scaled down
        self.place_in_area(area_label, 'E2', 'E5', scale_factor=0.8)
        
        self.play(
            Write(area_label),
            self.lecture[1].animate.set_color(COLOR_WHITE),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Specific events are sub-areas. 
        # Fill a vertical strip from the left side, with width 0.2 (0.2 * 3.0 = 0.6)
        strip_width = 3.0 * 0.2
        blue_strip = Rectangle(
            width=strip_width, 
            height=3.0, 
            fill_color=COLOR_BLUE, 
            fill_opacity=0.7, 
            stroke_width=0
        )
        # Align strip to the left inside the unit square
        blue_strip.align_to(unit_square, LEFT)
        blue_strip.align_to(unit_square, UP)

        # Add text labels 'P(A) = 0.2' and 'P(Not A) = 0.8'
        # Fix Issue 29: Use grid methods for precise positioning and reduced clutter
        label_p_a = Text("P(A) = 0.2", font_size=18, color=COLOR_WHITE)
        self.place_at_grid(label_p_a, 'C2', scale_factor=0.5)
        
        label_p_not_a = Text("P(Not A) = 0.8", font_size=18, color=COLOR_WHITE)
        self.place_in_area(label_p_not_a, 'C3', 'C5', scale_factor=0.5)

        self.play(
            FadeIn(blue_strip),
            Write(label_p_a),
            Write(label_p_not_a),
            self.lecture[2].animate.set_color(COLOR_BLUE),
            run_time=2
        )
        self.wait(2)
