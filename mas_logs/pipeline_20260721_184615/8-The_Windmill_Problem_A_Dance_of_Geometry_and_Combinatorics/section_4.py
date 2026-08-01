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

class Section4Scene(TeachingScene):
    def construct(self):
        # Title and lecture lines from shared state
        title_text = "The Hidden Invariant: The 'Balanced' Line"
        lecture_lines = [
            "Color points on each side of the line differently.",
            "Pick a line splitting points into two equal sets.",
            "As the line pivots, points swap sides across it.",
            "The current pivot and the new pivot trade places.",
            "The total count on each side remains perfectly balanced."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Hexadecimal colors (L008)
        BLUE_CLR = "#0000FF" 
        RED_CLR = "#FF0000"
        WHITE_CLR = "#FFFFFF"
        YELLOW_CLR = "#FFFF00"
        GREEN_CLR = "#00FF00"

        # Points setup - avoiding collinearity for clear rotation
        pivot_p = Dot(color=WHITE_CLR)
        blue_1 = Dot(color=BLUE_CLR)
        blue_2 = Dot(color=BLUE_CLR)
        red_1 = Dot(color=RED_CLR)
        red_2 = Dot(color=RED_CLR)

        # Positioning using the 6x6 grid system (Rule L002)
        # Using specific grid anchors to keep visuals in the safe right-side area
        self.place_at_grid(pivot_p, 'C3')
        self.place_at_grid(blue_1, 'B2')
        self.place_at_grid(blue_2, 'E2')
        self.place_at_grid(red_1, 'B4')
        self.place_at_grid(red_2, 'D5')
        
        all_points = VGroup(pivot_p, blue_1, blue_2, red_1, red_2)

        # === Animation for Lecture Line 1 ===
        # "Color points on each side of the line differently."
        self.lecture[0].set_color(YELLOW_CLR)
        
        # Initial vertical-ish line passing through the pivot at C3
        windmill_line = Line(
            self.grid['A3'] + UP*0.5, 
            self.grid['F3'] + DOWN*0.5, 
            color=WHITE_CLR
        )
        
        self.add(all_points)
        self.play(Create(windmill_line))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # "Pick a line splitting points into two equal sets."
        self.lecture[0].set_color(WHITE_CLR)
        self.lecture[1].set_color(YELLOW_CLR)
        
        blue_group = VGroup(blue_1, blue_2)
        red_group = VGroup(red_1, red_2)
        
        # Visual groupings using SurroundingRectangles
        rect_blue = SurroundingRectangle(blue_group, color=BLUE_CLR, buff=0.2)
        rect_red = SurroundingRectangle(red_group, color=RED_CLR, buff=0.2)
        
        # Issue 31: blue_count at A2, scaled to 0.7
        blue_count = Text("2", color=BLUE_CLR, font_size=24)
        self.place_at_grid(blue_count, 'A2', scale_factor=0.7)
        
        # Issue 29/30: red_count at B5, scaled to 0.8
        red_count = Text("2", color=RED_CLR, font_size=24)
        self.place_at_grid(red_count, 'B5', scale_factor=0.8)
        
        self.play(Create(rect_blue), Create(rect_red))
        self.play(Write(blue_count), Write(red_count))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # "As the line pivots, points swap sides across it."
        self.lecture[1].set_color(WHITE_CLR)
        self.lecture[2].set_color(YELLOW_CLR)
        
        self.play(FadeOut(rect_blue), FadeOut(rect_red))
        
        # Track the pivot center with a hidden Mobject to avoid NoneType issues
        pivot_pos_tracker = Dot(pivot_p.get_center(), radius=0).set_opacity(0)
        self.add(pivot_pos_tracker)
        
        # Angle tracker for rotation (start at vertical PI/2)
        angle_tracker = ValueTracker(PI/2)

        def line_updater(l):
            angle = angle_tracker.get_value()
            center = pivot_pos_tracker.get_center()
            dir_vec = np.array([np.cos(angle), np.sin(angle), 0])
            # Extend line enough to cover the grid height
            l.set_points_by_ends(center - dir_vec * 3.5, center + dir_vec * 3.5)

        windmill_line.add_updater(line_updater)
        
        # Rotate clockwise to hit red_1 (PI/4 angle from C3)
        self.play(angle_tracker.animate.set_value(PI/4), run_time=2.0)
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # "The current pivot and the new pivot trade places."
        self.lecture[2].set_color(WHITE_CLR)
        self.lecture[3].set_color(YELLOW_CLR)
        
        # Arrows for exchange (visual hint)
        arrow_to_red1 = Arrow(pivot_p.get_center(), red_1.get_center(), color=GREEN_CLR, buff=0.1)
        
        self.play(GrowArrow(arrow_to_red1))
        self.wait(0.5)
        
        # Swap colors and pivot position
        # The line will automatically update its position because of the updater on pivot_pos_tracker
        self.play(
            pivot_p.animate.set_color(RED_CLR),
            red_1.animate.set_color(WHITE_CLR),
            pivot_pos_tracker.animate.move_to(red_1.get_center()),
            FadeOut(arrow_to_red1),
            run_time=1.0
        )
        
        # Continue rotation to show it's now pivoting around red_1
        self.play(angle_tracker.animate.set_value(angle_tracker.get_value() - PI/6), run_time=1.5)
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # "The total count on each side remains perfectly balanced."
        self.lecture[3].set_color(WHITE_CLR)
        self.lecture[4].set_color(YELLOW_CLR)
        
        # New visual groups showing count preservation
        new_blue_group = VGroup(blue_1, blue_2)
        new_red_group = VGroup(pivot_p, red_2)
        
        rect_blue_f = SurroundingRectangle(new_blue_group, color=BLUE_CLR, buff=0.2)
        rect_red_f = SurroundingRectangle(new_red_group, color=RED_CLR, buff=0.2)
        
        self.play(Create(rect_blue_f), Create(rect_red_f))
        # Use Indicate for pulsing (L004)
        self.play(Indicate(blue_count, color=YELLOW_CLR), Indicate(red_count, color=YELLOW_CLR))
        self.wait(2.0)
        
        # Clean up updater
        windmill_line.remove_updater(line_updater)
