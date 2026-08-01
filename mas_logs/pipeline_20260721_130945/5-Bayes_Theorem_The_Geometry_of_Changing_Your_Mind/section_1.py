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

class Section1Scene(TeachingScene):
    def construct(self):
        # Data from storyboard and outline
        title = "The Core Intuition: Probability as Space"
        lines = [
            "Imagine all possibilities fit inside a unit square.",
            "The total area of this square equals one.",
            "We represent probability as a specific area within it."
        ]
        
        self.setup_layout(title, lines)
        
        # Define Colors
        COLOR_SQUARE_OUTLINE = WHITE
        COLOR_SQUARE_FILL = "#333333"
        COLOR_PROBABILITY_AREA = "#3498DB" # Blue

        # === Animation for Lecture Line 1 ===
        # Imagine all possibilities fit inside a unit square.
        # Draw a square outline (color: #FFFFFF) and fill it with dark grey (#333333).
        
        # Create square
        unit_square = Square(side_length=3.0, stroke_color=COLOR_SQUARE_OUTLINE, fill_color=COLOR_SQUARE_FILL, fill_opacity=1.0)
        
        # ISSUE 26 FIX: Move unit_square to B1-E4 area to provide more room for labels.
        self.place_in_area(unit_square, "B1", "E4")
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            Create(unit_square),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The total area of this square equals one.
        # Fade in the text 'Area = 1' at the center of the square.
        
        area_label = Text("Area = 1", font_size=24, color=WHITE)
        # Position label at center of square
        area_label.move_to(unit_square.get_center())
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            FadeIn(area_label),
            run_time=1.0
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We represent probability as a specific area within it.
        # Transform a portion of the square to blue (#3498DB), covering 10% of the total area.
        
        # Create a rectangle representing 10% of the square's area.
        blue_rect = Rectangle(
            width=unit_square.width, 
            height=unit_square.height * 0.1, 
            fill_color=COLOR_PROBABILITY_AREA, 
            fill_opacity=1.0, 
            stroke_width=0
        )
        # Position it at the bottom of the unit square
        blue_rect.move_to(unit_square.get_critical_point(DOWN), aligned_edge=DOWN)
        
        # Create label for the event
        event_label = Text("Event: 10% Probability", font_size=18, color=COLOR_PROBABILITY_AREA)
        
        # ISSUE 25 FIX: Position event_label at F2 with scale_factor=0.6 to avoid right-edge clipping.
        self.place_at_grid(event_label, "F2", scale_factor=0.6)
        
        # Pointer line from label to blue area
        pointer = Line(
            start=event_label.get_critical_point(UP),
            end=blue_rect.get_center(),
            color=COLOR_PROBABILITY_AREA,
            stroke_width=2
        ).add_tip(tip_length=0.15)

        self.play(
            self.lecture[2].animate.set_color(COLOR_PROBABILITY_AREA),
            FadeIn(blue_rect),
            Write(event_label),
            Create(pointer),
            run_time=1.5
        )
        self.wait(2)
