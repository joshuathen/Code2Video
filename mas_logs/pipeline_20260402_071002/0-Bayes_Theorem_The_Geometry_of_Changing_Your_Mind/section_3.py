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
        title = "The Evidence: Overlapping Realities"
        lines = [
            "Now, our Meow-Sensor provides new evidence.",
            "It triggers ninety percent of the time for cats.",
            "It also triggers twenty percent of the time without cats.",
            "Vertical columns represent the sensor's response.",
            "Overlapping regions show where the sensor actually triggers."
        ]
        self.setup_layout(title, lines)

        # Colors
        GREEN_COLOR = "#2ECC71"
        RED_COLOR = "#E74C3C"
        WHITE_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = YELLOW

        # Grid-based sizing (Square is 3x3 grid units from B2 to E5)
        # B2 top-left (1.5, 1.2), E5 bottom-right (4.5, -1.8)
        # Width = 3, Height = 3
        square_width = 3.0
        square_height = 3.0

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Base Square (Gray background)
        main_square = Rectangle(
            width=square_width, height=square_height, 
            stroke_width=2, color=GRAY_D, fill_opacity=0.1
        )
        self.place_in_area(main_square, "B2", "E5")
        
        # Horizontal Split (from Section 2: 10% Cat row at bottom, 90% No Cat row)
        # Cat row height = 0.3, No Cat row height = 2.7
        # Line at height 0.3 from bottom
        split_line = Line(
            start=main_square.get_left() + UP * (-1.5 + 0.3),
            end=main_square.get_right() + UP * (-1.5 + 0.3),
            color=WHITE, stroke_width=2
        )
        
        label_no_cat = Text("No Cat (90%)", font_size=16).next_to(main_square, RIGHT).shift(UP * 0.5)
        label_cat = Text("Cat (10%)", font_size=16).next_to(main_square, RIGHT).shift(DOWN * 1.3)
        
        self.play(Create(main_square), Create(split_line))
        self.play(Write(label_no_cat), Write(label_cat))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN_COLOR)
        
        # True Positive Area: 90% of bottom row width
        tp_width = 0.9 * square_width
        tp_height = 0.1 * square_height
        true_positive_rect = Rectangle(
            width=tp_width, height=tp_height,
            fill_color=GREEN_COLOR, fill_opacity=0.6, stroke_width=0
        )
        # Position at bottom-left of the main square
        true_positive_rect.move_to(
            main_square.get_bottom() + LEFT * (square_width/2 - tp_width/2) + UP * (tp_height/2)
        )
        
        self.play(FadeIn(true_positive_rect))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED_COLOR)
        
        # False Positive Area: 20% of top row width
        fp_width = 0.2 * square_width
        fp_height = 0.9 * square_height
        false_positive_rect = Rectangle(
            width=fp_width, height=fp_height,
            fill_color=RED_COLOR, fill_opacity=0.6, stroke_width=0
        )
        # Position at top-left of the main square (just above the split line)
        false_positive_rect.move_to(
            main_square.get_top() + LEFT * (square_width/2 - fp_width/2) + DOWN * (fp_height/2)
        )
        
        self.play(FadeIn(false_positive_rect))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Draw bold white rectangle outlining both positive regions
        # This is essentially the combined area where the sensor triggers.
        sensor_trigger_outline = VGroup(
            Rectangle(width=fp_width, height=fp_height, stroke_color=WHITE_COLOR, stroke_width=4),
            Rectangle(width=tp_width, height=tp_height, stroke_color=WHITE_COLOR, stroke_width=4)
        )
        sensor_trigger_outline[0].move_to(false_positive_rect)
        sensor_trigger_outline[1].move_to(true_positive_rect)
        
        self.play(Create(sensor_trigger_outline))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Labels
        label_sensor = Text("Sensor Beeps", font_size=18, color=WHITE_COLOR)
        # FIX Issue 33: Use place_in_area for centered label
        self.place_in_area(label_sensor, "A3", "A4", scale_factor=1.0)
        
        label_tp = Text("True Positive", font_size=14, color=GREEN_COLOR)
        # FIX Issue 34: Use place_in_area for balanced label
        self.place_in_area(label_tp, "F3", "F4", scale_factor=1.0)
        
        self.play(Write(label_sensor), Write(label_tp))
        self.wait(2)
