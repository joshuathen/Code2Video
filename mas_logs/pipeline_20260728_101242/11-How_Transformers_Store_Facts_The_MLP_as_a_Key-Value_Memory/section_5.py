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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Full Pipeline: Fact Retrieval in Action", 
            [
                "Together, these layers form a Key-Value Memory system.",
                "The model performs a \"soft lookup\" across all keys.",
                "Multiple facts can be retrieved and combined simultaneously.",
                "This allows for processing complex, multi-faceted queries.",
                "The MLP effectively acts as a massive internal database."
            ]
        )

        # Colors
        COLOR_INPUT = "#FFFFE0"
        COLOR_KEY = "#ADD8E6"
        COLOR_VALUE = "#90EE90"
        COLOR_OUTPUT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Together, these layers form a Key-Value Memory system.
        self.lecture[0].set_color(YELLOW)

        # Key Column
        key_rects = VGroup(*[Rectangle(height=0.6, width=1.4, color=COLOR_KEY) for _ in range(3)])
        key_labels = VGroup(
            Text("Key 1", font_size=18),
            Text("Shakespeare", font_size=18),
            Text("Key 3", font_size=18)
        )
        # Using specific grid positions
        self.place_at_grid(key_rects[0], "B2")
        self.place_at_grid(key_labels[0], "B2", scale_factor=0.8)
        self.place_at_grid(key_rects[1], "C2")
        # Fix for Issue #39: scale 'Shakespeare' label
        self.place_at_grid(key_labels[1], "C2", scale_factor=0.8)
        self.place_at_grid(key_rects[2], "D2")
        self.place_at_grid(key_labels[2], "D2", scale_factor=0.8)
        
        # Value Column
        value_rects = VGroup(*[Rectangle(height=0.6, width=1.4, color=COLOR_VALUE) for _ in range(3)])
        value_labels = VGroup(
            Text("Value 1", font_size=18),
            Text("Playwright", font_size=18),
            Text("Value 3", font_size=18)
        )
        self.place_at_grid(value_rects[0], "B4")
        self.place_at_grid(value_labels[0], "B4", scale_factor=0.8)
        self.place_at_grid(value_rects[1], "C4")
        self.place_at_grid(value_labels[1], "C4", scale_factor=0.8)
        self.place_at_grid(value_rects[2], "D4")
        self.place_at_grid(value_labels[2], "D4", scale_factor=0.8)

        key_title = Text("Keys", font_size=20, color=COLOR_KEY)
        val_title = Text("Values", font_size=20, color=COLOR_VALUE)
        
        # Manually positioning titles above columns using grid logic
        key_title.move_to(self.grid["B2"] + UP * 0.7)
        val_title.move_to(self.grid["B4"] + UP * 0.7)

        self.play(
            Create(key_rects), 
            Write(key_labels),
            Create(value_rects),
            Write(value_labels),
            Write(key_title),
            Write(val_title),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The model performs a "soft lookup" across all keys.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        input_vec_text = Text("Romeo & Juliet", font_size=20, color=COLOR_INPUT)
        # Fix for Issue #37: scale 'Romeo & Juliet' input vector
        self.place_at_grid(input_vec_text, "C1", scale_factor=0.4)
        
        # Input hits "Shakespeare" Key (C2)
        self.play(FadeIn(input_vec_text))
        self.play(input_vec_text.animate.move_to(self.grid["C2"]), run_time=1.5)
        
        # Soft lookup highlight
        highlight_c2 = key_rects[1].copy().set_fill(COLOR_KEY, opacity=0.6)
        highlight_b2 = key_rects[0].copy().set_fill(COLOR_KEY, opacity=0.2)
        highlight_d2 = key_rects[2].copy().set_fill(COLOR_KEY, opacity=0.2)
        
        self.play(
            FadeIn(highlight_c2),
            FadeIn(highlight_b2),
            FadeIn(highlight_d2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Multiple facts can be retrieved and combined simultaneously.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Lines from Keys to Values representing activations
        l1 = Line(self.grid["B2"], self.grid["B4"], color=COLOR_KEY, stroke_width=2)
        l2 = Line(self.grid["C2"], self.grid["C4"], color=COLOR_KEY, stroke_width=6)
        l3 = Line(self.grid["D2"], self.grid["D4"], color=COLOR_KEY, stroke_width=2)

        self.play(Create(l1), Create(l2), Create(l3))
        
        # Values light up
        val_highlight_c4 = value_rects[1].copy().set_fill(COLOR_VALUE, opacity=0.6)
        val_highlight_b4 = value_rects[0].copy().set_fill(COLOR_VALUE, opacity=0.2)
        val_highlight_d4 = value_rects[2].copy().set_fill(COLOR_VALUE, opacity=0.2)

        self.play(
            FadeIn(val_highlight_c4),
            FadeIn(val_highlight_b4),
            FadeIn(val_highlight_d4)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This allows for processing complex, multi-faceted queries.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Activation vectors emerge from values
        v1 = Vector(RIGHT * 0.8, color=COLOR_VALUE)
        v2 = Vector(RIGHT * 0.8, color=COLOR_VALUE)
        v3 = Vector(RIGHT * 0.8, color=COLOR_VALUE)
        
        self.place_at_grid(v1, "B5")
        self.place_at_grid(v2, "C5")
        self.place_at_grid(v3, "D5")

        self.play(GrowArrow(v1), GrowArrow(v2), GrowArrow(v3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The MLP effectively acts as a massive internal database.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Final summation to Output
        output_text = Text("Output: William Shakespeare", font_size=20, color=COLOR_OUTPUT)
        # Fix for Issue #38: scale output text
        self.place_at_grid(output_text, "C6", scale_factor=0.5)

        # Move vectors to combine at the output position
        self.play(
            v1.animate.move_to(self.grid["C6"]),
            v2.animate.move_to(self.grid["C6"]),
            v3.animate.move_to(self.grid["C6"]),
            run_time=1.5
        )
        
        self.play(
            FadeOut(v1), FadeOut(v2), FadeOut(v3),
            Write(output_text)
        )
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
        self.wait(2)
