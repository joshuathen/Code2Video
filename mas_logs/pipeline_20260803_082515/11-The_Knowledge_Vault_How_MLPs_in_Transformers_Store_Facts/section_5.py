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
        self.setup_layout("The Second Layer: Retrieving the Fact (The Values)", [
            "The second layer contains a corresponding bank of values.",
            "Each activated key neuron links to a value vector.",
            "If a key matches, its value is retrieved.",
            "This value vector represents the factual answer like \"Paris.\"",
            "The retrieved fact is added back to the stream."
        ])
        
        # Colors
        VALUE_COLOR = "#00BFFF"
        HIGHLIGHT_COLOR = "#FFFF00"
        STREAM_COLOR = WHITE

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(VALUE_COLOR)
        
        # Display a grid of blue vectors #00BFFF labeled "Value Vectors (W2)".
        value_vectors = VGroup()
        for r_idx, row in enumerate(["B", "C", "D"]):
            for col_idx, col in enumerate(["3", "4", "5", "6"]):
                vec = Arrow(start=LEFT*0.2, end=RIGHT*0.2, color=VALUE_COLOR, stroke_width=4, buff=0)
                self.place_at_grid(vec, f"{row}{col}")
                value_vectors.add(vec)
        
        bank_label = Text("Value Vectors (W2)", font_size=20, color=WHITE)
        self.place_at_grid(bank_label, "A4")
        
        self.play(FadeIn(value_vectors), FadeIn(bank_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(VALUE_COLOR)
        
        # Highlight the specific Value vector linked to the active Key.
        target_vector = value_vectors[4] # Index 4 in the new 4-col grid is roughly C3
        highlight_box = SurroundingRectangle(target_vector, color=HIGHLIGHT_COLOR, buff=0.1)
        
        # Updated: Starting link from C2 instead of C1 as per Issue 39 logic
        key_link_line = Line(self.grid["C2"], self.grid["C3"], color=HIGHLIGHT_COLOR, stroke_width=2).add_tip(tip_length=0.1)
        key_label = Text("Active Key", font_size=16, color=HIGHLIGHT_COLOR)
        self.place_at_grid(key_label, "C2") # Fix for Issue 39

        self.play(Create(highlight_box), Create(key_link_line), FadeIn(key_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(VALUE_COLOR)
        
        # If a key matches, its value is retrieved.
        retrieved_vector = target_vector.copy()
        self.add(retrieved_vector)
        self.play(retrieved_vector.animate.set_color(HIGHLIGHT_COLOR).scale(1.2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(VALUE_COLOR)
        
        # This value vector represents the factual answer like "Paris."
        # Animate this Value vector (labeled "Paris") moving downward.
        paris_label = Text("Paris", font_size=24, color=HIGHLIGHT_COLOR)
        paris_label.next_to(retrieved_vector, UP, buff=0.2)
        
        self.play(Write(paris_label))
        self.wait(0.5)
        
        # Move down towards the residual stream area
        # Target position is updated to align with the new stream area (F3-F6)
        target_stream_pos = self.grid["F4.5"] if "F4.5" in self.grid else (self.grid["F4"] + self.grid["F5"])/2
        self.play(
            retrieved_vector.animate.move_to(target_stream_pos),
            paris_label.animate.move_to(target_stream_pos + UP * 0.4),
            FadeOut(highlight_box),
            FadeOut(key_link_line),
            FadeOut(key_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(VALUE_COLOR)
        
        # The retrieved fact is added back to the stream.
        # Show a plus sign as Value joins the Residual Stream.
        # Update the Residual Stream box label to "Vector + Information".
        
        residual_box = Rectangle(width=4, height=0.6, color=STREAM_COLOR)
        self.place_in_area(residual_box, "F3", "F6") # Fix for Issue 41
        stream_label = Text("Residual Stream", font_size=18, color=STREAM_COLOR)
        self.place_in_area(stream_label, "F3", "F6") # Fix for Issue 41
        
        plus_sign = MathTex("+", font_size=36, color=WHITE)
        self.place_at_grid(plus_sign, "F2") # Fix for Issue 40
        
        self.play(Create(residual_box), Write(stream_label))
        self.play(FadeIn(plus_sign))
        
        new_stream_label = Text("Vector + Information", font_size=18, color=STREAM_COLOR)
        self.place_in_area(new_stream_label, "F3", "F6") # Fix for Issue 41
        
        self.play(
            Transform(stream_label, new_stream_label),
            retrieved_vector.animate.scale(0).move_to(residual_box.get_center()),
            paris_label.animate.scale(0).move_to(residual_box.get_center()),
            run_time=1.5
        )
        self.wait(2)
