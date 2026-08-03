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
        self.setup_layout(
            "The First Layer: Pattern Matching (The Keys)",
            [
                "The first MLP layer uses a matrix of keys.",
                "Input vectors act as queries searching for a match.",
                "We calculate the dot product between queries and keys.",
                "High alignment triggers a specific neuron in the layer.",
                "This detects patterns like 'The capital city of.'"
            ]
        )

        # Colors
        QUERY_COLOR = "#FFFF00"
        KEY_COLOR = "#00FF00"
        GLOW_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Define positions for key vectors
        key_grid_positions = ["C3", "C4", "C5", "D2", "D4", "D5", "E3", "E4"]
        key_vectors = VGroup()
        # Pre-defined angles for keys to ensure visual diversity and deterministic results
        angle_map = {
            "C3": PI/4, "C4": PI/2, "C5": 3*PI/4,
            "D2": 0, "D4": -PI/6, "D5": -PI/3,
            "E3": PI, "E4": 5*PI/4
        }
        
        for pos in key_grid_positions:
            angle = angle_map[pos]
            # Create a small arrow representing a key vector
            v = Arrow(ORIGIN, [0.6 * np.cos(angle), 0.6 * np.sin(angle), 0], color=KEY_COLOR, buff=0)
            self.place_at_grid(v, pos)
            key_vectors.add(v)
            
        key_label = Text("Key Vectors (W1)", color=KEY_COLOR, font_size=18)
        # Fix Issue 35: Balanced centering for wide text
        self.place_in_area(key_label, "B3", "B5", scale_factor=0.8)

        self.play(FadeIn(key_vectors), Write(key_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Query vector at D3 (central)
        query_vector = Arrow(ORIGIN, [0.6, 0, 0], color=QUERY_COLOR, buff=0)
        self.place_at_grid(query_vector, "D3")
        
        query_label = Text("Query (Input)", color=QUERY_COLOR, font_size=18)
        # Fix Issue 34: Position query_label at E3 to avoid overlap with left-side lecture notes
        self.place_at_grid(query_label, "E3", scale_factor=0.8)
        
        self.play(FadeIn(query_vector), Write(query_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Dot product symbol between Query (D3) and one specific Key (D4)
        dot_symbol = MathTex(r"\cdot", color=WHITE, font_size=40)
        mid_point = (self.grid["D3"] + self.grid["D4"]) / 2
        dot_symbol.move_to(mid_point)
        
        self.play(Write(dot_symbol))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # target_key corresponds to "D4" (index 4 in key_grid_positions list)
        target_key = key_vectors[4]
        target_angle = angle_map["D4"]
        
        # White glow for target key
        glow = target_key.copy().set_color(GLOW_COLOR).set_stroke(width=10, opacity=0.5)
        
        # Rotate query_vector toward the target key's direction
        angle_to_rotate = target_angle - query_vector.get_angle()
        
        self.play(
            Rotate(query_vector, angle=angle_to_rotate, about_point=self.grid["D3"]),
            FadeIn(glow),
            run_time=1.5
        )
        self.play(Indicate(target_key, color=GLOW_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        pattern_text = Text('"The capital city of."', color=HIGHLIGHT_COLOR, font_size=20, slant=ITALIC)
        # Fix Issue 36: Balanced centering for pattern text
        self.place_in_area(pattern_text, "F3", "F5", scale_factor=0.9)
        
        self.play(Write(pattern_text))
        self.wait(2)
        
        # Final cleanup/reset colors
        self.lecture[4].set_color(WHITE)
        self.wait(1)
