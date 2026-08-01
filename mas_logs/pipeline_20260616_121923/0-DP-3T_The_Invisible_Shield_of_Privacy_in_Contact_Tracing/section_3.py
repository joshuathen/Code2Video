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
        # Define the lecture lines based on original prompt requirements
        lecture_lines = [
            "Alice’s phone generates a unique daily Secret Key.",
            "Hash functions produce rotating Ephemeral IDs from this key.",
            "These IDs change every fifteen minutes to ensure privacy."
        ]
        
        self.setup_layout("Step 1: Generating Ephemeral IDs (The Rotating Mask)", lecture_lines)

        # Colors
        COLOR_SK = "#FFFF00"  # Yellow
        COLOR_ID = "#00FF00"  # Green
        COLOR_HIGHLIGHT = "#58C4DD" # Blue

        # === Animation for Lecture Line 1 ===
        # Alice’s phone generates a unique daily Secret Key.
        self.play(self.lecture[0].animate.set_color(COLOR_SK))
        
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg]
        phone_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg")
        phone_asset.set_color(WHITE)
        self.place_in_area(phone_asset, "B1", "E3", scale_factor=0.6)
        
        sk_text = Text("SK_t", color=COLOR_SK, font_size=24)
        self.place_at_grid(sk_text, "C2")
        
        self.play(FadeIn(phone_asset))
        self.play(Write(sk_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Hash functions produce rotating Ephemeral IDs from this key.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE) # Matching white color for formula and line
        )
        
        # Problem: The formula 'EpsID = Hash(SK_t || Time)' occupies a large 3x3 area (D4-F6).
        # Fix: Line 86 in original: self.place_in_area(formula, 'F4', 'F6', scale_factor=0.6)
        formula = Text("EpsID = Hash(SK_t || Time)", font_size=24, color=WHITE)
        self.place_in_area(formula, "F4", "F6", scale_factor=0.6)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # These IDs change every fifteen minutes to ensure privacy.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_ID)
        )
        
        id_box = Rectangle(height=0.8, width=1.5, color=COLOR_ID)
        self.place_at_grid(id_box, "B5")
        
        id_val = Text("A1B2", font_size=24, color=COLOR_ID)
        self.place_at_grid(id_val, "B5")
        
        time_label = Text("T = 0 min", font_size=20, color=WHITE)
        self.place_at_grid(time_label, "A5")
        
        self.play(Create(id_box), Write(id_val), Write(time_label))
        self.wait(1)
        
        # ID Rotation: 'A1B2' changes to 'C3D4'
        new_id_val = Text("C3D4", font_size=24, color=COLOR_ID)
        self.place_at_grid(new_id_val, "B5")
        new_time_label = Text("T = 15 min", font_size=20, color=WHITE)
        self.place_at_grid(new_time_label, "A5")
        
        self.play(
            Transform(id_val, new_id_val),
            Transform(time_label, new_time_label),
            Flash(id_box, color=COLOR_ID)
        )
        self.wait(1)

        # Privacy Visualization (Trail of IDs)
        # Problem: The trail IDs (trail_id1, trail_id2, trail_id3) are placed in row C.
        # Fix: Move to Row D: D4, D5, D6
        trail_id1 = Text("A1B2", font_size=18, color=COLOR_ID).set_opacity(0.4)
        trail_id2 = Text("C3D4", font_size=18, color=COLOR_ID).set_opacity(0.6)
        trail_id3 = Text("E5F6", font_size=22, color=COLOR_ID)
        
        self.place_at_grid(trail_id1, "D4")
        self.place_at_grid(trail_id2, "D5")
        self.place_at_grid(trail_id3, "D6")
        
        # Disconnect them with "X" to show no linkage
        cross1 = Cross(VGroup(trail_id1, trail_id2), stroke_width=2, scale_factor=0.3)
        cross2 = Cross(VGroup(trail_id2, trail_id3), stroke_width=2, scale_factor=0.3)
        
        self.play(
            FadeOut(id_val),
            FadeOut(id_box),
            FadeOut(time_label),
            FadeIn(trail_id1),
            FadeIn(trail_id2),
            FadeIn(trail_id3)
        )
        self.play(Create(cross1), Create(cross2))
        
        # Anonymity visual
        # Problem: The 'ANONYMOUS' label at F2 is positioned at a single point.
        # Fix: Line 166 in original: self.place_in_area(anon_text, 'F1', 'F3', scale_factor=0.8)
        shield = SurroundingRectangle(phone_asset, color=COLOR_HIGHLIGHT, buff=0.1)
        self.play(Create(shield))
        
        anon_text = Text("ANONYMOUS", font_size=24, color=COLOR_HIGHLIGHT)
        self.place_in_area(anon_text, "F1", "F3", scale_factor=0.8)
        self.play(Write(anon_text))
        
        self.wait(2)
