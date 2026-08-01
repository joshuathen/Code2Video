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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "Prerequisite 1: Bluetooth Low Energy (BLE) & Hashing"
        lines = [
            "Bluetooth Low Energy enables short-range digital handshakes.",
            "Cryptographic hashing turns secret keys into random IDs.",
            "These one-way functions prevent reversing the random IDs."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        BLE_BLUE = "#1E90FF"
        KEY_GOLD = "#FFD700"
        HASH_GRAY = "#A9A9A9"
        ID_LIME = "#ADFF2F"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLE_BLUE))
        
        # Phone Icons
        phone1 = RoundedRectangle(corner_radius=0.1, height=1.2, width=0.7, color=WHITE)
        phone2 = RoundedRectangle(corner_radius=0.1, height=1.2, width=0.7, color=WHITE)
        
        self.place_at_grid(phone1, "B2")
        self.place_at_grid(phone2, "B5")
        
        self.play(FadeIn(phone1), FadeIn(phone2))
        
        # Concentric Rings (Manual expansion to avoid always_redraw)
        def get_rings(center):
            return VGroup(*[Circle(radius=0.2 * i, color=BLE_BLUE, stroke_opacity=1 - 0.2*i).move_to(center) for i in range(1, 5)])

        rings1 = get_rings(self.grid["B2"])
        rings2 = get_rings(self.grid["B5"])
        
        self.play(
            LaggedStart(*[r.animate.scale(1.5).set_stroke(opacity=0) for r in rings1], lag_ratio=0.3),
            LaggedStart(*[r.animate.scale(1.5).set_stroke(opacity=0) for r in rings2], lag_ratio=0.3),
            run_time=2,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fading out phones to clear grid space for hashing (Resolves Issue 48)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(KEY_GOLD),
            FadeOut(phone1), FadeOut(phone2)
        )
        
        # Secret Key (Resolves Issue 46: moved to D1-D2)
        secret_key = Text("Secret Key", font_size=20, color=KEY_GOLD)
        self.place_in_area(secret_key, "D1", "D2", scale_factor=0.8)
        
        # Hash Box (Resolves Issue 47: moved to D3-D4)
        hash_box = Square(side_length=1.2, color=HASH_GRAY, fill_opacity=0.5)
        hash_label = Text("Hash\nFunction", font_size=16, color=WHITE).move_to(hash_box.get_center())
        hash_group = VGroup(hash_box, hash_label)
        self.place_in_area(hash_group, "D3", "D4", scale_factor=0.8)
        
        self.play(FadeIn(secret_key), FadeIn(hash_group))
        
        # Unique ID (Resolves Issue 47: moved to D6)
        unique_id = Text("8a3f...2e", font_size=20, color=ID_LIME)
        self.place_at_grid(unique_id, "D6", scale_factor=1.0)

        # Move key into box
        self.play(secret_key.animate.move_to(hash_group.get_center()).scale(0.5).set_opacity(0), run_time=1.5)
        
        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ID_LIME)
        )
        
        # Unique ID exits
        self.play(FadeIn(unique_id, shift=RIGHT), run_time=1.5)
        
        # Attempt to reverse (visualize failure)
        # Using grid D4 for the end of the arrow to point back at the hash box
        cross = Cross(unique_id, color=RED).scale(1.2)
        arrow_back = Arrow(start=self.grid["D6"], end=self.grid["D4"], color=RED)
        
        self.play(Create(arrow_back))
        self.play(Create(cross))
        
        self.wait(2)
