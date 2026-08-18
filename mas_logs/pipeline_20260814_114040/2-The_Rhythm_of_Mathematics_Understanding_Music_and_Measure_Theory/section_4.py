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
        self.setup_layout("Dynamic Application: Changing Meters", [
            "Different time signatures change the musical feel.",
            "A 4/4 march feels structured and heavy.",
            "A 3/4 waltz spins in a circle."
        ])
        
        # Load asset
        character = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png")
        
        # Elements for March and Waltz
        # B035: Offsetting animation vertical center (Row B/E)
        march_track = Square(side_length=2, color="#FF6347")
        waltz_track = RegularPolygon(n=3, radius=1.2, color="#9370DB")
        
        # Labels
        label_44 = Text("4/4 March", font_size=20, color="#FF6347")
        label_34 = Text("3/4 Waltz", font_size=20, color="#9370DB")

        # === Animation for Lecture Line 1 ===
        self.wait(1)
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF6347")
        # B002: Columns 4-6. B035: Row B for vertical offset.
        self.place_at_grid(march_track, "B4", scale_factor=0.5)
        self.place_at_grid(character, "B4", scale_factor=0.2)
        self.place_at_grid(label_44, "C4", scale_factor=0.8)
        self.play(Create(march_track), FadeIn(character), Write(label_44))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#9370DB")
        # B035: Row E for vertical offset.
        self.place_at_grid(waltz_track, "E4", scale_factor=0.5)
        self.place_at_grid(label_34, "F4", scale_factor=0.8)
        
        # Morphing: The character stays centered as per requirement
        self.play(
            ReplacementTransform(march_track, waltz_track),
            FadeOut(label_44),
            Write(label_34),
            run_time=2
        )
        self.wait(2)
