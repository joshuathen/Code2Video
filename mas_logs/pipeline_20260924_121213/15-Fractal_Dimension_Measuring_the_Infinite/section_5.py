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
        self.setup_layout("Summary and Synthesis", [
            "Fractal dimension bridges integer dimensions.",
            "It quantifies space-filling at infinity.",
            "It captures shape complexity perfectly."
        ])
        
        # Load assets
        fern_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fern.svg")
        lightning_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lightning.svg")
        
        # Table of dimensions
        table_labels = VGroup(
            Text("Dimension (D)", color=BLUE, font_size=24),
            Text("Object Type", color=GREEN, font_size=24)
        ).arrange(RIGHT, buff=1.0)
        
        data = [("1", "Line"), ("2", "Square"), ("3", "Cube")]
        
        rows = VGroup()
        for d, obj in data:
            row = VGroup(Text(d, font_size=22), Text(obj, font_size=22)).arrange(RIGHT, buff=1.5)
            rows.add(row)
        
        synthesis_table = VGroup(table_labels, rows.arrange(DOWN, buff=0.5)).arrange(DOWN, buff=0.8)
        
        # Applying requested placement adjustments
        self.place_in_area(synthesis_table, "A2", "E5", scale_factor=0.85)
        
        # Complexity label and lightning
        complexity_label = Text("Complexity", color=YELLOW, font_size=26)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(fern_icon, "B2", scale_factor=0.5)
        self.play(FadeIn(synthesis_table), FadeIn(fern_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        fractal_d = Text("D = 1.26 (Fractal)", color=RED, font_size=22)
        self.place_at_grid(fractal_d, "D4", scale_factor=0.9)
        self.play(FadeIn(fractal_d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.place_at_grid(complexity_label, "E3", scale_factor=0.8)
        self.place_at_grid(lightning_icon, "E5", scale_factor=0.5)
        self.play(FadeIn(complexity_label), FadeIn(lightning_icon))
        self.wait(2)
