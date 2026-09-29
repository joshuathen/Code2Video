from manim import *
import random

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
        self.setup_layout("Conclusion: Information Density", [
            "This strategy uses basic Hamming code principles.",
            "We encode information into simple parity checks.",
            "The grid becomes an error-correcting storage device."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Use asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg]
        # In this context, we will use SVGMobject or ImageMobject for the asset
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_asset, 'A3', 'F6', scale_factor=0.9)
        self.add(grid_asset)
        
        dots = VGroup(*[Dot(color="#00FFFF", radius=0.1) for _ in range(12)])
        grid_positions = [f"{r}{c}" for r in "ABCDEF" for c in "123456"]
        selected_positions = random.sample(grid_positions, 12)
        
        animations = []
        for i, dot in enumerate(dots):
            self.place_at_grid(dot, selected_positions[i])
            animations.append(FadeIn(dot))
        
        self.play(*animations)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        # Text for better explanation
        svm_explanation_text = Text("High Density Storage", font_size=20, color="#FF0000")
        self.place_in_area(svm_explanation_text, 'A1', 'B6', scale_factor=0.6)
        
        # Red dots for high density
        red_dots = VGroup(*[Dot(color="#FF0000", radius=0.15) for _ in range(5)])
        
        self.play(
            FadeOut(dots),
            FadeIn(svm_explanation_text),
            *[FadeIn(self.place_at_grid(red_dots[i], 'D4')) for i in range(5)]
        )
        self.wait(2)
