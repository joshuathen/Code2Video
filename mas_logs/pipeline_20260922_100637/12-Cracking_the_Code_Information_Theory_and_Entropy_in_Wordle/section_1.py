from manim import *

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
        lecture_lines = [
            "Claude Shannon defined information as reduced uncertainty.",
            "Unexpected events carry more information than expected ones.",
            "Consider a coin flip: high uncertainty means high information."
        ]
        self.setup_layout("The Core Concept: What is Information?", lecture_lines)
        
        # Elements
        info_text = Text("What is Information?", font_size=36, color=WHITE)
        # Using SVG asset
        coin0 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color="#FFD700")
        coin1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color="#FFD700")
        
        data_stream = VGroup(*[Text(x, font_size=24, color="#00BFFF") for x in "101100101"]).arrange(RIGHT)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(info_text, "A2", scale_factor=0.9)
        self.play(FadeIn(info_text))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeOut(info_text))
        self.place_at_grid(coin0, "B4", scale_factor=0.8)
        self.place_at_grid(coin1, "B5", scale_factor=0.8)
        self.play(Create(coin0), Create(coin1))
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        self.place_in_area(data_stream, "C3", "C6", scale_factor=0.6)
        self.play(Transform(VGroup(coin0, coin1), data_stream))
        self.lecture[1].set_color("#00BFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(data_stream))
        self.lecture[2].set_color("#FF4500")
        self.wait(2)
        self.play(FadeOut(self.lecture), FadeOut(self.title))
