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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Mass ratios affect the collision count.",
            "Powers of one hundred reveal Pi.",
            "One, one-hundred, ten-thousand ratios work.",
            "Three, thirty-one, three-hundred-fourteen collisions result.",
            "The digits of Pi appear naturally."
        ]
        self.setup_layout("The Unexpected Pattern: Scaling Mass", lecture_lines)
        
        # Assets: Changed ImageMobject to SVGMobject as the source files are .svg
        block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        weight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/weight.svg")
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")

        # === Animation for Lecture Line 1 ===
        # Display system with [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg] and [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/weight.svg] with mass ratios 1:100. Label #FFFFFF.
        ratio_text_1 = Text("Ratio 1:100", color="#FFFFFF")
        self.place_at_grid(ratio_text_1, 'B2', scale_factor=0.6)
        self.place_at_grid(block, 'A1', scale_factor=0.4)
        self.place_at_grid(weight, 'A2', scale_factor=0.4)
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(ratio_text_1), FadeIn(block), FadeIn(weight))

        # === Animation for Lecture Line 2 ===
        # Increase ratio to 1:10000. Label #FFFF00.
        ratio_text_2 = Text("Ratio 1:10,000", color="#FFFF00")
        self.place_at_grid(ratio_text_2, 'B5', scale_factor=0.6)
        self.lecture[1].set_color("#FFFF00")
        self.play(FadeIn(ratio_text_2))

        # === Animation for Lecture Line 3 ===
        # Show number of collisions [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg] increases dramatically. Label #00FFFF.
        collision_text = Text("Collisions Increase", color="#00FFFF")
        self.place_in_area(collision_text, 'D1', 'D3', scale_factor=0.55)
        self.place_at_grid(sphere, 'C4', scale_factor=0.4)
        self.lecture[2].set_color("#00FFFF")
        self.play(FadeIn(collision_text), FadeIn(sphere))

        # === Animation for Lecture Line 4 ===
        # Display connection to digits of Pi. Label #00FF00.
        pi_conn_text = Text("3, 31, 314...", color="#00FF00")
        self.place_in_area(pi_conn_text, 'D4', 'D6', scale_factor=0.55)
        self.lecture[3].set_color("#00FF00")
        self.play(FadeIn(pi_conn_text))

        # === Animation for Lecture Line 5 ===
        # Show visual pi representation against a [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg]. Label #FF00FF.
        pi_symbol = MathTex(r"\\pi", color="#FF00FF")
        self.place_at_grid(pi_symbol, 'E4', scale_factor=1.0)
        self.place_at_grid(wall, 'E3', scale_factor=0.5)
        self.lecture[4].set_color("#FF00FF")
        self.play(FadeIn(pi_symbol), FadeIn(wall))
        self.wait(2)
