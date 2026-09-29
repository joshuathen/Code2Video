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
            "Every circle has a special, constant ratio.",
            "It relates the circumference to the diameter.",
            "This ratio stays the same for all circles."
        ]
        self.setup_layout("The Hook: The Constant Ratio", lecture_lines)
        
        # Define objects
        wheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg", color=WHITE)
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        diameter_line = Line(LEFT, RIGHT, color=BLUE)
        ratio_tex = MathTex(r"\frac{C}{d}", font_size=48)
        pi_symbol = MathTex(r"\pi", color="#FFFF00", font_size=72)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(wheel, 'B3', scale_factor=0.7)
        self.play(FadeIn(wheel))
        self.lecture[0].set_color("#FF4500")

        # === Animation for Lecture Line 2 ===
        # The prompt for line 2 uses place_in_area for the group. 
        # Using a VGroup for the math objects:
        math_group = VGroup(ratio_tex)
        self.place_in_area(math_group, 'D2', 'E4', scale_factor=0.9)
        self.play(Write(ratio_tex))
        self.lecture[1].set_color("#00BFFF")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(coin, 'B5', scale_factor=0.3)
        self.play(FadeIn(coin))
        self.place_at_grid(pi_symbol, 'E4', scale_factor=1.0)
        self.play(Transform(ratio_tex, pi_symbol))
        self.lecture[2].set_color("#FFFF00")
        self.wait(1)
