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
        self.setup_layout("Prerequisites: The Harmonic Series and Convergence", [
            "Consider the infinite series 1/n to the s.",
            "It converges precisely when s is greater than 1.",
            "Think of this like Zeno’s paradox in reverse."
        ])
        
        # Assets
        hare = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hare.svg")
        tortoise = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tortoise.svg")
        
        # Elements
        harmonic_series = MathTex(r"\sum_{n=1}^{\infty} \frac{1}{n^s} = 1 + \frac{1}{2^s} + \frac{1}{3^s} + \dots", font_size=36)
        
        # === Animation for Lecture Line 1 ===
        # Using place_in_area as recommended by VideoCritic
        self.place_in_area(harmonic_series, 'B3', 'B6', scale_factor=0.9)
        hare_icon = self.place_at_grid(hare, "A3", scale_factor=0.5)
        self.play(FadeIn(harmonic_series), FadeIn(hare_icon))
        self.lecture[0].set_color("#00BFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00BFFF")
        condition = Text("s > 1", color="#32CD32", font_size=32)
        
        # Using place_at_grid as recommended by VideoCritic
        self.place_at_grid(condition, 'D4', scale_factor=1.2)
        self.play(FadeIn(condition))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00BFFF")
        
        # Simple representation of Zeno-like steps
        steps = VGroup(*[Square(side_length=0.3, color=WHITE).shift(i*0.4*RIGHT) for i in range(5)])
        
        # Using place_in_area as recommended by VideoCritic
        self.place_in_area(steps, 'E3', 'E5', scale_factor=0.8)
        tortoise_icon = self.place_at_grid(tortoise, "F6", scale_factor=0.5)
        
        self.play(Create(steps), FadeIn(tortoise_icon))
        self.wait(2)
