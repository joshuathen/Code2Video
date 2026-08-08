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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Independence: The 'No Influence' Condition", [
            "- Independence: Knowing B provides zero info about A.",
            "- Mathematically, this means P(A|B) equals P(A).",
            "- Imagine a coin flip and a spinning wheel."
        ])
        
        # === Animation for Lecture Line 1 ===
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color=BLUE)
        wheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg", color=YELLOW)
        
        self.place_at_grid(coin, "C2", scale_factor=0.8)
        self.place_at_grid(wheel, "C5", scale_factor=0.8)
        
        self.play(FadeIn(coin), FadeIn(wheel))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        equation = MathTex(r"P(A|B) = P(A)", color=WHITE)
        self.place_in_area(equation, "D3", "F5", scale_factor=0.9)
        
        self.play(Write(equation))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # B moves (spins) without changing A's size to show independence
        target_pos = self.grid["D5"]
        self.play(
            Rotate(wheel, angle=2*PI, run_time=2),
            wheel.animate.move_to(target_pos),
            run_time=2
        )
        self.lecture[2].set_color(GREEN)
        self.wait(1)
