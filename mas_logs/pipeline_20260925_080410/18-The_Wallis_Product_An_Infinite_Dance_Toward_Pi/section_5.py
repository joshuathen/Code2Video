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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Calculus meets the circle constant pi.", "Wallis provides deep numerical insight.", "A elegant dance toward the truth."]
        self.setup_layout("Summary and Significance", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        wallis = MathTex(r"\frac{\pi}{2} = \prod_{n=1}^{\infty} \left( \frac{4n^2}{4n^2-1} \right)", font_size=32)
        bell = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg").set_color(WHITE)
        bell.next_to(wallis, UP, buff=0.2)
        group1 = VGroup(wallis, bell)
        self.place_in_area(group1, "A3", "C6", scale_factor=0.9)
        self.play(Write(wallis), FadeIn(bell))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        gaussian = MathTex(r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}", font_size=32)
        gaussian.set_color("#00FF00")
        self.place_in_area(gaussian, "D3", "E6", scale_factor=0.9)
        self.play(FadeIn(gaussian))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        dice = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg").set_color("#FFD700")
        circle = Circle(radius=0.5, color="#FFD700")
        dice.next_to(circle, UP, buff=0.2)
        group3 = VGroup(circle, dice)
        self.place_at_grid(group3, "F3", scale_factor=0.7)
        self.play(Create(circle), FadeIn(dice))
        self.play(Indicate(wallis), Indicate(gaussian))
        self.wait(2)
