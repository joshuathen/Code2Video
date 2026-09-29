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
            "Uncertainty follows Heisenberg-Gabor inequality.",
            "Signal product has a minimum floor.",
            "We cannot shrink both windows simultaneously.",
            "Pushing time forces frequency expansion.",
            "Wave math limits precision boundaries."
        ]
        self.setup_layout("The Mathematical Bound", lecture_lines)
        
        # Define elements
        ineq_symbol = MathTex(r"\geq", color="#E0FFFF")
        formula = MathTex(r"\Delta t \cdot \Delta f \geq \frac{1}{4\pi}", color="#FFFFFF")
        
        # Load Assets
        stopwatch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stopwatch.svg")
        tuningfork = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tuningfork.svg")
        
        # Apply layout fixes from Critic
        self.place_in_area(ineq_symbol, 'B2', 'B4', scale_factor=2.5)
        self.place_in_area(formula, 'D2', 'D5', scale_factor=1.5)
        self.place_at_grid(stopwatch, 'C2', scale_factor=0.5)
        self.place_at_grid(tuningfork, 'C5', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(ineq_symbol), self.lecture[0].animate.set_color("#E0FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(formula), FadeIn(stopwatch), FadeIn(tuningfork), self.lecture[1].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF6347"))
        self.play(Indicate(formula, color=WHITE))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#98FB98"))
        self.wait(2)
