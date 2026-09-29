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
        self.setup_layout("Conclusion and Intuition", [
            "Simple integer ratios reconstruct transcendental constants.",
            "Convergence maps the journey to pi.",
            "Fractal adjustments refine our mathematical reach."
        ])
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#87CEEB")
        circle = Circle(radius=1.0, color=WHITE)
        self.place_at_grid(circle, 'D3', scale_factor=0.8)
        self.place_at_grid(compass, 'D2', scale_factor=0.5)
        self.play(Create(circle), FadeIn(compass))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF7F50")
        
        # Illustrate convergence/infinite factors
        factors = VGroup()
        for i in range(1, 6):
            frac = MathTex(r"\frac{" + str(2*i) + "}{" + str(2*i-1) + "}", font_size=20)
            factors.add(frac)
        factors.arrange(RIGHT, buff=0.2)
        self.place_in_area(factors, 'E2', 'E5', scale_factor=0.8)
        self.play(FadeIn(factors))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        
        result = MathTex(r"\frac{\pi}{2} \approx \prod_{n=1}^{\infty} \frac{2n}{2n-1} \cdot \frac{2n}{2n+1}", font_size=24, color="#FFD700")
        self.place_at_grid(result, 'C3', scale_factor=1.0)
        self.place_at_grid(protractor, 'C6', scale_factor=0.5)
        self.play(Write(result), FadeIn(protractor))
        self.wait(2)
