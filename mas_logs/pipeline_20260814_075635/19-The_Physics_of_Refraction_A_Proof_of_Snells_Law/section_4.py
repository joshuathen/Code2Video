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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Apply Fermat’s Principle: minimize total time.",
            "Set the derivative to zero.",
            "Result: n1 sin theta1 = n2 sin theta2.",
            "This confirms the law of refraction.",
            "Light chooses the fastest path."
        ]
        self.setup_layout("Deriving Snell’s Law via Fermat's Principle", lecture_lines)
        
        # Define objects
        math_eq = MathTex(r"\frac{dT}{dx} = \frac{\sin\theta_1}{v_1} - \frac{\sin\theta_2}{v_2} = 0", font_size=36)
        snell_law = MathTex(r"n_1 \sin\theta_1 = n_2 \sin\theta_2", font_size=40, color=YELLOW)
        box = SurroundingRectangle(snell_law, color=YELLOW, buff=0.2)
        
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        
        self.place_at_grid(laser, "A4", scale_factor=0.5)
        self.place_at_grid(prism, "F4", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(math_eq, "D3", scale_factor=0.85)
        self.play(FadeIn(laser), Write(math_eq))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(Indicate(math_eq))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(ReplacementTransform(math_eq, snell_law), FadeOut(laser))
        self.play(Create(box))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(GREEN))
        self.play(Indicate(VGroup(snell_law, box)))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(TEAL))
        self.play(FadeIn(prism))
        self.play(Flash(snell_law, color=WHITE))
        self.wait(1)
