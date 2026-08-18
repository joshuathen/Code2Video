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
        self.setup_layout("The Chain Rule", [
            "Chain Rule differentiates composite functions f(g(x)).",
            "Multiply outer derivative by inner derivative.",
            "Derivative is f'(g(x)) times g'(x).",
            "Visualize gear ratios for change propagation.",
            "See how inner change affects outer change."
        ])

        # Formula
        formula = MathTex(r"\\frac{d}{dx} [f(g(x))] = f'(g(x)) \\cdot g'(x)", font_size=36)
        self.place_in_area(formula, 'B3', 'C4', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.play(FadeIn(formula))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#F39C12")
        # Highlight outer and inner
        outer_d = MathTex(r"f'(g(x))", color="#E74C3C", font_size=36)
        inner_d = MathTex(r"g'(x)", color="#2ECC71", font_size=36)
        self.place_at_grid(outer_d, 'D2', scale_factor=1.0)
        self.place_at_grid(inner_d, 'D4', scale_factor=1.0)
        self.play(FadeIn(outer_d), FadeIn(inner_d))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#F1C40F")
        self.play(Indicate(formula))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#3498DB")
        # Gear representation
        gear1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gears.svg", color="#F1C40F")
        gear2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gears.svg", color="#F1C40F")
        self.place_at_grid(gear1, 'E5', scale_factor=0.8)
        self.place_at_grid(gear2, 'E6', scale_factor=0.8)
        self.play(Create(gear1), Create(gear2))
        
        # Gear rotation
        gear1.add_updater(lambda m, dt: m.rotate(-dt * 2))
        gear2.add_updater(lambda m, dt: m.rotate(dt * 1.25))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#9B59B6")
        target = Dot(color="#27AE60", radius=0.2)
        self.place_at_grid(target, 'F5', scale_factor=0.7)
        self.add(target)
        self.play(Indicate(target))
        self.wait(1)
