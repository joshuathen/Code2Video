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
            "Mathematically, the derivative is a limit.", 
            "Limit h approaches zero of the slope formula.", 
            "f prime of x equals the limit.", 
            "This finds the exact velocity at time t.", 
            "It provides precision for every changing rate."
        ]
        self.setup_layout("Formalizing the Derivative", lecture_lines)
        
        # Assets
        stopwatch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stopwatch.svg")
        odometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/odometer.svg")
        
        # Prepare objects
        # Definition: f'(x) = lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
        formula = MathTex(
            "f'(x)", "=", "\\lim_{h \\to 0}", "\\frac{f(x+h) - f(x)}{h}"
        )
        self.place_in_area(formula, "B2", "C6", scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(stopwatch, "A4", scale_factor=0.5)
        self.play(FadeIn(formula[0:3]), FadeIn(stopwatch))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(formula[3]))
        self.lecture[1].set_color("#FF00FF") # Highlight h->0 logic
        self.play(Indicate(formula[2]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(odometer, "D4", scale_factor=0.5)
        self.lecture[2].set_color("#00FF00") # Highlight f'(x)
        self.play(formula[0].animate.set_color("#00FF00"), FadeIn(odometer))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        vel_text = Text("Exact velocity at time t", font_size=20, color="#ADD8E6")
        self.place_at_grid(vel_text, "E4", scale_factor=0.85)
        self.play(Write(vel_text))
        self.lecture[3].set_color("#ADD8E6")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFA500")
        self.play(Flash(formula, color="#FFA500"))
        self.wait(2)
