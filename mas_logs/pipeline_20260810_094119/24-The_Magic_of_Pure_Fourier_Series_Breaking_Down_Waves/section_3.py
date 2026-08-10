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
        self.setup_layout("The Core Formula: The Fourier Recipe", [
            "The Fourier series is a mathematical recipe.", 
            "We sum sine and cosine components.", 
            "Weights determine the contribution of each.", 
            "Ingredients create a complex target shape.", 
            "Mixing waves builds the final function."
        ])
        
        # Load Assets
        bowl = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bowl.svg")
        oven = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/oven.svg")

        # Define Formula
        fourier_formula = MathTex(
            "f(t) = \\frac{a_0}{2} + \\sum_{n=1}^{\\infty} (a_n \\cos(nt) + b_n \\sin(nt))",
            font_size=32
        )
        # Applying layout fix 27
        self.place_in_area(fourier_formula, 'B2', 'B5', scale_factor=0.9)

        # Define component
        axes = Axes(x_range=[0, 6, 1], y_range=[-1, 1, 1], axis_config={"include_tip": False})
        wave = axes.plot(lambda x: np.sin(x), color="#9B59B6")
        component = VGroup(axes, wave)
        # Applying layout fixes 28 and 29
        self.place_in_area(component, 'C4', 'E6', scale_factor=0.55)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.place_at_grid(bowl, 'A3', scale_factor=0.3)
        self.play(Write(fourier_formula), FadeIn(bowl))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#E67E22")
        self.play(fourier_formula[0][7:9].animate.set_color("#E67E22"))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3498DB")
        self.play(
            fourier_formula[0][2:4].animate.set_color("#3498DB"),
            fourier_formula[0][11:13].animate.set_color("#3498DB"),
            fourier_formula[0][18:20].animate.set_color("#3498DB")
        )

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#9B59B6")
        self.play(Create(component))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN)
        self.place_at_grid(oven, 'F3', scale_factor=0.3)
        self.play(
            fourier_formula.animate.set_color(WHITE),
            component.animate.set_color(GREEN),
            FadeIn(oven)
        )
        self.wait(2)
