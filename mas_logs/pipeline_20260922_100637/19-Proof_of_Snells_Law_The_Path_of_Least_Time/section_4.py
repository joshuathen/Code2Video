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
        self.setup_layout("Calculus Derivation", [
            "Differentiate T(x) with respect to x.",
            "Set the derivative to zero.",
            "Transition to sine components."
        ])
        
        # Define equations
        eq_t = MathTex(r"T(x) = \frac{\sqrt{h_1^2 + x^2}}{v_1} + \frac{\sqrt{h_2^2 + (w-x)^2}}{v_2}", font_size=32)
        eq_dt = MathTex(r"\frac{dT}{dx} = \frac{x}{v_1\sqrt{h_1^2 + x^2}} - \frac{w-x}{v_2\sqrt{h_2^2 + (w-x)^2}} = 0", font_size=32)
        eq_snell = MathTex(r"\frac{\sin \theta_1}{v_1} = \frac{\sin \theta_2}{v_2}", font_size=40)

        # Asset loading with fallback
        try:
            sine_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        except:
            sine_icon = Dot(color=BLUE)
        sine_icon.scale(0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF33FF")
        self.place_in_area(eq_t, 'B3', 'B6', scale_factor=0.8)
        self.play(Write(eq_t))
        self.play(Transform(eq_t, eq_dt))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF33FF")
        # Visualizing the stationary point (slope = 0)
        dot = Dot(color=YELLOW).move_to(self.grid['E3'])
        label = MathTex(r"\frac{dT}{dx} = 0", color=YELLOW, font_size=30).next_to(dot, RIGHT)
        self.play(FadeIn(dot), Write(label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF33FF")
        # Transition to final equation
        self.play(FadeOut(eq_t), FadeOut(dot), FadeOut(label))
        
        # Place sine icon
        self.place_at_grid(sine_icon, 'B3')
        self.play(FadeIn(sine_icon))
        
        self.place_in_area(eq_snell, 'C2', 'C5', scale_factor=1.2)
        self.play(Write(eq_snell))
        self.wait(2)
