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
        self.setup_layout("Defining the Derivative", ["The derivative is a limit.", "Let the time interval shrink to zero.", "This gives us the instantaneous slope."])
        
        # Load asset
        stopwatch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stopwatch.svg")
        self.place_at_grid(stopwatch, 'B6', scale_factor=0.3)
        
        # Create formula
        formula = MathTex(
            "f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}",
            font_size=48
        )
        self.place_in_area(formula, 'B2', 'D4', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(stopwatch), Write(formula))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00BFFF")
        
        # Labels for components
        label1 = Text("Difference quotient", font_size=20, color="#00BFFF")
        self.place_at_grid(label1, 'E3', scale_factor=0.7)
        
        self.play(FadeIn(label1))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        
        # Highlight h->0
        box = SurroundingRectangle(formula[0][6:12], color="#FF4500", buff=0.1)
        label2 = Text("Limit as h approaches 0", font_size=20, color="#FF4500")
        self.place_at_grid(label2, 'E4', scale_factor=0.7)

        self.play(Create(box), Write(label2))
        self.wait(2)
