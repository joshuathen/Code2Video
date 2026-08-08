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
        self.setup_layout("Formalizing the Connection", [
            "Derivatives undo integration of functions.", 
            "This formalizes the inverse relationship.", 
            "They are perfect mathematical opposites."
        ])
        
        # Setup formula
        # d/dx [∫_{a}^{x} f(t)dt] = f(x)
        formula = MathTex(
            r"\frac{d}{dx}", 
            r"\left[", 
            r"\int_{a}^{x}", 
            r"f(t)", 
            r"dt", 
            r"\right]", 
            r"=", 
            r"f(x)"
        )
        self.place_at_grid(formula, 'C3', scale_factor=0.9)
        
        # Load dummy assets
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(icon1, 'B6', scale_factor=0.5)
        self.place_at_grid(icon2, 'E6', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(formula), FadeIn(icon1))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight d/dx (Green)
        self.play(formula[0].animate.set_color("#00FF00"))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)
        
        # Highlight Integral (Cyan)
        self.play(formula[2].animate.set_color("#00FFFF"))
        self.lecture[2].set_color("#00FFFF")
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Show substitution f(x) (Yellow)
        self.play(formula[7].animate.set_color("#FFCC00"))
        
        # Final box
        box = SurroundingRectangle(formula, color=WHITE, buff=0.2)
        self.place_in_area(box, 'C3', 'D4', scale_factor=0.9)
        self.play(Create(box), FadeIn(icon2))
        
        self.wait(2)
