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
        self.setup_layout("Summary and Conclusion", [
            "Newton's method reveals deep fractal beauty.", 
            "Complex dynamics create chaotic, stunning boundaries.", 
            "Mathematics transforms simple roots into infinite art."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using asset reference: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Placeholder for icon if needed, but per prompt only standard shapes/text are expected.
        formula = MathTex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}", color=BLUE)
        self.place_in_area(formula, 'B4', 'C6', scale_factor=0.9)
        self.play(Write(formula))
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Representing the fractal output (placeholder shape)
        fractal_placeholder = Circle(radius=1.5, color=GREEN_B, fill_opacity=0.5)
        self.place_at_grid(fractal_placeholder, 'D5', scale_factor=0.7)
        self.play(FadeIn(fractal_placeholder))
        self.play(self.lecture[1].animate.set_color(GREEN_B))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(1)
        
        # Final fade out (using asset ref technically as per storyboard)
        # Using asset placeholder: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        self.play(FadeOut(formula), FadeOut(fractal_placeholder), FadeOut(self.lecture), FadeOut(self.title))
        self.wait(1)
