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
        lecture_lines = ["We traveled from average to instantaneous.", "The derivative acts as a mathematical zoom.", "It gives us precision at a single point."]
        self.setup_layout("Conclusion and Summary", lecture_lines)
        
        # Define visuals
        recap_text = Text("Secant (Avg) \u2192 Tangent (Inst.)", font_size=28, color=BLUE)
        
        # Note: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg] is referenced. 
        # Using a fallback since the path points to '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg'
        zoom_lens = Circle(radius=1.0, color=YELLOW, stroke_width=4)
        zoom_text = Text("Zoom", font_size=24, color=YELLOW).next_to(zoom_lens, DOWN)
        zoom_group = VGroup(zoom_lens, zoom_text)
        
        deriv_formula = MathTex(r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}", font_size=36, color=TEAL)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(recap_text, 'A2', 'B4', scale_factor=0.9)
        self.play(FadeIn(recap_text))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(zoom_group, 'C4', scale_factor=0.6)
        self.play(FadeIn(zoom_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(TEAL))
        self.place_at_grid(deriv_formula, 'D4', scale_factor=0.8)
        self.play(Write(deriv_formula))
        
        self.wait(2)
