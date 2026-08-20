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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Introducing the 2-adic Metric", [
            "The 2-adic metric uses divisibility by two.", 
            "High powers of two make numbers small.", 
            "Binary digits determine 2-adic proximity."
        ])
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        calc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        
        # === Animation for Lecture Line 1 ===
        norm_text = MathTex(r"|x|_2 = 2^{-v_2(x)}", color=WHITE)
        self.place_in_area(norm_text, 'A2', 'A5', scale_factor=1.1)
        self.place_at_grid(ruler, 'A1', scale_factor=0.3)
        self.play(Write(norm_text), FadeIn(ruler))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        val_text = MathTex(r"v_2(8) = v_2(2^3) = 3", color="#FFFF00")
        self.place_in_area(val_text, 'B2', 'B5', scale_factor=0.9)
        self.play(Write(val_text))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        dist_text = MathTex(r"d_2(x, y) = |x - y|_2", color="#FF00FF")
        self.place_in_area(dist_text, 'C2', 'C5', scale_factor=0.9)
        self.place_at_grid(calc, 'C6', scale_factor=0.3)
        self.play(Write(dist_text), FadeIn(calc))
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.wait(2)
