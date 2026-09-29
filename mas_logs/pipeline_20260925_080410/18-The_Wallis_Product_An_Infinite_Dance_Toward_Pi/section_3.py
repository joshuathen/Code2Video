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
        lecture_lines = ["Substitute integrals into our ratio formula.", "Isolate pi over two through algebra.", "The beautiful Wallis product emerges.", "Each term refines our value.", "It is an infinite product."]
        self.setup_layout("Deriving the Wallis Product", lecture_lines)
        
        # Assets
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        note_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/notebook.svg")
        abacus_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg")
        
        # Define mobjects
        i_even = Tex("$I_{2n} = \\frac{(2n-1)!!}{(2n)!!} \\cdot \\frac{\\pi}{2}$", color=WHITE)
        i_odd = Tex("$I_{2n+1} = \\frac{(2n)!!}{(2n+1)!!}$", color=WHITE)
        wallis = Tex("$\\frac{\\pi}{2} = \\prod_{n=1}^{\\infty} \\left( \\frac{2n}{2n-1} \\cdot \\frac{2n}{2n+1} \\right)$", color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(i_even, 'C2', 'D3', scale_factor=0.7)
        self.place_in_area(i_odd, 'E2', 'F3', scale_factor=0.7)
        self.place_at_grid(calc_icon, "C1", scale_factor=0.5)
        self.place_at_grid(note_icon, "E1", scale_factor=0.5)
        self.play(Write(i_even), FadeIn(calc_icon), Write(i_odd), FadeIn(note_icon))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.place_in_area(wallis, 'C4', 'E6', scale_factor=0.9)
        self.place_at_grid(abacus_icon, "B4", scale_factor=0.5)
        self.play(FadeIn(wallis), FadeIn(abacus_icon))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # === Animation for Lecture Line 4 ===
        box = SurroundingRectangle(wallis, color=BLUE)
        self.play(Create(box))
        self.play(self.lecture[3].animate.set_color("#87CEEB"))
        
        # === Animation for Lecture Line 5 ===
        self.play(FadeOut(box))
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        self.wait(2)
