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
        self.setup_layout("The Redundancy Test: Linear Dependence", [
            "Linear dependence means redundant, wasteful vectors.",
            "One vector can be formed by others.",
            "Example: A third vector adds no reach."
        ])

        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color=WHITE)
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg", color=WHITE)
        
        # Define vectors
        v1 = Vector(RIGHT, color=BLUE)
        v3 = Vector(1.5 * RIGHT, color=GREEN)
        self.place_at_grid(v1, 'D2')
        self.place_at_grid(v3, 'D3')

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(ruler, 'D1', scale_factor=0.3)
        self.play(Create(v1), Create(v3), FadeIn(ruler))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FF5555"))
        label = MathTex(r"v_3 = c_1 v_1", color="#FF5555")
        self.place_in_area(label, 'B4', 'B5', scale_factor=0.7)
        self.place_at_grid(pencil, 'C4', scale_factor=0.3)
        self.play(Write(label), FadeIn(pencil))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        redundant = Text("Redundant", color=RED, font_size=24)
        self.place_at_grid(redundant, 'B2', scale_factor=0.6)
        self.play(Write(redundant))
        
        equation_group = VGroup(label, pencil)
        self.place_in_area(equation_group, 'C4', 'C6', scale_factor=0.65)
        
        self.wait(1)
        self.play(FadeOut(v3), FadeOut(equation_group), FadeOut(redundant), FadeOut(ruler))
        self.wait(1)
