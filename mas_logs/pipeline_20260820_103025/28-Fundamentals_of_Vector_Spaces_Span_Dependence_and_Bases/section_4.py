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
        self.setup_layout("Bases: The Minimal Blueprint", [
            "A basis is both linearly independent and spans.",
            "It is the minimal set of vectors.",
            "Bases describe every point in space uniquely."
        ])

        # Define assets
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # Define vectors
        b1 = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#FF5733")
        b2 = Arrow(start=ORIGIN, end=UP*1.5, color="#33FF57")
        l1 = MathTex("i", color="#FF5733").next_to(b1.get_end(), DOWN)
        l2 = MathTex("j", color="#33FF57").next_to(b2.get_end(), LEFT)
        basis_group = VGroup(b1, b2, l1, l2)
        
        v = Arrow(start=ORIGIN, end=RIGHT*1.0 + UP*1.0, color="#FFFF33")
        lv = MathTex("v", color="#FFFF33").next_to(v.get_end(), RIGHT)
        
        basis_label = Text("Basis", color="#32CD32")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#32CD32"))
        # Per VideoCritic (Issue 30/39): Fix basis_group position
        self.place_in_area(basis_group, 'A2', 'B3', scale_factor=0.6)
        self.play(Create(basis_group))
        self.play(FadeIn(grid_asset.scale(0.8).move_to(self.grid["C5"])))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#32CD32"))
        # Per VideoCritic (Issue 31/39): Fix v position
        self.place_at_grid(v, 'B4', scale_factor=0.7)
        self.place_at_grid(lv, 'B5', scale_factor=0.7)
        self.play(GrowArrow(v), Write(lv))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        # Per VideoCritic (Issue 32/39): Fix basis_label position
        self.place_at_grid(basis_label, 'F3', scale_factor=0.9)
        self.play(Write(basis_label))
        self.wait(2)
