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
        lecture_lines = [
            "Small disks move every other turn.",
            "Middle disks move every four turns.",
            "Large disks move every eight turns.",
            "Smallest disk moves at every odd step.",
            "Movement follows a perfect recursive pattern."
        ]
        self.setup_layout("The Algorithmic Logic", lecture_lines)

        # Formula Mobjects
        formula = MathTex("T(n) = 2T(n-1) + 1", font_size=48)
        formula.set_color_by_tex("2", "#32CD32")
        formula.set_color_by_tex("1", "#32CD32")
        
        disk_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        formula_label = VGroup(formula, disk_icon).arrange(DOWN)

        # Branching tree visualization
        tree = VGroup(
            Circle(radius=0.3, color=BLUE).move_to(ORIGIN),
            Line(ORIGIN, UP*1 + LEFT*1.5),
            Line(ORIGIN, UP*1 + RIGHT*1.5),
            Circle(radius=0.2, color=BLUE).move_to(UP*1 + LEFT*1.5),
            Circle(radius=0.2, color=BLUE).move_to(UP*1 + RIGHT*1.5)
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_in_area(formula, 'A2', 'A5', scale_factor=0.6)
        self.play(Write(formula))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_in_area(tree, 'C2', 'E5', scale_factor=0.5)
        self.play(FadeIn(tree))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(tree.animate.shift(DOWN*0.5).scale(1.2))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.place_at_grid(formula_label, 'B3', scale_factor=0.7)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(FadeOut(formula), FadeOut(tree), FadeOut(formula_label))
