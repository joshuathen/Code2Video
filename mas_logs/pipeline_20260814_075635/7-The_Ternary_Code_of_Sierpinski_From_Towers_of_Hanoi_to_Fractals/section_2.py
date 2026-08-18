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
        lecture_lines = [
            "Constrained Hanoi limits moves to adjacent pegs.",
            "Moves A to B are zero.",
            "Moves B to C are one.",
            "Illegal moves are represented by two.",
            "Legal paths exclude the digit two."
        ]
        self.setup_layout("The Constrained Towers of Hanoi", lecture_lines)
        
        # Define Pegs and Assets
        # Using SVG file for disks
        disk_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disks.svg"
        peg_a = SVGMobject(disk_asset).set_color(WHITE)
        peg_b = SVGMobject(disk_asset).set_color(WHITE)
        peg_c = SVGMobject(disk_asset).set_color(WHITE)
        
        label_a = Text("A", font_size=20)
        label_b = Text("B", font_size=20)
        label_c = Text("C", font_size=20)
        
        # Applying requested layout changes
        self.place_at_grid(peg_a, "B2", scale_factor=0.8)
        self.place_at_grid(peg_b, "D3", scale_factor=1.0)
        self.place_at_grid(peg_c, "B4", scale_factor=0.8)
        
        label_a.next_to(peg_a, DOWN)
        label_b.next_to(peg_b, DOWN)
        label_c.next_to(peg_c, DOWN)
        
        pegs = VGroup(peg_a, peg_b, peg_c, label_a, label_b, label_c)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(pegs))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        line_ab = Line(peg_a.get_center(), peg_b.get_center(), color=GREEN)
        arrow_ab = Arrow(start=peg_a.get_center(), end=peg_b.get_center(), color=GREEN)
        label_0 = Text("0 (A->B)", font_size=20, color=GREEN).next_to(line_ab.point_from_proportion(0.5), UP)
        self.play(Create(arrow_ab), Write(label_0))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        line_bc = Line(peg_b.get_center(), peg_c.get_center(), color=GREEN)
        arrow_bc = Arrow(start=peg_b.get_center(), end=peg_c.get_center(), color=GREEN)
        label_1 = Text("1 (B->C)", font_size=20, color=GREEN).next_to(line_bc.point_from_proportion(0.5), UP)
        self.play(Create(arrow_bc), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        arrow_ac = ArcBetweenPoints(peg_a.get_center(), peg_c.get_center(), angle=-TAU/6, color=RED)
        label_2 = Text("2 (A->C)", font_size=20, color=RED).next_to(arrow_ac.point_from_proportion(0.5), UP)
        self.play(Create(arrow_ac), Write(label_2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        cross = Cross(arrow_ac, stroke_color=RED)
        self.play(Create(cross))
        self.wait(2)
