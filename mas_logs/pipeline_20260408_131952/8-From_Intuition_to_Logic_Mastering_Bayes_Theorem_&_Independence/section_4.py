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
        title_text = "Step-by-Step Application: The Medical Test Paradox"
        lecture_lines = [
            'Rare events make high-accuracy tests very counter-intuitive.',
            'Most positive results come from the healthy population.',
            'Tree diagrams help map true versus false positives.',
            'Bayesian logic explains why rare diseases have low certainty.',
            'Always consider base rates before interpreting individual results.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define Colors
        COLOR_ROOT = "#ECF0F1"
        COLOR_SICK = "#3498DB"
        COLOR_HEALTHY = "#27AE60"
        COLOR_POS = "#E74C3C"
        COLOR_CIRCLE = "#F1C40F"
        
        # Load Asset
        PENG_PATH = "/mmfs1/data/home/jthen/Code2Video/assets/icon/peng.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        root_svg = SVGMobject(PENG_PATH).set_color(COLOR_ROOT).scale(0.3)
        root_txt = Text("1000 Penguins", font_size=24, color=COLOR_ROOT)
        root_content = VGroup(root_svg, root_txt).arrange(RIGHT, buff=0.2)
        root_box = SurroundingRectangle(root_content, color=COLOR_ROOT, buff=0.1)
        root_group = VGroup(root_box, root_content)
        
        # Fix Issue 37: Place root_group in A2-A3 with scale 0.7
        self.place_in_area(root_group, "A2", "A3", scale_factor=0.7)
        
        self.play(Create(root_box), FadeIn(root_content), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        sick_svg = SVGMobject(PENG_PATH).set_color(COLOR_SICK).scale(0.25)
        sick_txt = Text("1 Sick", font_size=22, color=COLOR_SICK)
        sick_node = VGroup(sick_svg, sick_txt).arrange(RIGHT, buff=0.2)
        
        healthy_svg = SVGMobject(PENG_PATH).set_color(COLOR_HEALTHY).scale(0.25)
        healthy_txt = Text("999 Healthy", font_size=22, color=COLOR_HEALTHY)
        healthy_node = VGroup(healthy_svg, healthy_txt).arrange(RIGHT, buff=0.2)
        
        self.place_at_grid(sick_node, "C2", scale_factor=0.85)
        # Fix Issue 38: Place healthy_node at C4 with scale 0.8
        self.place_at_grid(healthy_node, "C4", scale_factor=0.8)
        
        # Branches from beneath the root box (row B) to nodes (row C)
        branch_sick = Line(self.grid["B2"], self.grid["C2"], color=COLOR_SICK)
        branch_healthy = Line(self.grid["B3"], self.grid["C4"], color=COLOR_HEALTHY)
        
        self.play(
            Create(branch_sick), Create(branch_healthy),
            FadeIn(sick_node), FadeIn(healthy_node),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        pos_true = Text("1 Positive Test", font_size=20, color=COLOR_POS)
        pos_false = Text("10 Positive Tests", font_size=20, color=COLOR_POS)
        
        self.place_at_grid(pos_true, "E2", scale_factor=0.85)
        # Fix Issue 39: Place pos_false at E4 with scale 0.8
        self.place_at_grid(pos_false, "E4", scale_factor=0.8)
        
        # Vertical branches from row D to row E
        link_pos_true = Line(self.grid["D2"], self.grid["E2"], color=COLOR_POS)
        link_pos_false = Line(self.grid["D4"], self.grid["E4"], color=COLOR_POS)
        
        self.play(
            Create(link_pos_true), Create(link_pos_false),
            FadeIn(pos_true), FadeIn(pos_false),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        circ1 = Circle(color=COLOR_CIRCLE, stroke_width=4).scale(0.8).move_to(pos_true)
        circ2 = Circle(color=COLOR_CIRCLE, stroke_width=4).scale(0.8).move_to(pos_false)
        
        self.play(Create(circ1), Create(circ2), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        calc_txt = Text("1 / (1 + 10) ≈ 9%", font_size=32, color=WHITE)
        self.place_in_area(calc_txt, "F2", "F4", scale_factor=0.9)
        
        self.play(Write(calc_txt), run_time=1.5)
        self.wait(2)
