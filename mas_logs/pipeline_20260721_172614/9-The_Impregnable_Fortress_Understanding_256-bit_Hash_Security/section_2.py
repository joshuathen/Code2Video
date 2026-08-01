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
        # TITLE and LECTURE LINES
        title_str = "Prerequisite: The Power of Binary Exponents"
        lecture_lines = [
            "Adding one bit doubles the total possible combinations.",
            "Binary exponents grow at a staggering exponential rate.",
            "256 bits provide a massive pool of unique values."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors
        WHITE_CLR = "#FFFFFF"
        BLUE_CLR = "#ADD8E6"
        GOLD_CLR = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # A white '0' and '1' (#FFFFFF) appear for a 1-bit system.
        self.play(self.lecture[0].animate.set_color(WHITE_CLR))
        
        zero_text = Text("0", font_size=60, color=WHITE_CLR)
        one_text = Text("1", font_size=60, color=WHITE_CLR)
        
        self.place_at_grid(zero_text, "B2")
        self.place_at_grid(one_text, "B5")
        
        bit_label = Text("1-bit = 2 outcomes", font_size=24, color=WHITE_CLR)
        self.place_in_area(bit_label, "A2", "A5")
        
        self.play(FadeIn(bit_label), Write(zero_text), Write(one_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A blue branching tree (#ADD8E6) quickly expands to represent exponential growth.
        self.play(self.lecture[1].animate.set_color(BLUE_CLR))
        
        # Branching tree
        # Root (Level 0) - Centered between A3 and A4
        root = Dot(color=BLUE_CLR)
        self.place_in_area(root, "A3", "A4")
        
        # Level 1
        l1_n1 = Dot(color=BLUE_CLR)
        l1_n2 = Dot(color=BLUE_CLR)
        self.place_at_grid(l1_n1, "B2")
        self.place_at_grid(l1_n2, "B5")
        
        e1_1 = Line(root.get_center(), l1_n1.get_center(), color=BLUE_CLR)
        e1_2 = Line(root.get_center(), l1_n2.get_center(), color=BLUE_CLR)
        
        # Level 2
        l2_n1 = Dot(color=BLUE_CLR)
        l2_n2 = Dot(color=BLUE_CLR)
        l2_n3 = Dot(color=BLUE_CLR)
        l2_n4 = Dot(color=BLUE_CLR)
        self.place_at_grid(l2_n1, "C1")
        self.place_at_grid(l2_n2, "C3")
        self.place_at_grid(l2_n3, "C4")
        self.place_at_grid(l2_n4, "C6")
        
        e2_1 = Line(l1_n1.get_center(), l2_n1.get_center(), color=BLUE_CLR)
        e2_2 = Line(l1_n1.get_center(), l2_n2.get_center(), color=BLUE_CLR)
        e2_3 = Line(l1_n2.get_center(), l2_n3.get_center(), color=BLUE_CLR)
        e2_4 = Line(l1_n2.get_center(), l2_n4.get_center(), color=BLUE_CLR)
        
        tree = VGroup(root, l1_n1, l1_n2, e1_1, e1_2, l2_n1, l2_n2, l2_n3, l2_n4, e2_1, e2_2, e2_3, e2_4)
        
        # Representation of exponential growth (more nodes)
        l3_nodes = VGroup(*[Dot(radius=0.04, color=BLUE_CLR) for _ in range(8)]).arrange(RIGHT, buff=0.1)
        self.place_in_area(l3_nodes, "D1", "D6")
        
        l4_nodes = VGroup(*[Dot(radius=0.02, color=BLUE_CLR) for _ in range(16)]).arrange(RIGHT, buff=0.05)
        self.place_in_area(l4_nodes, "E1", "E6")

        self.play(FadeOut(zero_text), FadeOut(one_text), FadeOut(bit_label))
        self.play(Create(tree), run_time=1.5)
        self.play(FadeIn(l3_nodes, lag_ratio=0.1), FadeIn(l4_nodes, lag_ratio=0.05), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A long row of 256 gold light switches [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg] (#FFD700) fades into the number '$2^{256}$'.
        self.play(self.lecture[2].animate.set_color(GOLD_CLR))
        
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg"
        
        switches = VGroup()
        for _ in range(12):
            sw = SVGMobject(asset_path, height=0.5)
            sw.set_color(GOLD_CLR)
            switches.add(sw)
        
        switches.arrange(RIGHT, buff=0.1)
        dots = Text("...", color=GOLD_CLR, font_size=36)
        sw_display = VGroup(switches, dots).arrange(RIGHT, buff=0.2)
        self.place_in_area(sw_display, "B1", "B6", scale_factor=0.8)
        
        bits_label = Text("256 Bits", font_size=30, color=GOLD_CLR)
        self.place_in_area(bits_label, "C1", "C6")
        
        huge_val = MathTex("2^{256}", color=GOLD_CLR, font_size=120)
        self.place_in_area(huge_val, "D1", "F6")

        self.play(FadeOut(tree), FadeOut(l3_nodes), FadeOut(l4_nodes))
        self.play(FadeIn(sw_display), FadeIn(bits_label))
        self.wait(1.5)
        self.play(
            FadeOut(sw_display),
            FadeOut(bits_label),
            FadeIn(huge_val)
        )
        self.wait(2)
