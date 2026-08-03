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
        # Lecture lines
        lines = [
            "Let the small block have mass one.",
            "Let the large block have mass 100.",
            "Total bounces equal the first two digits of Pi.",
            "Increasing the mass ratio reveals more digits.",
            "This simple physical system somehow calculates Pi."
        ]
        
        self.setup_layout("The Phenomenon: The Digits of Pi", lines)
        
        # Colors for each stage
        C1 = "#0000FF" # Blue
        C2 = "#FFFF00" # Yellow
        C3 = "#00FF00" # Green
        C4 = "#DA70D6" # Orchid
        C5 = "#00FF00" # Green
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(C1))
        
        # Asset: block.svg
        small_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        small_block.set_color(C1)
        self.place_at_grid(small_block, "A2", scale_factor=0.5)
        
        m_label = Text("m=1", font_size=20, color=C1)
        self.place_at_grid(m_label, "B2", scale_factor=1.0)
        
        self.play(FadeIn(small_block), Write(m_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(C2))
        
        large_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        large_block.set_color(C2)
        self.place_at_grid(large_block, "A5", scale_factor=0.8)
        
        M_label = Text("M=100", font_size=20, color=C2)
        self.place_at_grid(M_label, "B5", scale_factor=1.0)
        
        self.play(FadeIn(large_block), Write(M_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(C3))
        
        # Collision counter
        counter_label = Text("Pi Digits:", font_size=22, color=C3)
        self.place_in_area(counter_label, "C3", "C4")
        
        # Use ValueTracker for the counter animation
        counter_tracker = ValueTracker(0)
        counter_num = DecimalNumber(0, num_decimal_places=0, color=C3)
        counter_num.add_updater(lambda d: d.set_value(counter_tracker.get_value()))
        self.place_at_grid(counter_num, "C5")
        
        self.play(FadeIn(counter_label), FadeIn(counter_num))
        self.play(counter_tracker.animate.set_value(31), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transition to table
        self.play(
            FadeOut(small_block, m_label, large_block, M_label, counter_label, counter_num),
            self.lecture[3].animate.set_color(C4)
        )
        
        # Table Headers
        h1 = Text("Mass Ratio", font_size=22, color=C4)
        h2 = Text("Collisions", font_size=22, color=C4)
        h3 = Text("Pi Digits", font_size=22, color=C4)
        
        self.place_in_area(h1, 'B1', 'B2', scale_factor=0.9)
        self.place_in_area(h2, 'B3', 'B4', scale_factor=0.9)
        self.place_in_area(h3, 'B5', 'B6', scale_factor=0.9)
        
        # Data Rows
        # Row 1: M=1
        r1_m = Text("1", font_size=22, color=WHITE)
        r1_c = Text("3", font_size=22, color=WHITE)
        r1_p = Text("3", font_size=22, color=WHITE)
        self.place_in_area(r1_m, 'C1', 'C2')
        self.place_in_area(r1_c, 'C3', 'C4')
        self.place_in_area(r1_p, 'C5', 'C6')
        
        # Row 2: M=100
        r2_m = Text("100", font_size=22, color=WHITE)
        r2_c = Text("31", font_size=22, color=WHITE)
        r2_p = Text("3.1", font_size=22, color=WHITE)
        self.place_in_area(r2_m, 'D1', 'D2')
        self.place_in_area(r2_c, 'D3', 'D4')
        self.place_in_area(r2_p, 'D5', 'D6')
        
        # Row 3: M=10,000
        r3_m = Text("10,000", font_size=22, color=WHITE)
        r3_c = Text("314", font_size=22, color=WHITE)
        r3_p = Text("3.14", font_size=22, color=WHITE)
        self.place_in_area(r3_m, 'E1', 'E2')
        self.place_in_area(r3_c, 'E3', 'E4')
        self.place_in_area(r3_p, 'E5', 'E6')
        
        self.play(FadeIn(VGroup(h1, h2, h3)))
        self.play(FadeIn(VGroup(r1_m, r1_c, r1_p, r2_m, r2_c, r2_p, r3_m, r3_c, r3_p)))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(C5))
        
        # Highlights: making collisions and pi digits green
        highlights = VGroup(r1_c, r1_p, r2_c, r2_p, r3_c, r3_p)
        self.play(highlights.animate.set_color(C5))
        self.wait(2)
