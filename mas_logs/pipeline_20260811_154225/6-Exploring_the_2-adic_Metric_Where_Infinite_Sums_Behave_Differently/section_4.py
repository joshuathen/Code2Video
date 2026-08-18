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
            "Standard metrics diverge at infinity.",
            "2-adic metrics show convergence instead.",
            "The 'tail' vanishes in 2-adic space.",
            "Digital clocks show binary bit growth.",
            "Limit is -1 in 2-adic view."
        ]
        self.setup_layout("Divergence vs. Convergence: The Shifted Lens", lecture_lines)
        
        # Setup visual elements
        real_line = NumberLine(x_range=[-1, 5], length=4, color=BLUE)
        two_adic_line = NumberLine(x_range=[-2, 2], length=4, color=YELLOW)
        
        # Assets
        clock_real = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")
        clock_2adic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")
        
        self.place_at_grid(real_line, "A2", scale_factor=0.8)
        self.place_at_grid(two_adic_line, "D2", scale_factor=0.8)
        self.place_at_grid(clock_real, "A5", scale_factor=0.4)
        self.place_at_grid(clock_2adic, "D5", scale_factor=0.4)
        
        real_label = Text("Real (R)", font_size=20, color=BLUE).next_to(real_line, UP)
        two_adic_label = Text("2-adic (Q2)", font_size=20, color=YELLOW).next_to(two_adic_line, UP)
        self.add(real_label, two_adic_label)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        point = Dot(color=RED).move_to(real_line.n2p(0))
        self.add(point)
        self.play(point.animate.move_to(real_line.n2p(4.5)), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        point2 = Dot(color=RED).move_to(two_adic_line.n2p(0))
        self.add(point2)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        tail_text = Text("Tail vanishes", font_size=18, color=YELLOW)
        self.place_at_grid(tail_text, "D3", scale_factor=0.7)
        self.play(FadeIn(tail_text))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        binary_display = VGroup(*[Text(str(i), font_size=16) for i in range(8)])
        self.place_in_area(binary_display, "B4", "C6", scale_factor=0.6)
        self.play(Write(binary_display))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        self.play(point2.animate.move_to(two_adic_line.n2p(-1)), run_time=2)
